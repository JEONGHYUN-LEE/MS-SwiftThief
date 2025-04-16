import time
import math
from os import path, makedirs

from torch import optim
from torch.utils.data import DataLoader
from torch.backends import cudnn
from torchvision import transforms
from utils.dataset_modules.image_dataset import ImageDataset
from utils.configs import configs
from .simsiam.loader import TwoCropsTransform
from .simsiam.criterion import SimSiamLoss, SoftSupSimSiamLossV17, CL_FGSM, SimSiamLoss_cost_sensitive
from tqdm import tqdm
# from .soft_loss import *
from torch.utils import data
import torch.nn.functional as F
import torch
import random
import numpy as np
from utils.my_tensor_dataset import MyDataset

"""
This code is based on the pytorch implementation of SimSiam.
https://github.com/Reza-Safdari/SimSiam-91.9-top1-acc-on-CIFAR10
"""

seed = 0
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)


def main(args, labeled_dataset, model, victim_model, batch_size,
         feat_dim, num_unlabeled, start_epochs, total_epochs, init_lr, costs, device):
    ssl_transforms = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.RandomResizedCrop(32, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    # sub-sampling 10K images from the surrogate dataset as described in our paper
    unlabeled_dataset = ImageDataset(root='/workspace/datasets/imagenet32',
                                     train=True,
                                     transform=TwoCropsTransform(ssl_transforms))
    subset_idx = torch.randperm(1000000)[:num_unlabeled]
    unlabeled_dataset = torch.utils.data.Subset(unlabeled_dataset, subset_idx)

    # loader for unqueried dataset U (expected batch num = 10000 / 512 = 19)
    unlabeled_data_loader = DataLoader(dataset=unlabeled_dataset,
                                       batch_size=batch_size,
                                       shuffle=True,
                                       num_workers=10,
                                       pin_memory=False,
                                       persistent_workers=True,
                                       prefetch_factor=8,
                                       drop_last=True)

    optimizer = optim.SGD(model.parameters(),
                          lr=0.06,
                          momentum=0.9,
                          weight_decay=5e-4)

    criterion = SimSiamLoss('simplified')
    soft_criterion = SoftSupSimSiamLossV17(device, configs[args.victim_dataset]['num_classes'])
    cost_sensitive_criterion = SimSiamLoss_cost_sensitive(costs)
    torch.cuda.set_device(int(device[-1]))
    model = model.cuda(int(device[-1]))
    victim_model = victim_model.cuda(int(device[-1]))
    criterion = criterion.cuda(int(device[-1]))
    cudnn.benchmark = False

    # Be Careful: only labeled dataset can be forwarded to the victim model
    # forward and get logit outputs (here is the only space in this file that the victim model is used)
    # ========================================  start  ========================================

    # loader for queried dataset Q
    labeled_data_loader = DataLoader(dataset=labeled_dataset,
                                     batch_size=100,
                                     shuffle=False,
                                     num_workers=10,
                                     drop_last=False)

    logits = []
    victim_model.eval()
    for x in labeled_data_loader:
        x = x.to(device)
        logit = victim_model(x)
        logits.append(logit.detach().cpu())
    logits = torch.cat(logits)

    sl_transforms = transforms.Compose([
        transforms.Normalize((0.0, 0.0, 0.0), (1 / 0.229, 1 / 0.224, 1 / 0.225)),
        transforms.Normalize((-0.485, -0.456, -0.406), (1.0, 1.0, 1.0)),
        transforms.ToPILImage(),
        transforms.RandomResizedCrop((32, 32), scale=(0.2, 1.), antialias=True),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])


    labeled_train_set = MyDataset(data=labeled_dataset, target=logits, transform=TwoCropsTransform(sl_transforms))

    # loader for queried dataset Q
    labeled_data_loader = DataLoader(dataset=labeled_train_set,
                                     batch_size=256,
                                     shuffle=True,
                                     num_workers=10,
                                     pin_memory=False,
                                     persistent_workers=True,
                                     prefetch_factor=8,
                                     drop_last=False)

    # ========================================  end  ========================================

    start_epoch = 1
    # routine

    print("Contrastive Learning Start")
    print("# unlabeled dataset: {}".format(len(unlabeled_data_loader.dataset)))
    print("# labeled dataset: {}".format(len(labeled_data_loader.dataset)))

    for epoch in tqdm(range(start_epoch, total_epochs)):
        adjust_learning_rate(optimizer, start_epochs + epoch, total_epochs, init_lr)
        print("Training...")

        # train for one epoch
        train_loss = train(args=args,
                           unlabeled_data_loader=unlabeled_data_loader,
                           labeled_data_loader=labeled_data_loader,
                           model=model,
                           criterion=criterion,
                           soft_criterion=soft_criterion,
                           cost_sensitive_criterion=cost_sensitive_criterion,
                           optimizer=optimizer,
                           epoch=epoch,
                           device=device)


def train(args, unlabeled_data_loader, labeled_data_loader, model,
          criterion, soft_criterion, cost_sensitive_criterion, optimizer, epoch, device):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(
        len(unlabeled_data_loader),
        [batch_time, losses, optimizer.param_groups[0]['lr']],
        prefix="Epoch: [{}]".format(epoch))
    end = time.time()
    # switch to train mode
    model.train()

    for i, (images, _) in enumerate(unlabeled_data_loader):
        try:
            (x1, x2), y = next(labeled_data_iter)
        except:
            labeled_data_iter = iter(labeled_data_loader)
            (x1, x2), y = next(labeled_data_iter)

        # input pairs for the self-supervised contrastive learning (SimSiam)
        images[0] = images[0].to(device)
        images[1] = images[1].to(device)

        # input pairs for the soft-supervised contrastive learning (Sup Con, with label)
        x1 = x1.to(device)
        x2 = x2.to(device)
        y = y.to(device)

        reg_adversary = CL_FGSM(model, 0.01, device)

        adv_x1 = reg_adversary(x1, x2)

        model.train()
        # output for the self-supervised contrastive learning (SimSiam)
        outs_unlabeled = model(im_aug1=images[0], im_aug2=images[1])

        # output for the soft-supervised contrastive learning (Sup Con, with label)
        outs_labeled = model(im_aug1=x1, im_aug2=x2)

        # loss for the self-supervised contrastive learning (SimSiam form)
        loss1 = criterion(outs_unlabeled['z1'], outs_unlabeled['z2'], outs_unlabeled['p1'], outs_unlabeled['p2'])

        # loss for the soft-supervised contrastive learning (SimSiam form)
        loss2 = soft_criterion(p=torch.cat([outs_labeled['p1'], outs_labeled['p2']], dim=0),
                               z=torch.cat([outs_labeled['z1'], outs_labeled['z2']], dim=0),
                               targets=torch.cat([(y / 1.0).softmax(dim=1), (y / 1.0).softmax(dim=1)], dim=0))

        # output for the representation sharpness minimization (SimSiam form)
        outs_labeled = model(im_aug1=adv_x1, im_aug2=x2)

        # loss for the representation sharpness minimization (SimSiam form)
        loss3 = cost_sensitive_criterion(outs_labeled['z1'], outs_labeled['z2'],
                                         outs_labeled['p1'], outs_labeled['p2'], y.argmax(dim=1))

        # loss for the soft-supervised contrastive learning (Sup Con)

        loss = loss1 + loss2 + 0.01*loss3
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        losses.update(loss.item(), x1.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 10 == 0:
            progress.display(i)

    return losses.avg


def adjust_learning_rate(optimizer, epoch, total_epochs, init_lr):
    """Decay the learning rate based on schedule"""
    lr = init_lr
    # cosine lr schedule
    lr *= 0.5 * (1. + math.cos(math.pi * epoch / 800))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def save_checkpoint(epoch, model, optimizer, acc, filename, msg, device):
    # state = {
    #     'epoch': epoch,
    #     'arch': args.arch,
    #     'state_dict': model.state_dict(),
    #     'optimizer': optimizer.state_dict(),
    #     'top1_acc': acc
    # }
    print(msg)
    model.cpu()
    state = model.module.state_dict()
    save_dir, file = path.split(filename)
    save_dir = path.join(save_dir, str(epoch))
    makedirs(save_dir, exist_ok=True)
    torch.save(state, path.join(save_dir, file))
    model.to(device)
    print(msg)


def load_checkpoint(model, optimizer, filename):
    checkpoint = torch.load(filename, map_location='cuda:0')
    start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

    return start_epoch, model, optimizer


if __name__ == '__main__':
    main()
