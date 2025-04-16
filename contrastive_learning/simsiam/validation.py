# https://github.com/zhirongw/lemniscate.pytorch/blob/master/test.py
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
from torch import nn
from utils.dataset_modules.image_dataset import ImageDataset


class KNNValidation(object):
    def __init__(self, dataset, batch_size, model, feat_dim, device, K=1):
        self.dataset = dataset
        self.model = model
        self.device = device
        self.feat_dim = feat_dim
        self.K = K
        if dataset == 'cifar10':
            base_transforms = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        elif dataset == 'tiny_imagenet':
            base_transforms = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
        if dataset == 'cifar10':
            train_dataset = datasets.CIFAR10(root='/workspace/Dataset',
                                             train=True,
                                             download=True,
                                             transform=base_transforms)
        elif dataset == 'tiny_imagenet':
            print(dataset)
            train_dataset = ImageDataset(root='/workspace/Dataset/tiny-imagenet-200',
                                         train=True,
                                         transform=base_transforms)
        else:
            assert False, 'invalid dataset'

        self.train_dataloader = DataLoader(train_dataset,
                                           batch_size=batch_size,
                                           shuffle=False,
                                           num_workers=10,
                                           pin_memory=True,
                                           drop_last=True)

        if dataset == 'cifar10':
            val_dataset = datasets.CIFAR10(root='/workspace/Dataset',
                                           train=False,
                                           download=True,
                                           transform=base_transforms)
        elif dataset == 'tiny_imagenet':
            print(dataset)
            val_dataset = ImageDataset(root='/workspace/Dataset/tiny-imagenet-200',
                                       train=False,
                                       transform=base_transforms)
        else:
            assert False, 'invalid dataset'

        self.val_dataloader = DataLoader(val_dataset,
                                         batch_size=batch_size,
                                         shuffle=False,
                                         num_workers=10,
                                         pin_memory=True,
                                         drop_last=True)

    def _topk_retrieval(self):
        """Extract features from validation split and search on train split features."""
        if self.dataset == 'cifar10':
            n_data = self.train_dataloader.dataset.data.shape[0]
        elif self.dataset == 'tiny_imagenet':
            n_data = len(self.train_dataloader.dataset)
        else:
            assert False, 'invalid dataset'
        feat_dim = self.feat_dim

        self.model.eval()
        if 'cuda' in self.device:
            torch.cuda.empty_cache()

        train_features = torch.zeros([feat_dim, n_data], device=self.device)
        with torch.no_grad():
            for batch_idx, (inputs, _) in enumerate(self.train_dataloader):
                inputs = inputs.to(self.device)
                batch_size = inputs.size(0)

                # forward
                features = self.model(inputs)
                features = nn.functional.normalize(features)
                train_features[:, batch_idx * batch_size:batch_idx * batch_size + batch_size] = features.data.t()

            train_labels = torch.LongTensor(self.train_dataloader.dataset.targets).cuda()

        total = 0
        correct = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.val_dataloader):
                targets = targets.cuda(non_blocking=True)
                batch_size = inputs.size(0)
                features = self.model(inputs.to(self.device))

                dist = torch.mm(features, train_features)
                yd, yi = dist.topk(self.K, dim=1, largest=True, sorted=True)
                candidates = train_labels.view(1, -1).expand(batch_size, -1)
                retrieval = torch.gather(candidates, 1, yi)

                retrieval = retrieval.narrow(1, 0, 1).clone().view(-1)

                total += targets.size(0)
                correct += retrieval.eq(targets.data).sum().item()
        top1 = correct / total

        return top1

    def eval(self):
        return self._topk_retrieval()
