import torch
import numpy as np
from tqdm import tqdm
from utils.configs import configs
from sklearn.neighbors import KernelDensity
import random

seed = 0
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)


def sampler(sampling_type, victim_dataset, unlabeled_dataset, labeled_dataset,
            model, victim_model, num_samples, device):
    # get indices

    if sampling_type == 'entropy':
        indices = entropy_based(model=model,
                                unlabeled_dataset=unlabeled_dataset,
                                device=device)
    elif sampling_type == 'imbalance_kde':  # prioritization rarely queried class
        indices = imbalance_aware_kde(victim_dataset=victim_dataset,
                                      model=model,
                                      victim_model=victim_model,
                                      labeled_dataset=labeled_dataset,
                                      unlabeled_dataset=unlabeled_dataset,
                                      device=device)
    elif sampling_type == 'random':
        indices = torch.randperm(len(unlabeled_dataset))
    else:
        assert False, 'invalid sampling type'

    # get samples
    samples = unlabeled_dataset[indices[:num_samples]]

    # update labeled dataset
    new_labeled_dataset = (torch.cat([labeled_dataset, samples])).clone()

    # update unlabeled dataset
    new_unlabeled_dataset = (unlabeled_dataset[indices[num_samples:]]).clone()

    return new_unlabeled_dataset, new_labeled_dataset


def entropy_based(model, unlabeled_dataset, device):
    model.eval()
    # make unlabeled loader
    data_loader = torch.utils.data.DataLoader(
        unlabeled_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=8,
    )

    # get entropy
    entropies = []
    for inputs in tqdm(data_loader):
        inputs = inputs.to(device)
        outputs = model(inputs)
        softmax = outputs.softmax(dim=1)

        entropy = -torch.sum(softmax * torch.log(softmax), dim=1)
        entropies.append(entropy.detach().cpu())
    entropies = torch.cat(entropies)

    # get top-k indices
    indices = torch.argsort(entropies, descending=True)

    return indices


def imbalance_aware_kde(victim_dataset, model, victim_model, labeled_dataset, unlabeled_dataset, device):
    model.eval()
    victim_model.eval()

    labeled_loader = torch.utils.data.DataLoader(
        labeled_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=8,
    )

    unlabeled_loader = torch.utils.data.DataLoader(
        unlabeled_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=8,
    )

    # Be Careful: only labeled dataset can be forwarded to the victim model
    # forward and get prediction results (here is the only space in this file that the victim model is used)
    # ========================================  start  ========================================

    pred_list = []
    for x in tqdm(labeled_loader):
        x = x.to(device)
        output = victim_model(x)
        pred = output.argmax(dim=1)
        pred_list.append(pred.detach().cpu())
    pred_list = torch.cat(pred_list)

    # ========================================  end  ========================================

    labeled_feature_list = []
    for x in tqdm(labeled_loader):
        x = x.to(device)
        feature = model.backbone(x)
        labeled_feature_list.append(feature.detach().cpu())
    labeled_feature_list = torch.cat(labeled_feature_list)

    unlabeled_feature_list = []
    for x in tqdm(unlabeled_loader):
        x = x.to(device)
        feature = model.backbone(x)
        unlabeled_feature_list.append(feature.detach().cpu())
    unlabeled_feature_list = torch.cat(unlabeled_feature_list)

    # num samples of each class
    class_cnt = torch.bincount(pred_list, minlength=configs[victim_dataset]['num_classes']).detach().cpu().float()

    # find rarest class (except for the class with 0 sample)
    nonzero_indices = torch.nonzero(class_cnt)
    min_val, min_index = torch.min(class_cnt[nonzero_indices], dim=0)
    min_index = nonzero_indices[min_index].item()
    ur_class = min_index

    ur_kde = KernelDensity(kernel='gaussian', bandwidth=0.5,
                           atol=0.0005, rtol=0.01).fit(labeled_feature_list[pred_list == ur_class])
    print('fit complete')

    unlabeled_feature_loader = torch.utils.data.DataLoader(
        unlabeled_feature_list,
        batch_size=1000,
        shuffle=False,
        num_workers=8,
    )

    score_list = []
    for unlabeled_feature in tqdm(unlabeled_feature_loader):
        ur_kde_score = ur_kde.score_samples(unlabeled_feature)
        score_list.append(ur_kde_score)
    score_list = np.concatenate(score_list)
    score_list = torch.tensor(score_list)
    print('score compute complete: ', score_list.shape)
    indices = torch.argsort(score_list, descending=True)

    return indices
