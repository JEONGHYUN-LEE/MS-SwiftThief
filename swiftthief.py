from utils.get_datasets import get_dataset
import argparse
import torch
from utils.configs import configs
import models
from contrastive_learning.simsiam.model_factory import SimSiam
import torch.optim as optim
from utils.train import train_kd, test, samples_per_class
import os
import numpy as np
import random
from attack_tools.sampling import sampler
from tqdm import tqdm
from contrastive_learning.main_sup_soft import main as contrastive_learning
import gc


def sampling_scheduling(attack_train_loader, victim_model):
    cnt = samples_per_class(
        loader=attack_train_loader,
        model=victim_model,
        num_classes=configs[args.victim_dataset]["num_classes"],
        device=args.device,
    )

    mean_cnt = cnt.mean()
    mean_ur = cnt[cnt < mean_cnt].mean()
    c_ur = (cnt < mean_cnt).sum()
    remain_budget = args.query_budget - cnt.sum()

    switch = remain_budget <= c_ur * (mean_cnt - mean_ur)
    if switch:
        sampling_type = "imbalance_kde"
    else:
        sampling_type = "entropy"
    print(
        "remain budget: {} avg gap: {} c_ur: {} chosen sampling: {}".format(
            remain_budget, mean_cnt - mean_ur, c_ur, sampling_type
        )
    )
    return sampling_type


def main():
    _, test_dataset = get_dataset(args.victim_dataset, size=(32, 32))
    print("total attack data load")

    attack_dataset = torch.load("unlabeled_dataset.pt")

    labeled_dataset = attack_dataset[: int(args.query_budget * 0.1)]
    unlabeled_dataset = attack_dataset[int(args.query_budget * 0.1) :]

    # print('seed sampling start')
    # for x, _ in tqdm(attack_dataset_sampler):
    #     if len(labeled_dataset) * sampling_batch_size >= int(args.query_budget * 0.1):
    #         unlabeled_dataset.append(x)
    #     else:
    #         labeled_dataset.append(x)
    print("ratio", len(unlabeled_dataset), len(labeled_dataset))

    # make attack train loader with sampled_attack_dataset
    attack_train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(labeled_dataset),
        batch_size=100,
        shuffle=True,
        num_workers=4,
    )

    # make test loader
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=100,
        shuffle=False,
        num_workers=4,
    )

    victim_model = models.resnet18_cifar(configs[args.victim_dataset]["num_classes"])
    victim_model.load_state_dict(torch.load(args.victim_path)["model_dict"])
    victim_model.to(args.device)

    attack_model = SimSiam(configs[args.victim_dataset]["num_classes"])
    attack_model.mode = "sl"
    attack_model.to(args.device)

    cnt = samples_per_class(
        loader=attack_train_loader,
        model=victim_model,
        num_classes=configs[args.victim_dataset]["num_classes"],
        device=args.device,
    )

    beta = 0.99
    costs = (1 - beta) / (1 - beta ** (cnt + 1.0))
    costs = costs / costs.sum()
    costs = costs.to(args.device)
    print(cnt)
    print(costs)

    # optimizer & scheduler
    optimizer = optim.SGD(
        attack_model.parameters(), lr=args.sl_lr, momentum=0.9, weight_decay=5e-4
    )
    train_accs = []
    test_accs = []

    # contrastive learning
    cl_epochs = 0
    num_unlabeled = 5000
    attack_model.mode = "ssl"
    contrastive_learning(
        args=args,
        labeled_dataset=labeled_dataset,
        model=attack_model,
        victim_model=victim_model,
        batch_size=512,
        feat_dim=2048,
        num_unlabeled=num_unlabeled,
        start_epochs=cl_epochs,
        total_epochs=40,
        init_lr=0.06,
        costs=costs,
        device=args.device,
    )
    attack_model.mode = "sl"

    cl_epochs += 40

    # routine start
    sampling_type_list = []
    class_histogram_list = []
    for epoch in range(args.sl_epoch):
        if epoch % args.sl_aug_interval == 0 and epoch != 0:
            print("query sampling start")
            sampling_type = sampling_scheduling(attack_train_loader, victim_model)

            if sampling_type == "entropy":
                unlabeled_dataset, labeled_dataset = sampler(
                    sampling_type=sampling_type,
                    victim_dataset=args.victim_dataset,
                    unlabeled_dataset=unlabeled_dataset,
                    labeled_dataset=labeled_dataset,
                    model=attack_model,
                    victim_model=victim_model,
                    num_samples=int(args.query_budget * 0.1),
                    device=args.device,
                )
                gc.collect()

            elif sampling_type == "imbalance_kde":
                for _ in range(5):
                    unlabeled_dataset, labeled_dataset = sampler(
                        sampling_type=sampling_type,
                        victim_dataset=args.victim_dataset,
                        unlabeled_dataset=unlabeled_dataset,
                        labeled_dataset=labeled_dataset,
                        model=attack_model,
                        victim_model=victim_model,
                        num_samples=int(int(args.query_budget * 0.1) / 5),
                        device=args.device,
                    )
                    gc.collect()
                    print("ratio", len(unlabeled_dataset), len(labeled_dataset))
            attack_train_loader = torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(labeled_dataset),
                batch_size=100,
                shuffle=True,
                num_workers=4,
            )

            print("fin ratio", len(unlabeled_dataset), len(labeled_dataset))

            # update status
            sampling_type_list.append(sampling_type)
            cnt = samples_per_class(
                loader=attack_train_loader,
                model=victim_model,
                num_classes=configs[args.victim_dataset]["num_classes"],
                device=args.device,
            )
            class_histogram_list.append(cnt)

            # contrastive learning
            if epoch == 450:
                total_epochs = 100
            else:
                total_epochs = 40

            cnt = samples_per_class(
                loader=attack_train_loader,
                model=victim_model,
                num_classes=configs[args.victim_dataset]["num_classes"],
                device=args.device,
            )

            beta = 0.99
            costs = (1 - beta) / (1 - beta ** (cnt + 1.0))
            costs = costs / costs.sum()
            costs = costs.to(args.device)
            print(cnt)
            print(costs)

            attack_model.mode = "ssl"
            num_unlabeled += 5000
            contrastive_learning(
                args=args,
                labeled_dataset=labeled_dataset,
                model=attack_model,
                victim_model=victim_model,
                batch_size=512,
                feat_dim=2048,
                num_unlabeled=num_unlabeled,
                start_epochs=cl_epochs,
                total_epochs=total_epochs,
                init_lr=0.06,
                costs=costs,
                device=args.device,
            )
            attack_model.mode = "sl"
            cl_epochs += 40

        # output matching
        train_acc = train_kd(
            loader=attack_train_loader,
            model=attack_model,
            teacher=victim_model,
            optimizer=optimizer,
            device=args.device,
        )

        test_acc = test(loader=test_loader, model=attack_model, device=args.device)
        if epoch % 20 == 0:
            v_test_acc = test(
                loader=test_loader, model=victim_model, device=args.device
            )
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        print(
            "Epoch: {}, train_acc: {}, test_acc: {}, v_test_acc: {}".format(
                epoch,
                round(train_acc * 100, 2),
                round(test_acc * 100, 2),
                round(v_test_acc * 100, 2),
            )
        )

    results = {
        "train_accs": train_accs,
        "test_accs": test_accs,
        "v_test_acc": v_test_acc,
        "model_dict": attack_model.cpu().state_dict(),
        "labeled_dataset": labeled_dataset,
        "sampling_type_list": sampling_type_list,
        "class_histogram_list": class_histogram_list,
    }

    os.makedirs(os.path.split(args.save_path)[0], exist_ok=True)
    torch.save(results, args.save_path)


parser = argparse.ArgumentParser(description="Train a model")
# fundamental for ms attack
parser.add_argument("--device", type=str)
parser.add_argument("--victim_dataset", type=str)
parser.add_argument("--attack_dataset", type=str)
parser.add_argument("--query_budget", type=int)

parser.add_argument("--sl_lr", type=float, default=1e-2)
parser.add_argument("--sl_epoch", type=int)
parser.add_argument("--sl_aug_interval", type=int)

parser.add_argument("--victim_path", type=str)
parser.add_argument("--save_path", type=str)

args = parser.parse_args()

seed = 0
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

if __name__ == "__main__":
    main()
