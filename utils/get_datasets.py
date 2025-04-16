from .dataset_modules import ImageDataset
from torchvision.transforms import transforms
import torchvision.datasets as datasets
from os.path import join
from .configs import configs, groups

dataset_path = '/workspace/datasets'


def get_dataset(dataset, size='default'):
    # ================================== set size ==================================

    default_size = (32, 32)
    if size == 'default':
        size = default_size

    # ================================== set transform ==================================
    config = configs[dataset]

    if dataset in groups['cifar']:
        transform_train = transforms.Compose([
            transforms.Resize(size),
            transforms.RandomCrop(size[0], padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(config['mean'], config['std']),
        ])
        transform_test = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize(config['mean'], config['std']),
        ])

    elif dataset in groups['mnist']:
        transform_train = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.RandomCrop(size[0], padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(config['mean'], config['std']),
        ])
        transform_test = transforms.Compose([
            transforms.Resize(size),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(config['mean'], config['std']),
        ])

    else:
        assert False, 'group of given dataset is not defined'

    # ================================== gen dataset ==================================

    if dataset == 'mnist':
        trainset = datasets.MNIST(root=dataset_path,
                                  train=True,
                                  download=True,
                                  transform=transform_train)

        testset = datasets.MNIST(root=dataset_path,
                                 train=False,
                                 download=True,
                                 transform=transform_test)

    elif dataset == 'svhn':
        trainset = datasets.SVHN(root=dataset_path,
                                 split='train',
                                 transform=transform_train,
                                 download=True)

        testset = datasets.SVHN(root=dataset_path,
                                split='test',
                                transform=transform_test,
                                download=True)

    elif dataset == 'gtsrb':
        trainset = datasets.GTSRB(root=dataset_path, split='train',
                                  transform=transform_train, download=True)

        testset = datasets.GTSRB(root=dataset_path, split='test',
                                 transform=transform_test, download=True)

    elif dataset == 'cifar10':
        trainset = datasets.CIFAR10(root=dataset_path,
                                    train=True,
                                    transform=transform_train,
                                    download=True)

        testset = datasets.CIFAR10(root=dataset_path,
                                   train=False,
                                   transform=transform_test,
                                   download=True)

    elif dataset == 'eurosat':
        trainset = ImageDataset(root=join(dataset_path, 'eurosat_82'), train=True, transform=transform_train)

        testset = ImageDataset(root=join(dataset_path, 'eurosat_82'), train=False, transform=transform_test)

    elif dataset == 'imagenet':
        trainset = ImageDataset(root=join(dataset_path, 'imagenet32'), train=True,
                                transform=transform_train)

        testset = None  # we don't use imagenet testset (or validation set)

    else:
        assert False, 'undefined dataset name'

    return trainset, testset
