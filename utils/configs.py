groups = {
    'mnist': ['mnist'],
    'cifar': ['svhn', 'gtsrb', 'cifar10', 'eurosat', 'imagenet'],
}

configs = {
    'mnist': {
        'mean': [0., 0., 0.],
        'std': [1., 1., 1.],
        'num_classes': 10,
    },
    'svhn': {
        'mean': [0.4377, 0.4438, 0.4728],
        'std': [0.1980, 0.2010, 0.1970],
        'num_classes': 10,
    },
    'gtsrb': {
        'mean': [0.3337, 0.3064, 0.3171],
        'std': [0.2672, 0.2564, 0.2629],
        'num_classes': 43,
    },
    'cifar10': {
        'mean': [0.4914, 0.4822, 0.4465],
        'std': [0.2023, 0.1994, 0.2010],
        'num_classes': 10,
    },
    'imagenet': {
        'mean': [0.485, 0.456, 0.406],
        'std': [0.229, 0.224, 0.225],
        'num_classes': 1000,
    },
    'eurosat': {
        'mean': [0.3438, 0.3800, 0.4074],
        'std': [0.2033, 0.1363, 0.1146],
        'num_classes': 10,
    },
}