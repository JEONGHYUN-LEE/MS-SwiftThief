import os.path as osp
from torchvision.datasets.folder import ImageFolder


class ImageDataset(ImageFolder):
    def __init__(
        self,
        train=True,
        transform=None,
        target_transform=None,
        root=None,
        **kwargs
    ):
        if not osp.exists(root):
            raise ValueError(
                "Dataset not found at {}. Please download it first.".format(
                    root,
                )
            )

        # Initialize ImageFolder
        split = "train" if train else "test"
        super().__init__(
            root=osp.join(root, split),
            transform=transform,
            target_transform=target_transform,
        )
        self.root = root

        print(
            "=> done loading {} ({}) with {} examples".format(
                self.__class__.__name__, split, len(self.samples)
            )
        )


class ImageDataset_c(ImageFolder):
    def __init__(
        self,
        train=True,
        transform=None,
        target_transform=None,
        root=None,
        **kwargs
    ):
        if not osp.exists(root):
            raise ValueError(
                "Dataset not found at {}. Please download it first.".format(
                    root,
                )
            )

        # Initialize ImageFolder
        super().__init__(
            root=root,
            transform=transform,
            target_transform=target_transform,
        )
        self.root = root
