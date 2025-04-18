from torch import nn
from .resnet_cifar import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152


class projection_MLP(nn.Module):
    def __init__(self, in_dim, out_dim, num_layers=2):
        super().__init__()
        hidden_dim = out_dim
        self.num_layers = num_layers

        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )

        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.BatchNorm1d(out_dim, affine=False)  # Page:5, Paragraph:2
        )

    def forward(self, x):
        if self.num_layers == 2:
            x = self.layer1(x)
            x = self.layer3(x)
        elif self.num_layers == 3:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)

        return x


class prediction_MLP(nn.Module):
    def __init__(self, in_dim=2048):
        super().__init__()
        out_dim = in_dim
        hidden_dim = int(out_dim / 4)

        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)

        return x


class SimSiam(nn.Module):
    def __init__(self, num_class, arch=None):
        super(SimSiam, self).__init__()
        self.mode = 'ssl'
        if arch:
            self.backbone = SimSiam.get_backbone(arch)
        else:
            self.backbone = SimSiam.get_backbone('resnet18')
        out_dim = self.backbone.fc.weight.shape[1]
        self.backbone.fc = nn.Identity()
        self.fc = nn.Linear(out_dim, num_class)

        self.projector = projection_MLP(out_dim, 2048,
                                        2)

        self.encoder = nn.Sequential(
            self.backbone,
            self.projector
        )

        self.predictor = prediction_MLP(2048)

    @staticmethod
    def get_backbone(backbone_name):
        return {'resnet18': ResNet18(),
                'resnet34': ResNet34(),
                'resnet50': ResNet50(),
                'resnet101': ResNet101(),
                'resnet152': ResNet152()}[backbone_name]

    def forward(self, im_aug1, im_aug2=None):
        if im_aug2 is None:
            if self.mode == 'ssl':
                z1 = self.encoder(im_aug1)
                p1 = self.predictor(z1)
                return p1
            elif self.mode == 'sl':
                logit = self.fc(self.backbone(im_aug1))
                return logit
            else:
                assert False, 'invalid mode'
        else:
            if self.mode != 'ssl':
                assert False, 'something wrong, the mode must be `ssl` for contrastive learning'
            z1 = self.encoder(im_aug1)
            z2 = self.encoder(im_aug2)

            p1 = self.predictor(z1)
            p2 = self.predictor(z2)

        return {'z1': z1, 'z2': z2, 'p1': p1, 'p2': p2}







