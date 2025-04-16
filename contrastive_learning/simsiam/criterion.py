from torch import nn
import torch


class SimSiamLoss(nn.Module):
    def __init__(self, version='simplified'):
        super().__init__()
        self.ver = version

    def asymmetric_loss(self, p, z):
        if self.ver == 'original':
            z = z.detach()  # stop gradient

            p = nn.functional.normalize(p, dim=1)
            z = nn.functional.normalize(z, dim=1)

            return -(p * z).sum(dim=1).mean()

        elif self.ver == 'simplified':
            z = z.detach()  # stop gradient
            return - nn.functional.cosine_similarity(p, z, dim=-1).mean()

    def forward(self, z1, z2, p1, p2):

        loss1 = self.asymmetric_loss(p1, z2)
        loss2 = self.asymmetric_loss(p2, z1)

        return 0.5 * loss1 + 0.5 * loss2


class SupSimSiamLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device

    def forward(self, p, z, targets):
        z = z.detach()  # stop gradient

        p = nn.functional.normalize(p, dim=1)
        z = nn.functional.normalize(z, dim=1)

        dot_product = -torch.mm(p, z.T)

        mask_similar_class = (targets.unsqueeze(1).repeat(1, targets.shape[0]) == targets).to(self.device)
        mask_anchor_out = (1 - torch.eye(dot_product.shape[0])).to(self.device)
        mask_combined = mask_similar_class * mask_anchor_out

        dot_product_selected = dot_product * mask_combined
        return dot_product_selected[dot_product_selected.nonzero(as_tuple=True)].mean()


class SoftSupSimSiamLossV17(nn.Module):
    def __init__(self, device, num_classes):
        super().__init__()
        self.device = device
        self.num_classes = num_classes

    def forward(self, p, z, targets):
        z = z.detach()  # stop gradient

        p = nn.functional.normalize(p, dim=1)
        z = nn.functional.normalize(z, dim=1)

        dot_product = -torch.mm(p, z.T)

        entr = -(targets*targets.log()).sum(dim=1)
        entr[torch.isnan(entr)] = 0.
        norm_entr = entr / torch.log(torch.tensor(self.num_classes))
        reversed_norm_entr = 1 - norm_entr
        mask_similar_class1 = torch.outer(reversed_norm_entr, reversed_norm_entr)

        mask_similar_class2 = torch.nn.functional.cosine_similarity(targets.T.repeat(len(targets), 1, 1),
                                                                   targets.unsqueeze(2)).to(self.device)


        
        mask_anchor_out = (1 - torch.eye(dot_product.shape[0])).to(self.device)
        mask_combined = mask_similar_class1 * mask_similar_class2 * mask_anchor_out
        
        dot_product_selected = dot_product * mask_combined

        return dot_product_selected[dot_product_selected.nonzero(as_tuple=True)].mean()


class CL_FGSM(nn.Module):
    def __init__(self, model, eps, device):
        super().__init__()
        self.device = device
        self.model = model
        self.eps = eps

    def asymmetric_loss(self, p, z):

        z = z.detach()  # stop gradient

        p = nn.functional.normalize(p, dim=1)
        z = nn.functional.normalize(z, dim=1)

        return -(p * z).sum(dim=1).mean()


    def forward(self, x1, x2):
        self.model.eval()

        x1.requires_grad = True

        outs = self.model(im_aug1=x1, im_aug2=x2)
        loss1 = self.asymmetric_loss(outs['p1'], outs['z2'])
        loss2 = self.asymmetric_loss(outs['p2'], outs['z1'])

        loss = 0.5 * loss1 + 0.5 * loss2

        loss.backward()

        adv_x1 = x1 + self.eps * x1.grad.sign()

        return adv_x1.detach()


class SimSiamLoss_cost_sensitive(nn.Module):
    def __init__(self, costs):
        super().__init__()
        self.costs = costs

    def asymmetric_loss(self, p, z, targets):

        z = z.detach()  # stop gradient

        p = nn.functional.normalize(p, dim=1)
        z = nn.functional.normalize(z, dim=1)

        return -((p * z).sum(dim=1) * self.costs[targets]).mean()

    def forward(self, z1, z2, p1, p2, targets):

        loss1 = self.asymmetric_loss(p1, z2, targets)
        loss2 = self.asymmetric_loss(p2, z1, targets)

        return 0.5 * loss1 + 0.5 * loss2


class SupSimSiamLossSum(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device

    def forward(self, p, z, targets):
        z = z.detach()  # stop gradient

        p = nn.functional.normalize(p, dim=1)
        z = nn.functional.normalize(z, dim=1)

        dot_product = -torch.mm(p, z.T)

        mask_similar_class = (targets.unsqueeze(1).repeat(1, targets.shape[0]) == targets).to(self.device)
        mask_anchor_out = (1 - torch.eye(dot_product.shape[0])).to(self.device)
        mask_combined = mask_similar_class * mask_anchor_out

        dot_product_selected = dot_product * mask_combined
        return dot_product_selected[dot_product_selected.nonzero(as_tuple=True)].sum()