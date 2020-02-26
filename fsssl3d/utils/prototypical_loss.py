import torch
import torch.nn as nn
from torch.nn import functional as F

class PrototypicalLoss(nn.Module):
    def __init__(self, num_support):
        super(PrototypicalLoss, self).__init__()
        self.num_support = num_support

    def forward(self, pred, target):
        classes = torch.unique(target)
        n_classes = len(classes)
        n_query = target.eq(classes[0].item()).sum().item() - self.num_support

        support_idxs = list(map(lambda c: target.eq(c).nonzero()[:self.num_support].squeeze(1), classes))
        query_idxs = torch.stack(list(map(lambda c: target.eq(c).nonzero()[self.num_support:], classes))).view(-1)

        prototypes = torch.stack([pred[idx_list].mean(0) for idx_list in support_idxs])
        query_sample = pred[query_idxs]

        dists = euclidean_dist(query_sample, prototypes)

        log_p_y = F.log_softmax(-dists, dim=1).view(n_classes, n_query, -1)

        target_inds = torch.arange(0, n_classes)
        target_inds = target_inds.view(n_classes, 1, 1)
        target_inds = target_inds.expand(n_classes, n_query, 1).long().cuda()

        loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
        _, y_hat = log_p_y.max(2)
        acc_val = y_hat.eq(target_inds.squeeze()).float().mean()

        return loss_val,  acc_val

def euclidean_dist(x, y):
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    if d != y.size(1):
        raise Exception

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)