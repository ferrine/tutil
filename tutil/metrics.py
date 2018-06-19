import torch
import torch.functional as F
import numpy as np

__all__ = [
    'evaluate'
]


def where(cond, xt, xf):
    ret = torch.zeros_like(xt)
    ret[cond] = xt[cond]
    ret[cond ^ 1] = xf[cond ^ 1]
    return ret


def logmeanexp(x, dim=None, keepdim=False):
    """Stable computation of log(mean(exp(x))"""
    if dim is None:
        x, dim = x.view(-1), 0
    xm, _ = torch.max(x, dim, keepdim=True)
    x = where((xm == float('inf')) | (xm == float('-inf')),
              xm,
              xm + torch.log(torch.mean(torch.exp(x - xm), dim, keepdim=True)))
    return x if keepdim else x.squeeze(dim)


def logits2acc(outputs, targets):
    return np.mean(outputs.cpu().numpy().argmax(axis=1) == targets.data.cpu().numpy())


def evaluate(net, dataloader, num_ens=1):
    """Calculate ensemble accuracy and NLL"""
    accs = []
    nlls = []
    for i, (inputs, labels) in enumerate(dataloader):
        inputs =torch.autograd.Variable(inputs.cuda(async=True))
        labels = torch.autograd.Variable(labels.cuda(async=True))
        outputs = torch.zeros(inputs.shape[0], net.num_classes, num_ens).cuda()
        for j in range(num_ens):
            outputs[:, :, j] = F.log_softmax(net(inputs), dim=1).data
        accs.append(logits2acc(logmeanexp(outputs, dim=2), labels))
        nlls.append(F.nll_loss(torch.autograd.Variable(logmeanexp(outputs, dim=2)),
                               labels, size_average=False).data.cpu().numpy())
    return np.mean(accs), np.sum(nlls)
