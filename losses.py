import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from torch.autograd import Variable
try:
    from itertools import  ifilterfalse
except ImportError: # py3k
    from itertools import  filterfalse as ifilterfalse



def structure_loss(pred, mask):
    weit  = 1+5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15)-mask)
    wbce  = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce  = (weit*wbce).sum(dim=(2,3))/weit.sum(dim=(2,3))

    pred  = torch.sigmoid(pred)
    inter = ((pred*mask)*weit).sum(dim=(2,3))
    union = ((pred+mask)*weit).sum(dim=(2,3))
    wiou  = 1-(inter+1)/(union-inter+1)
    return (wbce+wiou).mean()

def bce2d_new(input, target, reduction='mean'):
        assert(input.size() == target.size())
        pos = torch.eq(target, 1).float()
        neg = torch.eq(target, 0).float()
        # ing = ((torch.gt(target, 0) & torch.lt(target, 1))).float()

        num_pos = torch.sum(pos)
        num_neg = torch.sum(neg)
        num_total = num_pos + num_neg

        alpha = num_neg / num_total
        beta = 1.1 * num_pos / num_total
        # target pixel = 1 -> weight beta
        # target pixel = 0 -> weight 1-beta
        weights = alpha * pos + beta * neg

        return F.binary_cross_entropy_with_logits(input, target, weights, reduction=reduction)

def BCE_IOU(pred, mask):
    weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask)
    wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask)*weit).sum(dim=(2, 3))
    union = ((pred + mask)*weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1)/(union - inter+1)
    return wbce.mean(), wiou.mean()

# --------------------------- BINARY Lovasz LOSSES ---------------------------
def lovasz_hinge(logits, labels, per_image=True, ignore=None):
    """
    Binary Lovasz hinge loss
      logits: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)
      labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
      per_image: compute the loss per image instead of per batch
      ignore: void class id
    """
    if per_image:
        loss = mean(lovasz_hinge_flat(*flatten_binary_scores(log.unsqueeze(0), lab.unsqueeze(0), ignore))
                          for log, lab in zip(logits, labels))
    else:
        loss = lovasz_hinge_flat(*flatten_binary_scores(logits, labels, ignore))
    return loss


# def lovasz_hinge_flat(logits, labels):
#     """
#     Binary Lovasz hinge loss
#       logits: [P] Variable, logits at each prediction (between -\infty and +\infty)
#       labels: [P] Tensor, binary ground truth labels (0 or 1)
#       ignore: label to ignore
#     """
#     if len(labels) == 0:
#         # only void pixels, the gradients should be 0
#         return logits.sum() * 0.
#     signs = 2. * labels.float() - 1.
#     errors = (1. - logits * Variable(signs))
#     errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
#     perm = perm.data
#     gt_sorted = labels[perm]
#     grad = lovasz_grad(gt_sorted)
#     loss = torch.dot(F.relu(errors_sorted), Variable(grad))
#     return loss

def lovasz_hinge_flat(logits, labels):
    """Binary Lovasz hinge loss.
    Args:
        logits (torch.Tensor): [P], logits at each prediction
            (between -infty and +infty).
        labels (torch.Tensor): [P], binary ground truth labels (0 or 1).
    Returns:
        torch.Tensor: The calculated loss.
    """
    if len(labels) == 0:
        # only void pixels, the gradients should be 0
        return logits.sum() * 0.
    signs = 2. * labels.float() - 1.
    errors = (1. - logits * signs)
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    perm = perm.data
    gt_sorted = labels[perm]
    grad = lovasz_grad(gt_sorted)
    loss = torch.dot(F.relu(errors_sorted), grad)
    return loss

def flatten_binary_scores(scores, labels, ignore=None):
    """
    Flattens predictions in the batch (binary case)
    Remove labels equal to 'ignore'
    """
    scores = scores.view(-1)
    labels = labels.view(-1)
    if ignore is None:
        return scores, labels
    valid = (labels != ignore)
    vscores = scores[valid]
    vlabels = labels[valid]
    return vscores, vlabels


class StableBCELoss(torch.nn.modules.Module):
    def __init__(self):
         super(StableBCELoss, self).__init__()
    def forward(self, input, target):
         neg_abs = - input.abs()
         loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
         return loss.mean()


def binary_xloss(logits, labels, ignore=None):
    """
    Binary Cross entropy loss
      logits: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)
      labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
      ignore: void class id
    """
    logits, labels = flatten_binary_scores(logits, labels, ignore)
    loss = StableBCELoss()(logits, labels.float())
    return loss

# def lovasz_grad(gt_sorted):
#     """
#     Computes gradient of the Lovasz extension w.r.t sorted errors
#     See Alg. 1 in paper
#     """
#     p = len(gt_sorted)
#     gts = gt_sorted.sum()
#     intersection = gts - gt_sorted.float().cumsum(0)
#     union = gts + (1 - gt_sorted).float().cumsum(0)
#     jaccard = 1. - intersection / union
#     if p > 1: # cover 1-pixel case
#         jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
#     return jaccard

# new implementation from mmseg
# https://github.com/open-mmlab/mmsegmentation/blob/master/mmseg/models/losses/lovasz_loss.py

def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1:  # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def mean(l, ignore_nan=False, empty=0):
    """
    nanmean compatible with generators.
    """
    l = iter(l)
    if ignore_nan:
        l = ifilterfalse(isnan, l)
    try:
        n = 1
        acc = next(l)
    except StopIteration:
        if empty == 'raise':
            raise ValueError('Empty mean')
        return empty
    for n, v in enumerate(l, 2):
        acc += v
    if n == 1:
        return acc
    return acc / n

def isnan(x):
    return x != x







def bce_loss(pred, mask):
    pred = torch.sigmoid(pred)
    bce_loss = nn.BCELoss()
    bce = bce_loss(pred, mask)
    return bce

# def bce2d_new_weights(input, target):
#     assert(input.size() == target.size())
#     pos = torch.eq(target, 1).float()
#     neg = torch.eq(target, 0).float()
#     # ing = ((torch.gt(target, 0) & torch.lt(target, 1))).float()

#     num_pos = torch.sum(pos)
#     num_neg = torch.sum(neg)
#     num_total = num_pos + num_neg

#     alpha = num_neg  / num_total
#     beta = 1.1 * num_pos  / num_total
#     # target pixel = 1 -> weight beta
#     # target pixel = 0 -> weight 1-beta
#     weights = alpha * pos + beta * neg

#     return weights

def BinaryDiceLoss(predict, target, valid_mask=None):
        smooth=1
        p=2
        reduction='mean'
        
        target = target.float()
        valid_mask = torch.ones_like(target) 
        
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)
        valid_mask = valid_mask.contiguous().view(valid_mask.shape[0], -1)

        num = torch.sum(torch.mul(predict, target) * valid_mask, dim=1) * 2 + smooth
        den = torch.sum((predict.pow(p) + target.pow(p)) * valid_mask, dim=1) + smooth

        loss = 1 - num / den

        if reduction == 'mean':
            # return dict(loss=loss.mean())
            return loss.mean()
        elif reduction == 'sum':
            return loss.sum()
        elif reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(reduction))


def reflection_loss(y_true, y_pred, glass_mask=None):
    # Flatten the inputs
    y_true_flat = y_true.view(-1)
    y_pred_flat = y_pred.view(-1)

    # Compute binary cross-entropy loss
    bce_loss = F.binary_cross_entropy(y_pred_flat, y_true_flat)

    if glass_mask is None:
        return bce_loss 

    glass_mask_flat = glass_mask.view(-1)
    penalty = torch.sum(torch.abs(y_pred_flat * glass_mask_flat - y_pred_flat))

    loss = bce_loss + penalty
    return loss



