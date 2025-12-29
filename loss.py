import torch
import torch.nn as nn
from torch.distributions.multivariate_normal import MultivariateNormal


def KLDivLoss(latents, dataset, device):
    """Per-dim diagonal KL(batch || target). Stable for any latent dim, batch size.

    Pulls batch's per-dim mean+variance toward GT distribution stats stored on
    `dataset.target_mean` and `dataset.target_var` (precomputed once before training
    from the supervision target latents).
    """
    eps = 1e-6
    batch_mean = latents.mean(dim=0)
    batch_var = latents.var(dim=0, unbiased=False) + eps
    gt_mean = dataset.target_mean.to(device)
    gt_var = dataset.target_var.to(device) + eps
    kl = 0.5 * (torch.log(gt_var / batch_var)
                + (batch_var + (batch_mean - gt_mean).pow(2)) / gt_var - 1.0)
    loss = kl.mean()
    with torch.no_grad():
        det = batch_var.prod().detach()
    return loss, det


def DistMatchLoss(latents, dataset, device):
    """Direct per-dim moment matching: (batch_std - gt_std)^2 + (batch_mean - gt_mean)^2.

    More stable than KL: linear gradients, no log/exp. Pulls encoder output's per-dim
    mean and std to match GT distribution. Variant of VICReg variance term and
    CORAL moment-matching style losses.

    Requires `dataset.target_mean` and `dataset.target_var` (precomputed).
    """
    eps = 1e-6
    batch_mean = latents.mean(dim=0)
    batch_std = latents.std(dim=0, unbiased=False) + eps
    gt_mean = dataset.target_mean.to(device)
    gt_std = (dataset.target_var.to(device) + eps).sqrt()
    std_loss = ((batch_std - gt_std) ** 2).mean()
    mean_loss = ((batch_mean - gt_mean) ** 2).mean()
    loss = std_loss + mean_loss
    with torch.no_grad():
        ratio = (batch_std / gt_std).mean().detach()
    return loss, ratio


def AttRepLoss(latents, fruit_ids, device, delta_rep=.5):
    """
    Attraction-Repulsion loss with proper handling of self-comparisons.
    
    For each sample, compute distance to all other samples and apply hinge loss:
    - Attract samples from the same fruit (positive pairs)
    - Repel samples from different fruits (negative pairs)
    - Exclude self-comparisons (distance to itself)
    """
    h_loss = 0
    hinged_loss = torch.nn.HingeEmbeddingLoss(margin=delta_rep, reduction='none')
    
    torch.set_printoptions(linewidth=500, sci_mode=False)
    
    batch_size = latents.shape[0]
    
    for idx, (f_id, c_lat) in enumerate(zip(fruit_ids, latents)):
        # Compute distances to all samples in batch
        dist = torch.linalg.norm(c_lat - latents, dim=1)
        
        # Create mask: +1 for same fruit (attract), -1 for different fruit (repel)
        mask = (fruit_ids == f_id)*2 - 1
        
        # Compute hinge loss for all pairs
        c_loss = hinged_loss(dist, mask.cuda())
        
        # Exclude self-comparison (set loss to 0 for current sample comparing to itself)
        c_loss[idx] = 0.0
        
        h_loss += c_loss.sum()
    
    # Normalize by the number of valid pairs: batch_size * (batch_size - 1)
    # This keeps the loss magnitude consistent regardless of batch size
    h_loss = h_loss / (batch_size * (batch_size - 1))
    
    return h_loss


def SuperLoss(pred, gt):
    return nn.MSELoss()(pred.cuda(), gt.cuda())


def SDFLoss(pred, target, target_weights, sdf_trunc, points):

    pred = torch.clamp(pred, min=-sdf_trunc, max=sdf_trunc)

    # import ipdb;ipdb.set_trace()
    # targets = []

    # import utils
    # for i in  range(target.shape[0]):
    #     pnt = points[i].points
    #     tar = target[i]
    #     con = tar.abs() <= 1
    #     # import ipdb;ipdb.set_trace()
    #     data = torch.cat((pnt[con[:,0]], tar[con].unsqueeze(dim=1)), dim=1)
    #     pcd = utils.visualize_sdf(data.detach().cpu().numpy())
    #     targets.append(pcd)

    # import open3d as o3d
    # o3d.visualization.draw_geometries(targets, window_name='all')

    target = target[target_weights !=0 ]
    pred = pred[target_weights !=0 ]

    narrow_band = target.abs() < 1
    target = target[narrow_band]
    pred = pred[narrow_band]

    pred /= sdf_trunc # normalized sdf predd
    
    pred = log_transform(pred)
    target = log_transform(target)
    
    return nn.L1Loss()(pred, target)


def log_transform(sdf):
        return sdf.sign() * (sdf.abs() + 1.0).log()

def RegLatentLoss(batch_vecs, code_reg_lambda, epoch):
    loss = torch.abs(1 - torch.norm(batch_vecs, dim=1)).mean()
    loss *= code_reg_lambda
    return loss


def SizeAwareContrastiveLoss(latents, fruit_idx, dates_days, vol_proxy,
                              vol_thr=0.05, days_thr=7, margin=1.0):
    """
    Size-aware contrastive loss for temporal partial PCD training.

    Positive pair: same fruit + |dates_diff|<=days_thr + |vol_i-vol_j|/max(vol)<vol_thr
    Negative pair: different fruit
    Neutral (no gradient): same fruit with large size/time gap (growing), or vol missing

    Args:
        latents: (B, D) encoded latents
        fruit_idx: (B,) long tensor — per-sample fruit identity (integer index)
        dates_days: (B,) float/long tensor — days since reference per sample
        vol_proxy: (B,) float tensor — ellipsoid volume per sample (NaN/-1 if missing)
        vol_thr: positive-pair relative volume-diff threshold (e.g. 0.05)
        days_thr: positive-pair |day-diff| threshold (e.g. 7)
        margin: negative-pair hinge margin (target pairwise latent distance)

    Returns:
        scalar tensor
    """
    B = latents.shape[0]
    if B < 2:
        return torch.zeros((), device=latents.device)

    # pairwise L2 distance (B,B)
    diff = latents.unsqueeze(0) - latents.unsqueeze(1)
    dist = torch.linalg.norm(diff, dim=-1)

    fi = fruit_idx.view(-1)
    dd = dates_days.view(-1).float()
    vp = vol_proxy.view(-1).float()

    same_fruit = (fi.unsqueeze(0) == fi.unsqueeze(1))
    day_diff = torch.abs(dd.unsqueeze(0) - dd.unsqueeze(1))
    days_close = day_diff <= days_thr

    vol_valid = (vp.unsqueeze(0) > 0) & (vp.unsqueeze(1) > 0)
    vol_max = torch.maximum(vp.unsqueeze(0), vp.unsqueeze(1))
    vol_diff_ratio = torch.abs(vp.unsqueeze(0) - vp.unsqueeze(1)) / torch.clamp(vol_max, min=1e-6)
    vol_close = vol_diff_ratio < vol_thr

    # Positive: same fruit + close time + close size + valid vol + not self
    positive = same_fruit & days_close & vol_close & vol_valid
    positive.fill_diagonal_(False)

    # Negative: different fruit (no time/vol constraints needed)
    negative = ~same_fruit

    n_pos = positive.sum().item()
    n_neg = negative.sum().item()

    pos_term = dist[positive].sum() if n_pos > 0 else torch.zeros((), device=latents.device)
    neg_term = torch.clamp(margin - dist[negative], min=0.0).sum() if n_neg > 0 else torch.zeros((), device=latents.device)

    total = n_pos + n_neg
    if total == 0:
        return torch.zeros((), device=latents.device)
    return (pos_term + neg_term) / float(total)
