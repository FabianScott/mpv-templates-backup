import numpy as np
import math
import torch
import torch.nn.functional as F
import typing

import torch
import torch.nn.functional as F


def match_snn(desc1: torch.Tensor, desc2: torch.Tensor, th: float = 0.8):
    '''Function, which finds nearest neighbors for each vector in desc1,
    which satisfy first to second nearest neighbor distance <= th check

    Return:
        torch.Tensor: indexes of matching descriptors in desc1 and desc2
        torch.Tensor: L2 descriptor distance ratio 1st to 2nd nearest neighbor


    Shape:
        - Input :math:`(B1, D)`, :math:`(B2, D)`
        - Output: :math:`(B3, 2)`, :math:`(B3, 1)` where 0 <= B3 <= B1
    '''
    B1, D = desc1.shape
    B2, D = desc2.shape
    # Calculate pairwise distances between descriptors
    dists = torch.cdist(desc1, desc2)
    # Find the nearest and second nearest neighbors
    topk_vals, topk_idxs = torch.topk(dists, 2, dim=1, largest=False)

    top1_idx = torch.stack((torch.arange(len(topk_idxs)), topk_idxs[:, 0])).T
    # top2_idx = torch.stack((torch.arange(len(topk_idxs)), topk_idxs[:, 1])).T

    dist_ratios = topk_vals[:, 0] / topk_vals[:, 1]
    mask = dist_ratios <= th

    return top1_idx[mask], dist_ratios[mask]


def match_snn_mutual(desc1: torch.Tensor, desc2: torch.Tensor, th: float = 0.8):
    '''Function, which finds nearest neighbors for each vector in desc1,
    which satisfy first to second nearest neighbor distance <= th check

    Return:
        torch.Tensor: indexes of matching descriptors in desc1 and desc2
        torch.Tensor: L2 descriptor distance ratio 1st to 2nd nearest neighbor


    Shape:
        - Input :math:`(B1, D)`, :math:`(B2, D)`
        - Output: :math:`(B3, 2)`, :math:`(B3, 1)` where 0 <= B3 <= B1
    '''
    B1, D = desc1.shape
    B2, D = desc2.shape
    # Calculate pairwise distances between descriptors
    dists = torch.cdist(desc1, desc2)
    # Find the nearest and second nearest neighbors
    topk_vals, topk_idxs = torch.topk(dists, 2, dim=1, largest=False)
    topk_vals2, topk_idxs2 = torch.topk(dists.T, 2, dim=1, largest=False)

    mask = torch.zeros_like(dists, dtype=torch.bool)
    for i in range(B1):
        if i in topk_idxs2[topk_idxs[i][0]]:
            mask[i, topk_idxs[i],] = True
        if i in topk_idxs2[topk_idxs[i][1]]:
            mask[i, topk_idxs[i][1],] = True

    closest_pairs = torch.nonzero(mask, as_tuple=False)

    second_closest_pair_vals = torch.tensor([topk_vals[idx, 1] for idx in closest_pairs[:, 0]])
    dist_ratio = dists[closest_pairs[:, 0], closest_pairs[:, 1]] / second_closest_pair_vals
    # Filter matches based on the distance ratio threshold
    matches_mask = dist_ratio <= th

    return closest_pairs[matches_mask], dist_ratio[matches_mask]



if __name__ == "__main__":
    v1 = torch.tensor([[0, 1], [1, 1], [-1, 1], [0, 0.5]]).view(-1, 2).float()
    v2 = torch.cat([torch.tensor([[3, 3.], [5, 5.]]), v1, torch.tensor([[2., 2.]])], dim=0)
    print(v1)
    print(v2)
    desc1 = v1
    desc2 = v2
    match_idxs, vals = match_snn(desc2, desc1, 0.8)
