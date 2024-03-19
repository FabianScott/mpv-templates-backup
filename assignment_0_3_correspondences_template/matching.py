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
    topk_vals2, topk_idxs2 = torch.topk(dists.T, 2, dim=1, largest=False)

    mask = torch.zeros_like(dists, dtype=torch.bool)
    for i in range(B1):
        if i in topk_idxs2[topk_idxs[i][0]]:
            mask[i, topk_idxs[i],] = True
        if i in topk_idxs2[topk_idxs[i][1]]:
            mask[i, topk_idxs[i][1],] = True

    closest_pairs = torch.nonzero(mask, as_tuple=False)

    second_closest_pairs = torch.tensor([topk_vals[idx, 1] for idx in closest_pairs[:, 0]])
    dist_ratio = dists[closest_pairs[:, 0], closest_pairs[:, 1]] / second_closest_pairs
    # Filter matches based on the distance ratio threshold
    matches_mask = dist_ratio <= th

    return closest_pairs[matches_mask], dist_ratio[matches_mask]
