import numpy as np
import math
import torch
import torch.nn.functional as F
import typing
from kornia.geometry.homography import normalize_points
from kornia.utils import safe_inverse_with_mask


def hdist(H: torch.Tensor, pts_matches: torch.Tensor):
    '''Function, calculates one-way reprojection error
    
    Return:
        torch.Tensor: per-correspondence Euclidian squared error


    Shape:
        - Input :math:`(3, 3)`, :math:`(B, 4)`
        - Output: :math:`(B, 1)`
    '''
    B, dim = pts_matches.shape
    # The original coords
    vec = torch.concat((pts_matches[:, :2], torch.ones((B, 1))), dim=1)
    # The projected coords
    vec_ = pts_matches[:, 2:]
    H_vec = vec @ H
    H_vec[:, 0] /= H_vec[:, 2]
    H_vec[:, 1] /= H_vec[:, 2]
    dist = torch.square(vec_ - (H_vec[:, :2])).sum(dim=1)

    return dist


def sample(pts_matches: torch.Tensor, num: int = 4):
    '''Function, which draws random sample from pts_matches
    
    Return:
        torch.Tensor:

    Args:
        pts_matches: torch.Tensor: 2d tensor
        num (int): number of correspondences to sample

    Shape:
        - Input :math:`(B, 4)`
        - Output: :math:`(num, 4)`
    '''
    indices = torch.randperm(pts_matches.size(0))
    sample = pts_matches[indices[:num]]
    return sample


def getH(min_sample):
    '''Function, which estimates homography from minimal sample
    Return:
        torch.Tensor:

    Args:
        min_sample: torch.Tensor: 2d tensor

    Shape:
        - Input :math:`(B, 4)`
        - Output: :math:`(3, 3)`
    '''
    # Construct the matrix C, to get results close to those of Kornia I use their function
    # If this is not okay I would love to understand what to do instead:
    points1, transform1 = normalize_points(min_sample[:,:2][None])
    points2, transform2 = normalize_points(min_sample[:,2:][None])
    C = torch.zeros(8, 9)
    for i in range(4):
        (x1, y1), (x2, y2) = points1[0, i], points2[0, i]
        # The construction here is the same as Kornia, because the one displayed in the
        # Exercise is different and thus produces different results:
        C[2*i] = torch.tensor([0, 0, 0, -x1, -y1, -1, x1 * y2, y1 * y2, y2])
        C[2*i+1] = torch.tensor([x1, y1, -1, 0, 0, 0, -x1 * x2, -y1 * x2, -x2])
    if torch.linalg.matrix_rank(C) < 8:
        return None
    C = C.unsqueeze(0)
    C = C.transpose(-2, -1) @ C
    _, _, V = torch.linalg.svd(C, full_matrices=True)
    V = V.transpose(-2, -1)
    H = V[..., -1].reshape(3, 3)
    H = torch.linalg.inv(transform2)[0] @ (H @ transform1)
    print(H[:, -1,-1], )
    H_norm = H/(H[:, -1,-1] + 1e-10)
    return H_norm

from kornia.geometry.homography import find_homography_dlt

def nsamples(n_inl: int, num_tc: int, sample_size: int, conf: float):
    return torch.log(1-conf) / torch.log(1-(n_inl ** sample_size))


def ransac_h(pts_matches: torch.Tensor, th: float = 4.0, conf: float = 0.99, max_iter: int = 1000):
    '''Function, which robustly estimates homography from noisy correspondences
    
    Return:
        torch.Tensor: per-correspondence Euclidian squared error

    Args:
        pts_matches: torch.Tensor: 2d tensor
        th (float): pixel threshold for correspondence to be counted as inlier
        conf (float): confidence
        max_iter (int): maximum iteration, overrides confidence
        
    Shape:
        - Input  :math:`(B, 4)`
        - Output: :math:`(3, 3)`,   :math:`(B, 1)`
    '''
    i = 0
    sample_size = 4
    H_best, support_best = torch.eye(3), -torch.inf

    while i < max_iter:
        current_points = sample(pts_matches, sample_size)
        H = getH(current_points)
        if H is not None:
            support = hdist(H, pts_matches)
            if torch.sum(support) > torch.sum(support_best):
                H_best, support_best = H, support
            max_iter = nsamples(n_inl=torch.sum(support_best > th).item(), num_tc=0, sample_size=sample_size, conf=conf)

    inl = support_best > th
    return H_best, inl
