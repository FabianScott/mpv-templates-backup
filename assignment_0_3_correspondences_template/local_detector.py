import numpy as np
import math
import torch
import torch.nn.functional as F
import typing
from imagefiltering import *


def harris_response(x: torch.Tensor,
                    sigma_d: float,
                    sigma_i: float,
                    alpha: float = 0.04) -> torch.Tensor:
    r"""Computes the Harris cornerness function.The response map is computed according the following formulation:

    .. math::
        R = det(M) - alpha \cdot trace(M)^2

    where:

    .. math::
        M = \sum_{(x,y) \in W}
        \begin{bmatrix}
            I^{2}_x & I_x I_y \\
            I_x I_y & I^{2}_y \\
        \end{bmatrix}

    and :math:`k` is an empirically determined constant
    :math:`k ∈ [ 0.04 , 0.06 ]`

    Args:
        x: torch.Tensor: 4d tensor
        sigma_d (float): sigma of Gaussian derivative
        sigma_i (float): sigma of Gaussian blur, aka integration scale
        alpha (float): constant

    Return:
        torch.Tensor: Harris response

    Shape:
      - Input: :math:`(B, C, H, W)`
      - Output: :math:`(B, C, H, W)`
    """
    temp = spatial_gradient_first_order(x=x, sigma=sigma_d)
    Ix, Iy = temp[:, :, 0], temp[:, :, 1]
    Ixx = gaussian_filter2d(x=Ix ** 2, sigma=sigma_i)
    Ixy = gaussian_filter2d(x=Ix * Iy, sigma=sigma_i)
    Iyy = gaussian_filter2d(x=Iy ** 2, sigma=sigma_i)

    det_M = (Ixx * Iyy) - (Ixy ** 2)
    trace_M = (Ixx + Iyy) ** 2
    out = det_M - alpha * trace_M

    return out


def nms2d(x: torch.Tensor, th: float = 0):
    r"""Applies non maxima suppression to the feature map in 3x3 neighborhood.
    Args:
        x: torch.Tensor: 4d tensor
        th (float): threshold
    Return:
        torch.Tensor: nmsed input

    Shape:
      - Input: :math:`(B, C, H, W)`
      - Output: :math:`(B, C, H, W)`
    """
    out = torch.zeros_like(x)
    return out


def harris(x: torch.Tensor, sigma_d: float, sigma_i: float, th: float = 0):
    r"""Returns the coordinates of maximum of the Harris function.
    Args:
        x: torch.Tensor: 4d tensor
        sigma_d (float): scale
        sigma_i (float): scale
        th (float): threshold

    Return:
        torch.Tensor: coordinates of local maxima in format (b,c,h,w)

    Shape:
      - Input: :math:`(B, C, H, W)`
      - Output: :math:`(N, 4)`, where N - total number of maxima and 4 is (b,c,h,w) coordinates
    """
    harris_mat = harris_response(x, sigma_d=sigma_d, sigma_i=sigma_i)
    # To get coordinates of the responses, you can use torch.nonzero function
    out = torch.nonzero(harris_mat > th)
    return out


def create_scalespace(x: torch.Tensor, n_levels: int, sigma_step: float):
    r"""Creates a scale pyramid of image, usually used for local feature
    detection. Images are consequently smoothed with Gaussian blur.
    Args:
        x: torch.Tensor :math:`(B, C, H, W)`
        n_levels (int): number of the levels.
        sigma_step (float): blur step.

    Returns:
        Tuple(torch.Tensor, List(float)):
        1st output: image pyramid, (B, C, n_levels, H, W)
        2nd output: sigmas (coefficients for scale conversion)
    """

    image_pyramid = []
    sigmas = [1]

    for level in range(0, n_levels):
        sigma = sigmas[level] * sigma_step
        smoothed_image = gaussian_filter2d(x, sigma=sigma)
        image_pyramid.append(smoothed_image)
        sigmas.append(sigma)

    image_pyramid = torch.stack(image_pyramid, dim=2)  # Stack along the third dimension to create the pyramid tensor

    return image_pyramid, sigmas


def nms3d(x: torch.Tensor, th: float = 0):
    r"""Applies non maxima suppression to the scale space feature map in 3x3x3 neighborhood.
    Args:
        x: torch.Tensor: 5d tensor
        th (float): threshold
    Shape:
      - Input: :math:`(B, C, D, H, W)`
      - Output: :math:`(B, C, D, H, W)`
    """
    neighborhood = x.unfold(2, 3, 1).unfold(3, 3, 1).unfold(4, 3, 1)

    # Find the maximum value in each neighborhood
    max_values, _ = neighborhood.max(dim=(2, 3, 4,), keepdim=True)

    # Create a mask for values greater than or equal to the threshold
    mask = (x >= th) & (x == max_values)

    # Apply the mask to the original tensor
    output = x * mask.float()

    return output

def scalespace_harris_response(x: torch.Tensor,
                               n_levels: int = 40,
                               sigma_step: float = 1.1):
    r"""First computes scale space and then computes the Harris cornerness function 
    Args:
        x: torch.Tensor: 4d tensor
        n_levels (int): number of the levels, (default 40)
        sigma_step (float): blur step, (default 1.1)

    Shape:
      - Input: :math:`(B, C, H, W)`
      - Output: :math:`(B, C, N_LEVELS, H, W)`, List(floats)
    """
    out = torch.zeros_like(x)
    return out


def scalespace_harris(x: torch.Tensor,
                      th: float = 0,
                      n_levels: int = 40,
                      sigma_step: float = 1.1):
    r"""Returns the coordinates of maximum of the Harris function.
    Args:
        x: torch.Tensor: 4d tensor
        th (float): threshold
        n_levels (int): number of scale space levels (default 40)
        sigma_step (float): blur step, (default 1.1)
        
    Shape:
      - Input: :math:`(B, C, H, W)`
      - Output: :math:`(N, 5)`, where N - total number of maxima and 5 is (b,c,d,h,w) coordinates
    """
    # To get coordinates of the responses, you can use torch.nonzero function
    # Don't forget to convert scale index to scale value with use of sigma
    image_pyramid, sigmas = create_scalespace(x, n_levels, sigma_step)
    harris_responses_coords = []

    for scale_level in range(0, n_levels):
        coords = harris(image_pyramid[:, :, scale_level], sigma_d=sigmas[scale_level], sigma_i=sigmas[scale_level], th=th)
        harris_responses_coords.append(coords)

    # Stack the list of coordinates along the first dimension to get the final tensor
    harris_responses_coords = torch.cat(harris_responses_coords, dim=0)
    harris_responses_coords = torch.cat((torch.ones((harris_responses_coords.size(0), 1)), harris_responses_coords), dim=1)
    return harris_responses_coords
