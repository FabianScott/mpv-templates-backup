import os.path
import cv2
import numpy as np
import typing
from types import SimpleNamespace

from matplotlib import pyplot as plt


def fft2(x):
    return np.fft.fft2(x, axes = [0, 1])

def ifft2(x): 
    return np.fft.ifft2(x, axes = [0, 1])

def read_image(idx): 
    fname = os.path.join('data', '%03i.jpg' % idx)
    img = cv2.imread(fname).astype(float)/255.0
    # BGR -> RGB: 
    img = img[:,:,[2,1,0]]
    # unlike in KLT, we work with color images
    return img

def kcf_train(x: np.array, # training image
              y: np.array, # desired response
              pars: SimpleNamespace):

    k = kernel_correlation(x, x, pars)
    alphaf = fft2(y) / (fft2(k) + pars.lam)
    return alphaf 
 
def kcf_detect(alphaf: np.array, # tracker filter
               x: np.array, # training image
               z: np.array, # next image
               pars: SimpleNamespace):

    k = kernel_correlation(x, z, pars);
    responses = ifft2(alphaf * fft2(k));
    # return the response, remove imag components which 
    # are of order 1e-16 and are there only due to numerical 
    # imprecision: 
    return responses.real
 
def kernel_correlation(x1: np.array, 
                       x2: np.array, 
                       pars: SimpleNamespace): 

    c = ifft2( ( fft2(x1).conj() * fft2(x2) ).sum(axis=2) )
    c = c.real 
    d = (x1**2).sum() + (x2**2).sum() - 2 * c;

    if pars.kernel_type == 'rbf': 
        k = np.exp( (-1 / pars.rbf_sigma**2) * d / d.size);
    elif pars.kernel_type == 'linear': 
        k = c;
    else: 
        raise Exception('Unknown kernel type: {}'.format(pars.kernel_type))   

    return k

def kcf_init(img, bbox, pars): 
    
    # tracker state: 
    S = SimpleNamespace()

    # copy init bbox: 
    S.x, S.y, S.w, S.h = bbox
    w, h = S.w, S.h

    # define where to place the center of response
    cx, cy = int( w/2 ), int( h/2 )
    # compute target response: 
    sigma = max(w, h)/10.0
    x, y = np.meshgrid(np.arange(0, w) - cx, np.arange(0, h) - cy)
    response = np.exp( -(x**2 + y**2)/(2*sigma**2) )
    
    # store ideal response and its center: 
    S.y_response = response
    S.cx = cx
    S.cy = cy 
    # ==== 

    # compute cosine/uniform window for the given size of image: 
    if pars.envelope_type == 'cos': 
        wx = (1 + np.cos( x / (w/2.0) * np.pi ))/2.0
        wy = (1 + np.cos( y / (h/2.0) * np.pi ))/2.0
        window = wx * wy 
    elif pars.envelope_type == 'uniform': # uniform window
        window = np.ones_like(x)
    else: 
        raise Exception('Unknown window weighting type: {}'.format(pars.envelope_type))   
    S.envelope = np.repeat(window[:,:,np.newaxis], 3, axis=2)
    
    # initial training patch: 
    x, y, w, h = bbox
    S.x_train = img[y:y+h, x:x+w, :]
 
    # the init tracker filter: 
    S.alphaf = kcf_train(S.x_train * S.envelope, S.y_response, pars)

    return S

def track_kcf(img_next: np.array, 
              S: SimpleNamespace, 
              pars: SimpleNamespace):

    x, y, w, h = S.x, S.y, S.w, S.h
    z = img_next[y:y+h, x:x+w, :]

    # compute responses for z
    responses = kcf_detect(S.alphaf, S.x_train * S.envelope, z * S.envelope, pars)
    # find the location of the maximum in the responses, 
    # From that, compute how the patch has shifted from the previous frame: 
    dy, dx = np.unravel_index(np.argmax(responses), responses.shape[:2])
    dx -=  (responses.shape[0] // 2)
    dy -=  (responses.shape[1] // 2)
    # if dy > h // 2:
    #     dy -= h
    # if dx > w // 2:
    #     dx -= w
    # update the position of the bbox: 
    S.x += dx
    S.y += dy

    fig, axs = plt.subplots(2, 3)
    axs[0, 0].set_title(f'Train Image')
    axs[0, 0].imshow(S.x_train)
    axs[1, 0].set_title(f'Next Image')
    axs[1, 0].imshow(img_next)
    axs[1, 0].scatter([x], [y], color='blue', s=50, alpha=.7)
    axs[1, 0].scatter([S.x], [S.y], color='red', s=50, alpha=.7)
    axs[0, 1].set_title(f'Initial patch, x: {x}, y: {y}')
    axs[0, 1].imshow(z)

    # extract the patch from the current image again, using
    # new estimate of bbox position 
    x, y = S.x, S.y
    patch_next = img_next[y:y+h, x:x+w, :]

    axs[0, 2].set_title(f'Response, dx: {dx}, dy: {dy}')
    axs[0, 2].imshow(responses, cmap='gray')
    axs[1, 2].set_title(f'Patch Next, y: {y}, x: {x}')
    axs[1, 2].imshow(patch_next)
    plt.tight_layout()
    plt.show()
    # update the training image S.x_train by patch_next (with adaptation) 
    gamma = pars.gamma
    S.x_train = (1 - gamma) * S.x_train + gamma * patch_next * S.envelope

    # recompute alphaf for the updated training image: 
    S.alphaf = kcf_train(S.x_train, S.y_response, pars)

    # keep responses and current patch in S, to be easily accessible 
    # for inspection
    S.responses = responses
    S.patch_next = patch_next 
    return dx, dy # just for easy access for testing
