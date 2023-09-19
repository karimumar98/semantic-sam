import numpy as np
import torch

from scipy.stats import truncnorm
import random

def get_truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)


def mutate_bbox (bbox, side_choice = "mean"):
    ## Following description of paper, we randomly mutate the size of the bounding box
    bbox = torch.clone(bbox)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]

    if side_choice == "mean":
        l = (w + h)/2.0
    elif side_choice == "max":
        l = max(w, h)
    elif side_choice == "min":
        l = min(w, h)
    else:
        raise NotImplementedError()

    sampler = get_truncated_normal(mean = 0, sd = 0.1*l, low = -20, upp = 20)
    for i in range(4):
        ## Add to each coordinate a randomly sampled deviation, sampling should have std 10% of the length of the bbox (max 20 pixels)
        bbox[i] += sampler.rvs()

    return bbox

def get_random_point_in_mask (masks, foreground):
    '''
        Input masks is a BxHxW tensor
        Output is a Bx2 Tensor, where each of the B entries is a random index within the associated mask
    '''
    ## selects a point uniformly at random within the mask's foreground, the point is return as a point in [0,w]x[0,h] where w,h are the width and height of the input mask
    if not foreground:
        ## TODO: Is this efficient??
        mask = mask == False
    

    idx_choices = [torch.argwhere(mask) for mask in masks]
    coords = [random.choice(idx_c) for idx_c in idx_choices]
    # breakpoint()
    return torch.stack([x.flip(dims=(0,)) for x in coords]).unsqueeze(dim=1)
    return torch.stack(coords).unsqueeze(dim=1)
    


def get_next_points(pred_masks, actual_masks):

    #breakpoint()
    n = pred_masks.shape[0]

    pred_masks[pred_masks > 0.0] = 1.0
    pred_masks[pred_masks <= 0.0] = 0.0

    actual_masks = actual_masks.float()
    error = pred_masks - actual_masks.to(pred_masks.device)

    # a zero in the error matrix means that at this pixel, the model's ouput and GT are the same
    # -1 means prediction missed this pixel (false negative)
    # +1 means predicted part of mask, but not in GT (false positive)

    # as in SAM, we sample a random point either in the +1 or -1 region with equal probability
    points = []
    labels = []

    for i in range(n):
        choices = []
        if (error[i] == -1).nonzero().shape[0] != 0:
            choices += [0]

        if (error[i] == 1).nonzero().shape[0] != 0:
            choices += [1]
        #breakpoint()
        assert choices != [], "prediction is perfect"
        k = random.choice(choices)
        if k == 0:
            indices = (error[i] == -1).nonzero()
            #print(indices.shape)
            p = random.choice(indices)
            l = 0
        else:
            indices = (error[i] == 1).nonzero()            
            #print(indices.shape)
            p = random.choice(indices)
            l = 1
        points += [p]
        labels += [l]
    #breakpoint()
    points = torch.stack(points).unsqueeze(1)
    labels = torch.tensor(labels).unsqueeze(1)

    # return as [n,1,2] and [n,1] tensors
    return points, labels

           

    # for each mask we sample a random point within the error region

    ## Used in iterative training to determine the next point prompt, point will be in the error region of the predicted mask
    raise NotImplementedError()
