from torch import nn
from torch.nn.functional import threshold, normalize, pad, interpolate, binary_cross_entropy_with_logits, mse_loss, softmax, cross_entropy
import torch
from torchvision.ops import focal_loss, masks_to_boxes, generalized_box_iou_loss
import numpy as np
from torch.distributed.nn.functional import all_gather
from torchmetrics import JaccardIndex
import torch

import sys
sys.path.append("..")
from segment_anything.utils.config import configurable
from segment_anything.utils.multinode_utils import unequal_all_gather, all_gather_with_gradient

import time


class Criterion(nn.Module):
    ## Aggregated all loss related functionalities here for better searching of optimal loss
    # The different losses passed as a list
    @configurable
    def __init__(self, loss_fns = [], loss_weights = [], global_loss = False, reduction_strategy = "coupled", buffer_size = 512, dist = False, async_all_gather = False):
        super(Criterion, self).__init__()

        self.loss_fns = loss_fns
        self.loss_weights = loss_weights
        self.reduction_strategy = reduction_strategy
        self.global_loss = global_loss
        self.buffer_size = buffer_size
        print("crit buffer size: ", buffer_size)
        self.dist = dist
        self.async_all_gather = async_all_gather

    @classmethod
    def from_config (cls, cfg):
        loss_cfg = cfg["training"]['loss']
        global_loss = loss_cfg["global_loss"]
        reduction_strategy = loss_cfg['loss_reduction']
        ## TODO: tidy this up
        if "COCO" in cfg['DATA_LOADERS']:
            buffer_size = cfg["DATA_LOADERS"]["COCO"]["num_masks_per_image"] * cfg["DATA_LOADERS"]["COCO"]["batch_size"] #* (3 if cfg["training"]["train_args"]['multimask'] else 1)
        elif "SEGINW" in cfg['DATA_LOADERS']:
            buffer_size = cfg["DATA_LOADERS"]["SEGINW"]["num_masks_per_image"] * cfg["DATA_LOADERS"]["SEGINW"]["batch_size"] * (3 if cfg["training"]["train_args"]['multimask'] else 1)
        else:
            buffer_size = (1 + cfg["DATA_LOADERS"]["LAION"]["num_masks_per_image"] )* cfg["DATA_LOADERS"]["LAION"]["batch_size"]
        dist = cfg["distributed"]['use_deistributed']
        #breakpoint()
        loss_fns = []
        loss_weights = []

        try:
            async_all_gather = loss_cfg['async_all_gather']
        except:
            async_all_gather = False

        for loss_name in loss_cfg['loss_functions']:
            l = loss_cfg['loss_functions'][loss_name]
            weight = l.pop('weight')
            loss_fns.append(
                globals()[loss_name](**l)
            )
            loss_weights.append(weight)
            #breakpoint()

        return {
            "loss_fns" : loss_fns,
            "loss_weights" : loss_weights,
            "reduction_strategy" : reduction_strategy,
            "global_loss": global_loss,
            "buffer_size" : buffer_size,
            "dist" : dist,
            "async_all_gather" : async_all_gather
        }


    def add_loss (self, loss_dict, new_loss, weight):
        # Helper function, iterates over new loss values and adds them to existing loss weighted
        for key in new_loss:
            loss_dict[key] = loss_dict.get(key, 0.0) + new_loss[key] * weight

        return loss_dict

    def global_all_gather(self, model_outputs):
        # Wrapper to handle all distributed gathering of outputs
        for key in model_outputs:
            # Currently only use semantic stuff for global loss
            if "semantic" in key or "cosine" in key:
                model_outputs[key] = unequal_all_gather (model_outputs[key], buffer_size = self.buffer_size, async_all_gather = self.async_all_gather)
            elif "num_masks_per_sample" in key:
                model_outputs[key] = unequal_all_gather (torch.tensor(model_outputs[key]).to("cuda"), buffer_size = len(model_outputs[key])).to("cpu").int().tolist()[0]
        return model_outputs



    def forward(self, model_outputs):
        if self.dist and self.global_loss:
            model_outputs = self.global_all_gather (model_outputs)

        loss_dict = {}
        for loss_fn, loss_weight in zip(self.loss_fns, self.loss_weights):
            loss_dict = self.add_loss(
                loss_dict,
                loss_fn(**model_outputs),
                loss_weight
            )

        if self.reduction_strategy == "uncoupled":
            total_loss = 0.0
            for key in loss_dict:
                #breakpoint()
                loss_dict[key] = torch.mean(torch.min(loss_dict[key], dim=1).values)
                total_loss += loss_dict[key]
            loss_dict['total_loss'] = total_loss            

        elif self.reduction_strategy == "coupled":
            # If the reduction is coupled, we have to also all_gather the mask loss for addition
            if self.dist and self.global_loss:
                loss_dict['mask_loss'] = all_gather_with_gradient(loss_dict['mask_loss'])
                
            loss_dict['total_loss'] = loss_dict['mask_loss'] + loss_dict['semantic_loss']
            val, ind = torch.min(loss_dict['total_loss'], dim=1)
            loss_dict['total_loss'] = torch.mean(val)
            loss_dict['mask_loss'] = torch.stack([loss_dict['mask_loss'][i, j] for i, j in enumerate(ind)])
            loss_dict['semantic_loss'] = torch.stack([loss_dict['semantic_loss'][i, j] for i, j in enumerate(ind)])

            loss_dict['mask_loss'] = torch.mean(loss_dict['mask_loss'])
            loss_dict['semantic_loss'] = torch.mean(loss_dict['semantic_loss'])

        else:
            raise NotImplementedError()        

        return loss_dict


class DiceLoss(nn.Module):
    def __init__(self, eps=1e-7, reduction = "mean"):
        super(DiceLoss, self).__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, logits, targets):
        # TODO Double check
        #breakpoint()
        num = targets.size(0)
        prob = torch.sigmoid(logits)
        prob = prob.view(num, -1)
        targets = targets.view(num, -1)
        intersection = (prob * targets).sum(1).float()
        dice = 2.0 * (intersection + self.eps) / (prob.sum(1) + targets.sum(1) + self.eps)
        loss = 1 - dice
        if self.reduction == "mean":
            return loss.mean()
        return loss

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        #breakpoint()
        targets = targets.float()
        p = torch.sigmoid(logits)
        ce_loss = binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss

        if self.reduction == "none":
            pass
        elif self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        else:
            raise ValueError(
                f"Invalid Value for arg 'reduction': '{self.reduction} \n Supported reduction modes: 'none', 'mean', 'sum'"
            )

        return loss

class CombinedLoss(nn.Module):
    def __init__(self, dice_weight=1, focal_weight=20):
        super(CombinedLoss, self).__init__()
        self.dice_loss = DiceLoss(reduction = "none")
        self.focal_loss = FocalLoss(reduction = "none")
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight

    def forward(self, target_masks, pred_masks, **kwargs):


        if pred_masks.dim() != 3:
            bs, n = pred_masks.shape[:2]
            pred_masks = pred_masks.flatten(0,1)
            target_masks = target_masks.repeat_interleave(n, dim = 0)
        else:
            n = 1
            bs = pred_masks.shape[0]
        #breakpoint()

        dice = self.dice_loss(pred_masks, target_masks).reshape(bs, n)
        focal = self.focal_loss(pred_masks, target_masks).mean(dim=(1,2)).reshape(bs, n)
        return {
            "mask_loss" : self.dice_weight * dice + self.focal_weight * focal
        }
    

class MSE_Loss (nn.Module):
    def __init__(self, temperature=1.0):
        super(MSE_Loss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.jaccard = JaccardIndex(task = "binary").to("cuda")

    def forward(self, pred_iou, target_masks, pred_masks, **kwargs):

        bs, n = pred_iou.shape
        if pred_masks.dim() == 4:
            pred_masks = pred_masks.flatten(0,1)

        target_masks = target_masks.repeat_interleave(n, dim = 0)
        #first calculate actual iou
        actual_iou = torch.stack([self.jaccard(a,b) for a,b in zip(pred_masks, target_masks)])
        actual_iou = actual_iou.reshape(bs, n)

        squared_error = (pred_iou - actual_iou)**2

        return {
            "mask_loss" : squared_error
        }

class Cosine_MSE_Loss (nn.Module):
    def __init__(self, temperature=1.0):
        super(Cosine_MSE_Loss, self).__init__()
        self.mse_loss = nn.MSELoss()


    def forward(self, pred_cosine, pred_semantic, semantic_target, **kwargs):


        bs, n = pred_cosine.shape
        if pred_semantic.dim() == 3:
            pred_semantic = pred_semantic.flatten(0,1)

        semantic_target = semantic_target.repeat_interleave(n, dim = 0)

        #first calculate actual cs
        actual_cs = torch.nn.functional.cosine_similarity(pred_semantic,semantic_target)
        actual_cs = actual_cs.reshape(bs, n)

        squared_error = (pred_cosine - actual_cs)**2

        return {
            "semantic_loss" : squared_error
        }



class SymmetricContrastiveLoss(torch.nn.Module):
    
    def __init__(self, temperature):
        super(SymmetricContrastiveLoss, self).__init__()
        self.temperature = np.exp(temperature)
    
    def forward(self, semantic_target, pred_semantic, logits_scale, **kwargs):


        ## Ghetto handle legacy input in format [B, clip_dim]
        if len(pred_semantic.shape) == 2:
            # Artificially add 1-way multimask output
            pred_semantic = pred_semantic.unsqueeze(dim=1)
        bs, n, _ = pred_semantic.shape

        semantic_target = normalize(semantic_target, dim=1)
        pred_semantic = normalize(pred_semantic, dim=2)

        pred_semantic = pred_semantic.flatten(0,1)

        # scaled pairwise cosine similarities [n, n]
        logits = semantic_target @ pred_semantic.T * self.temperature

        labels = torch.diag(torch.ones(bs)).repeat_interleave(n,1).to(logits.device)
        target_loss = cross_entropy(logits, labels)

        labels = torch.arange(semantic_target.shape[0]).repeat_interleave(n).to(semantic_target.device) # Shape: [bs, bs*n]
        text_loss = cross_entropy(logits.T, labels, reduction='none').reshape(bs, n)
        loss = (target_loss + text_loss) / 2


        return {"semantic_loss" : loss}

        ## TODO: Make sure this loss is correct lol
    




    
            




