import torch
import random
from torch.distributed.nn.functional import all_gather
from .multinode_utils import all_gather_with_gradient

from .train_utils import *
import torchvision
import time

'''
    This module contains all of the different types of training steps that can be performed

'''
def get_train_step_function (train_step_function_name):
    # Used to initialize training from cfg, returns a pointer to the train step used
    train_steps_map = {
        "train_step" : train_step,
        "train_step_general" : train_step_general,
        "train_step_fine_tune_coco" : train_step_fine_tune_coco
    }
    return train_steps_map[train_step_function_name]


def remove_key(d, key):
    ## Quick fix for removing key from dict
    del d[key]
    return d


def train_step_general(
    model, 
    batched_inputs, 
    criterion, 
    idx = 0,
    **kwargs):
    
    # *General train step implementation
    model.train()
    # We get a mixed batch, the first inputs are semantic, then SA-1B
    semantic_input, mask_input = batched_inputs
    len_semantic, len_mask = len(semantic_input), len(mask_input)
    batched_inputs = semantic_input + mask_input


    mask_resize = torchvision.transforms.Resize((256, 256))

    ## Decide what kind of a prompt is used

    for i in range(len(batched_inputs)):
        # For each sample in the batch, randomly choose a prompt type
        prompt_choice, weights = ["bbox"], [1.0]
        sample = batched_inputs[i]

        if sample["ground_truth_type"] == "mask":
            prompt_choice += ["point"]
            weights += [1.0]

        # if sample["ground_truth_type"] == "semantic" and "whole_image" in kwargs and kwargs["whole_image"]: 
        #     # We only wanna do the whole image stuff for semantic inputs
        #     prompt_choice += ["whole_image"]
        #     weights += [kwargs["whole_image_prob"]]

        if sample["ground_truth_type"] == "semantic" and "text_prompt" in kwargs and kwargs["text_prompt"]: 
            # We only wanna do the whole image stuff for semantic inputs
            prompt_choice += ["text"]
            weights += [kwargs["text_prompt_prob"]]

        if kwargs["multiprompt"]:
            # Use potentially multiple prompts in training loop, like SAM
            num_proompts = len(prompt_choice)

            ## Make it skewed towards fewer prompts
            w = [1/(x+1) for x in range(num_proompts)]
            ## Shuffle choices
            random.shuffle(prompt_choice)
            k = random.choices([i for i in range(1, num_proompts + 1)], weights = w, k = 1)[0] ## Choose random number of proompts starting from 1
            proompt_type = random.sample(prompt_choice, k = k)
            
        else:
            # Make a weighted choice of prompt types
            proompt_type = random.choices(prompt_choice, weights = weights, k= 1)

        if "point" in proompt_type:
            # And add a random point in foreground
            sample = dict({
                ## This needs to be num_masks_per_batchx1x2 -> since each point is a seperate query
                "point_coords" : get_random_point_in_mask(sample["masks"], foreground = True),
                ## This needs to be num_masks_per_batchx1 --> since each point is a seperate query
                "point_labels" : torch.ones(sample["masks"].shape[0], 1)
            }, **sample)

        if "text" in proompt_type:
            # Remove BBox if we have a semantic target and set the clip embeding as the prompt
            sample = dict({"semantic_prompt" : sample["semantic_targets"]})

        if "bbox" in proompt_type:
            if "mutate_bbox" in kwargs and kwargs["mutate_bbox"]:
                ## Modify BBoxes size like SAM paper
                sample["box_proompts"] = torch.stack([mutate_bbox(box) for box in sample["boxes"]])
            else:                
                sample["box_proompts"] = torch.stack([box for box in sample["boxes"]])

            # TODO: should we replace all boxes, or just add one to the, should we add it randomly or in every iter???
            #if "whole_image" in prompt_choice:
            if sample["ground_truth_type"] == "semantic" and "whole_image" in kwargs and kwargs["whole_image"]:
                ## TODO: Also maybe pick a random point in theimage
                # Replace the bbox with a bbox that spans whole image (Image size is stored in h,w so flip)
                max_dim = max(sample['image_size'][1], sample['image_size'][0])
                x,y = sample['image_size'][1]/max_dim, sample['image_size'][0]/max_dim
                whole_image_prompt = torch.tensor([0.0, 0.0, x*1024, y*1024]).unsqueeze(0)
                # The target is now the embedding of the whole image annotation

                sample["box_proompts"] = torch.cat((sample["box_proompts"], whole_image_prompt))
                sample["semantic_targets"] = torch.cat((sample["semantic_targets"], sample["image_level_features"].unsqueeze(0)))

                #TODO: Fix
                semantic_input[i]['boxes'] = torch.cat((semantic_input[i]['boxes'], whole_image_prompt))
        batched_inputs[i] = sample


    # BxNxclip_dim where N is the number of masks per image
    semantic_target = [x['semantic_targets'] for x in semantic_input]
    ## Add bbox as a target to aid localitazion loss during semantic training
    semantic_box_target = [x['boxes'] for x in semantic_input]
    # BxNx1024x1024 where N is the number of masks per image
    masks = [x['masks'] for x in mask_input]
        # breakpoint()
    ## Pass all inputs through in single pass

    if "multimask" in kwargs and kwargs["multimask"]:
        multimask_output = True
    else:
        multimask_output = False

    # Now actually do a forward pass of the model
    outputs, logits_scale = model(batched_inputs, multimask_output)
        

    pred_semantic = [x["semantic_features"] for x in outputs]
    pred_masks = [x["masks"] for x in outputs]
    pred_iou = [x["iou_predictions"] for x in outputs]
    pred_cosine = [x["cosine_predictions"] for x in outputs]

    # Record number of gt masks per sample, needed for downstream applications
    num_masks_per_sample = [x['semantic_targets'].shape[0] for x in semantic_input]

    ## Split output so that we only calculate loss on the correct part of the output
    #Semantic Part
    pred_semantic = pred_semantic[:len_semantic]
    pred_semantic_masks = pred_masks[:len_semantic]
    pred_cosine = pred_cosine[:len_semantic]
    #SA-1B part
    pred_masks = pred_masks[len_semantic:]
    pred_iou = pred_iou[len_semantic:]

    ## concat targets and predictions to calc loss
    pred_semantic = torch.cat(pred_semantic)
    pred_iou = torch.cat(pred_iou)
    pred_cosine = torch.cat(pred_cosine)
    pred_semantic_masks = torch.cat(pred_semantic_masks)
    semantic_box_target = torch.cat(semantic_box_target).to(model.device)
    pred_masks = torch.cat(pred_masks)
    semantic_target = torch.cat(semantic_target).to(model.device)
    masks = torch.cat(masks).to(model.device)


    # Construct a dict of all of the model outputs and targets to pass to criterion
    model_outputs = {
        "target_masks" : masks,
        "pred_masks" : pred_masks,
        "semantic_target": semantic_target,
        "pred_semantic" : pred_semantic,
        "pred_iou" : pred_iou,
        "num_masks_per_sample" : num_masks_per_sample,
        "pred_cosine" : pred_cosine,
        "logits_scale" : logits_scale
    }

    losses = criterion(model_outputs)
    return losses


  

def train_step_fine_tune_coco(
    model, 
    batched_inputs, 
    criterion, 
    idx = 0,
    **kwargs):

    
    # *General train step implementation
    model.train()
    #breakpoint()
    coco_input = batched_inputs[0]


    ## Decide what kind of a prompt is used
    # for debugging:
    model_input = []
    debug_prompt_types = []

    #breakpoint()
    for i in range(len(coco_input)):
        # For each sample in the batch, randomly choose a prompt type
        
        # prompt_choice, weights = ["bbox", "point"], [1.0, 1.0]
        prompt_choice, weights = ["point"], [1.0]
        sample = coco_input[i]


        if kwargs["multiprompt"]:
            # Use potentially multiple prompts in training loop, like SAM
            num_proompts = len(prompt_choice)
            ## Make it heavily biased towards fewer prompts
            w = [1/(x+1) for x in range(num_proompts)]
            ## Shuffle choices
            random.shuffle(prompt_choice)
            k = random.choices([i for i in range(1, num_proompts + 1)], weights = w, k = 1)[0] ## Choose random number of proompts starting from 1

            proompt_type = random.sample(prompt_choice, k = k)
            
        else:
            # Make a weighted choice of prompt types
            proompt_type = random.choices(prompt_choice, weights = weights, k= 1)


        if "point" in proompt_type:
            # And add a random point in foreground
            #breakpoint()
            #breakpoint()
            sample = dict({
                ## This needs to be num_masks_per_batchx1x2 -> since each point is a seperate query
                "point_coords" : get_random_point_in_mask(sample["masks"], foreground = True),
                ## This needs to be num_masks_per_batchx1 --> since each point is a seperate query
                "point_labels" : torch.ones(sample["masks"].shape[0], 1)
            }, **sample)

        if "bbox" in proompt_type:
            if "mutate_bbox" in kwargs and kwargs["mutate_bbox"]:
                ## Modify BBoxes size like SAM paper
                sample["box_proompts"] = torch.stack([mutate_bbox(box) for box in sample["boxes"]])
            else:                
                sample["box_proompts"] = torch.stack([box for box in sample["boxes"]])
        #breakpoint()

        model_input.append(sample)
        debug_prompt_types += [proompt_type]

    #breakpoint()

    # BxNxclip_dim where N is the number of masks per image
    semantic_target = [x['semantic_targets'] for x in coco_input]
    ## Add bbox as a target to aid localitazion loss during semantic training
    semantic_box_target = [x['boxes'] for x in coco_input]
    # BxNx1024x1024 where N is the number of masks per image
    masks = [x['masks'] for x in coco_input]
        # breakpoint()

    if "multimask" in kwargs and kwargs["multimask"]:
        #raise NotImplementedError()
        multimask_output = True
    else:
        multimask_output = False

    if "iterative" in kwargs and kwargs["iterative"]:
        iterative = True
    else:
        iterative = False

    #breakpoint()

    if not iterative:
        # Just pass through once
        outputs, logits_scale = model(model_input, multimask_output)
    else:
        prev_mask_outputs = [None for _ in batched_inputs] ## Initial input should be empty
        prev_semantic_output = [None for _ in batched_inputs] ## Initial semantic output should also be empty
        prev_iou_output = [None for _ in batched_inputs] ## Initial semantic output should also be empty
        num_clicks = kwargs["num_clicks"]
        for i in range(num_clicks):
            with torch.no_grad():
                ## Add next point and Mask to the batched inputs

                ## Pass all inputs through in single pass
                # TODO: For now, only single mask output is used
                #breakpoint()
                outputs = model(batched_inputs, False)

                pred_semantic = [x["semantic_features"] for x in outputs]

                pred_masks = [x["masks"] for x in outputs[len_semantic:]]
                pred_masks_semantic = [x["masks"] for x in outputs[:len_semantic]]
                pred_iou = [x["iou_predictions"] for x in outputs[len_semantic:]]

                # Sample a random point in the error region between GT mask and predicte    
                #       breakpoint()
                next_point_label = [get_next_points(a,b) for a,b in zip(pred_masks, masks)]
                #breakpoint()
                next_points, next_labels = zip(*next_point_label)

                # Construct next_input for mask data

                next_mask_input = [
                    {
                        "image" : sample['image'],
                        "mask_prompt" : mask_resize(mask.unsqueeze(1)),
                        "point_coords" : next_point,
                        "point_labels" : next_label
                    } for sample, mask, next_point, next_label in zip(mask_input, pred_masks, next_points, next_labels)
                ]

                

                next_semantic_input = [
                    {
                        "image" : sample['image'],
                        "mask_prompt" : mask_resize(mask.unsqueeze(1)),
                        "semantic_prompt": semantic_prompt.squeeze(dim=1)
                    } for sample, mask, semantic_prompt in zip(semantic_input, pred_masks_semantic, pred_semantic)
                ]
                #breakpoint()
                batched_inputs = next_semantic_input + next_mask_input

            
        # Final pass, actually pass through with gradient
        #breakpoint()
        outputs = model(batched_inputs, False)


    pred_semantic = [x["semantic_features"] for x in outputs]
    pred_masks = [x["masks"] for x in outputs]
    pred_iou = [x["iou_predictions"] for x in outputs]
    #breakpoint()
    num_masks_per_sample = [x.shape[0] for x in semantic_target]


    ## concat targets and predictions to calc loss
    pred_semantic = torch.cat(pred_semantic)
    pred_iou = torch.cat(pred_iou)
    semantic_box_target = torch.cat(semantic_box_target).to(model.device)

    #breakpoint()
    pred_masks = torch.cat(pred_masks)
    semantic_target = torch.cat(semantic_target).to(model.device)
    masks = torch.cat(masks).to(model.device)


    pred_cosine = [x["cosine_predictions"] for x in outputs]
    pred_cosine = torch.cat(pred_cosine)


    # Construct a dict of all of the model outputs and targets to pass to criterion
    model_outputs = {
        "target_masks" : masks,
        "pred_masks" : pred_masks,
        "semantic_target": semantic_target.float(),
        "pred_semantic" : pred_semantic,
        "pred_iou" : pred_iou,
        "num_masks_per_sample" : num_masks_per_sample,
        "pred_cosine" : pred_cosine,
        "logits_scale" : logits_scale
        ## Currently not used
        # "pred_semantic_masks" : pred_semantic_masks
    }



    #breakpoint()
    losses = criterion(model_outputs)

    return losses

    #breakpoint()

    


    optimizer.zero_grad()
    losses["total_loss"].backward()
    optimizer.step()
    

    return losses