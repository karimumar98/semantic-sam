import argparse
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import os
from itertools import islice
import tqdm
import builtins

## Distributed imports
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from socket import gethostname
from segment_anything.utils.transforms import ResizeLongestSide
from segment_anything.utils.config import read_config
from segment_anything.utils.optimizer import build_optimizer
import webdataset as wds
import numpy as np
import random
import json

import wandb
from segment_anything.eval import seginw


from torch.utils.tensorboard import SummaryWriter

## Custom imports
import sys
# sys.path.append("/cluster/project/zhang/umarka/clip_detector/semantic-segment-anything/segment_anything")
from segment_anything.modeling.frozensam import FrozenSAM
from segment_anything.utils.data import get_data_loaders, mixed_dataloader
from segment_anything.utils.loss import DiceLoss, FocalLoss, CombinedLoss, ContrastiveLoss, SymmetricContrastiveLoss, MultiNodeGlobalLoss, MSE_Loss, GeneralizedContrastiveLoss, MaskedContrastiveLoss, SimpleLoss, Criterion
from segment_anything.utils.multinode_utils import supress_output, setup_multinode_training, all_gather_with_gradient
from segment_anything.utils.train_steps import get_train_step_function
from datetime import timedelta

from hanging_threads import start_monitoring
# start_monitoring(seconds_frozen=100, test_interval=100)

from segment_anything.eval import eval_coco, Capturing
# from segment_anything.eval import seginw
from torch import nn

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default=None, type=str)
    parser.add_argument('-d', '--debug', default=None, type=str)
    parser.add_argument('-e', '--eval_start', action='store_true')
    parser.add_argument('-w', '--suppress_wand', action='store_true')
    parser.add_argument('-p', '--suppress_print', action='store_false')
    

    args = parser.parse_args()

    ## Some controls to make sure input is valid

    return args

    

def main():
    
    args = parse_args()
    assert args.config, "Please provide a config file"
    
    cfg = read_config(args.config)

    #model = FrozenSAM(cfg)
    


    ## Setup either multinode or single node training
    if cfg["distributed"]['use_deistributed'] and 'WORLD_SIZE' in os.environ:
        print("SETTING UP DISTRIBUTED TRAINING")
        args.world_size, args.rank, args.local_rank = setup_multinode_training(
            cfg["training"]['seed'],
            cfg["distributed"]['backend']
        )
        print("after init model", args.rank, torch.cuda.mem_get_info())

        ## Distribute data among the nodes:
        if 'LAION' in cfg['DATA_LOADERS']:
            cfg['DATA_LOADERS']['LAION']['num_train_samples_per_epoch'] = cfg['DATA_LOADERS']['LAION']['num_train_samples_per_epoch'] // args.world_size
        # suppress printing if not on master process 
        if args.rank!=0 and args.suppress_print: supress_output()
        device_ids=[args.local_rank]

        model = FrozenSAM(**cfg["model"])

        data_loader_args = {
            "std" : model.sam.pixel_std.cpu().numpy(),
            "mean" : model.sam.pixel_mean.cpu().numpy(),
            "image_size": model.sam.image_encoder.img_size,
        }
        #print(model.device)
        model = DDP(model, device_ids=device_ids, find_unused_parameters=True)
        train_params = []
        if "text_decoder" in cfg["training"]['train_params'] and cfg["training"]['train_params']["text_decoder"]:
            print("TRAINING: text_decoder")
            train_params += list(model.module.text_decoder.parameters())
        if "sam_image_encoder" in cfg["training"]['train_params'] and cfg["training"]['train_params']["sam_image_encoder"]:
            print("TRAINING: image_encoder")
            train_params += list(model.module.sam.image_encoder.parameters())
        if "sam_prompt_encoder" in cfg["training"]['train_params'] and cfg["training"]['train_params']["sam_prompt_encoder"]:
            print("TRAINING: prompt_encoder")
            train_params += list(model.module.sam.prompt_encoder.parameters())
        if "sam_mask_decoder" in cfg["training"]['train_params'] and cfg["training"]['train_params']["sam_mask_decoder"]:
            print("TRAINING: mask_decoder")
            train_params += list(model.module.sam.mask_decoder.parameters())
        if "text_encoder" in cfg["training"]['train_params'] and cfg["training"]['train_params']["text_encoder"] and model.module.text_encoder is not None:
            print("TRAINING: text_encoder")
            train_params + list(model.module.text_encoder.parameters())

        ## Add temp
        train_params += [model.module.temperature]

    else:
        ## Mainly for debugging, vanilla single node training, wandb gets disabled
        print("NOT TRAINING WITH MULTIPLE NODES")
        args.world_size, args.rank, args.local_rank = 1, 0, 0
        model = FrozenSAM(**cfg["model"])
        data_loader_args = {
            "std" : model.sam.pixel_std.cpu().numpy(),
            "mean" : model.sam.pixel_mean.cpu().numpy(),
            "image_size": model.sam.image_encoder.img_size,
        }
        train_params = []
        if "text_decoder" in cfg["training"]['train_params'] and cfg["training"]['train_params']["text_decoder"]:
            print("TRAINING: text_decoder")
            train_params += list(model.text_decoder.parameters())
        if "sam_image_encoder" in cfg["training"]['train_params'] and cfg["training"]['train_params']["sam_image_encoder"]:
            print("TRAINING: image_encoder")
            train_params += list(model.sam.image_encoder.parameters())
        if "sam_prompt_encoder" in cfg["training"]['train_params'] and cfg["training"]['train_params']["sam_prompt_encoder"]:
            print("TRAINING: prompt_encoder")
            train_params += list(model.sam.prompt_encoder.parameters())
        if "sam_mask_decoder" in cfg["training"]['train_params'] and cfg["training"]['train_params']["sam_mask_decoder"]:
            print("TRAINING: mask_decoder")
            train_params += list(model.sam.mask_decoder.parameters())
        if "text_encoder" in cfg["training"]['train_params'] and cfg["training"]['train_params']["text_encoder"] and model.text_encoder is not None:
            print("TRAINING: text_encoder")
            train_params + list(model.text_encoder.parameters())
        
        ## Add temp
        train_params += [model.temperature]

        print(len(train_params))
        #breakpoint()

        cfg['training']['loss']['global_loss'] = False
        #cfg["output"]['wandb'] = True
 

    if args.suppress_wand:
        cfg["output"]['wandb'] = False
    print ("CONFIG:")
    ## Print config
    print(json.dumps(cfg, indent = 4))
    print("args: ")
    print(args)
    print(type(args))
        
    ## Figure out how many steps are requires to reach the total_samples seen
#    total_steps = cfg['training']['samples_seen'] // (cfg['DATA_LOADERS']['LAION']['batch_size'] * args.world_size)
    # update the epoch size to reach samples seen across all nodes
    #cfg['DATA_LOADERS']['LAION']['num_train_samples_per_epoch'] = cfg['DATA_LOADERS']['LAION']['num_train_samples_per_epoch'] // args.world_size

        


    
    train_dataloaders, val_dataloaders = get_data_loaders(cfg, data_loader_args)

    if "COCO" in cfg['DATA_LOADERS']:
        total_steps = len(train_dataloaders[0])
    if "SEGINW" in cfg['DATA_LOADERS']:
        total_steps = cfg['DATA_LOADERS']['SEGINW']["num_samples_per_epoch"]
    else:
        # total_steps = cfg['DATA_LOADERS']['LAION']['num_train_samples_per_epoch'] // (cfg['DATA_LOADERS']['LAION']['batch_size'] * args.world_size)
        total_steps = cfg['DATA_LOADERS']['LAION']['num_train_samples_per_epoch'] // cfg['DATA_LOADERS']['LAION']['batch_size']

    criterion = Criterion(cfg)

    ## As there are now a shitload of different types of training steps depending on what type of data and what prompts and if interactive or not, here is a helper function to retrieve the correct training step:
    train_step_function = get_train_step_function(cfg["training"]["train_step_function"])

    # Get optimizer type defined in cfg
    optimizer = getattr(torch.optim, cfg["training"]["optimizer"]['type'])(
        train_params, 
        **cfg["training"]["optimizer"]['optim_params'])

    # Had to do this here to handle multi gpu loading without OOM
    # # TODO: Move this to frozensam __init__()
    if cfg["model"]["checkpoint"]:
        print("Loading Checkpoint")
        checkpoint_path = cfg["model"]["checkpoint"]
        with open(checkpoint_path, "rb") as f:
        
            state_dict = torch.load(f, map_location=model.device)

        if "optimizer_state_dict" in state_dict:
            #pass
            print("loading optimizer from state dict")
            #optimizer.load_state_dict(state_dict['optimizer_state_dict'], strict=False)

        if "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]

        try:
            if isinstance(model, DDP):
                model.module.load_state_dict(state_dict)
            else:
                model.load_state_dict(state_dict)

        except Exception as e:
            if "sam_vit_" in checkpoint_path:
                ## Handle sam base checkpoints
                if isinstance(model, DDP):
                    model.module.sam.load_state_dict(state_dict, strict=False)
                else:
                    print(model.sam.load_state_dict(state_dict, strict=False))
            
            #print(e)
            print("Skipping loading of unmatched parameters")
            # print(self.load_state_dict(state_dict, strict=False)) 
    print(model.device)

    device = model.device 
    #model= torch.nn.DataParallel(model)
    # device = torch.device(‘cuda:{args.ran’)
    model.to(device)
    print("after loading checkpoint", args.rank, torch.cuda.mem_get_info())

    #breakpoint()
    
    # if torch.cuda.device_count() > 1:
    #     print(f"Using {torch.cuda.device_count()} GPUs")
    #     model = nn.DataParallel(model)  # Wrap your model with DataParallel
    # else:
    #     print("Using single GPU or CPU")

    output_path = os.path.join(cfg["output"]["output_folder"], cfg["output"]["experiment"])

    args.log_dir = os.path.join(output_path, "logs")
    os.makedirs(args.log_dir, exist_ok=True)
    args.model_dir = os.path.join(output_path, "models")
    os.makedirs(args.model_dir, exist_ok=True)


    writer = SummaryWriter(log_dir=args.log_dir)
    global_step = 0
    if args.rank == 0 and cfg["output"]['wandb']:
        wandb.init(project='SSAM_on_LAION')

    ## Retrieve runtime args for training: kinda ghetto
    args.save_steps  = cfg['training']['save_steps']
    args.global_loss = cfg['training']['loss']['global_loss']
    args.log_every = cfg['output']['log_every']
    args.epochs = cfg['training']['epochs']
    args.eval_every = cfg['eval']['eval_every']
    args.eval_steps = cfg['eval']['eval_steps']
    args.dist = cfg["distributed"]['use_deistributed']
    args.wandb = cfg["output"]['wandb']
    #total_steps = cfg["training"]["samples_seen"]

    if "COCO" in cfg['DATA_LOADERS']:
        args.semantic_buffer_size = cfg["DATA_LOADERS"]["COCO"]["num_masks_per_image"] * cfg["DATA_LOADERS"]["COCO"]["batch_size"]
    elif "SEGINW" in cfg['DATA_LOADERS']:
        args.semantic_buffer_size = cfg["DATA_LOADERS"]["SEGINW"]["num_masks_per_image"] * cfg["DATA_LOADERS"]["SEGINW"]["batch_size"] #* (3 if cfg["training"]["train_args"]['multimask'] else 1)
    else:
        args.semantic_buffer_size = cfg["DATA_LOADERS"]["LAION"]["num_masks_per_image"] * cfg["DATA_LOADERS"]["LAION"]["batch_size"]
    print("buffer: ", args.semantic_buffer_size)
    # Hacked solution to pass arguments from config to training step
    train_args = cfg["training"]['train_args'] if "train_args" in cfg['training'] else {}


    # val_results = evaluate(model, 10)
    #breakpoint()
    # if args.rank == 0 and args.eval_start:
    if args.eval_start:
        val_results = evaluate(model, val_dataloaders, train_step_function, criterion, train_args, args, cfg, args.eval_steps)
        print(json.dumps(val_results, indent = 4))
        if args.wandb:
            wandb.log(val_results)

    # Snyc all nodes      
    if args.world_size > 1:
        print(f"[{args.rank}] before barrier")
        torch.distributed.barrier() 
        print(f"[{args.rank}] after barrier")

    #model = torch.compile(model)

    #breakpoint()

    print("train step: ", train_step_function)

    train(
        model, 
        train_dataloaders, 
        val_dataloaders, 
        criterion, 
        optimizer, 
        args.local_rank, 
        args.epochs, 
        writer, 
        args, 
        cfg,
        args.eval_every, 
        args.log_every, 
        global_step=global_step, 
        total_steps=total_steps,
        train_step_function = train_step_function,
        amp =  cfg["training"]['train_args']['amp'],
        train_args = train_args,)


    dist.destroy_process_group()
    if args.rank == 0 and args.wandb:
        pass
        ## TODO: Close wandb, can wait until training does not constantly crash lol

def train(
    model, 
    train_dataloaders, 
    val_dataloaders, 
    criterion, 
    optimizer, 
    device, 
    epochs, 
    writer, 
    args, 
    cfg,
    eval_every=1, 
    log_every=1, 
    global_step=0, 
    total_steps = 100000, 
    train_step_function=get_train_step_function("train_step"),
    amp = True,
    train_args = {}):

    ## Have to add this to avoid a "truncated file" error when reading image through the webdataset loader
    ## TODO: Find more elegant way to avoid this error
    from PIL import Image, ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    scaler = torch.cuda.amp.GradScaler()

    
    print("total_steps ", total_steps)
    print("save steps: ", args.save_steps)
    for epoch in range(epochs):
        if args.rank == 0:
            print(f"Epoch: {epoch+1}/{epochs}")
        running_loss = 0
        running_loss_total = 0
        running_loss_mask = 0
        running_loss_semantic = 0

#        dataloader_wrapper = mixed_dataloader_version_2 ([train_dataloader, sa1b_loader], [0.8, 0.2], total_steps, args.rank)
        dataloader_wrapper = mixed_dataloader (
            train_dataloaders, 
            total_steps
        )

        count = {}
        with tqdm.tqdm(total=total_steps, position=0, leave=True) as progress_bar:
            for idx, batched_inputs in enumerate(dataloader_wrapper):
            #for idx, _ in enumerate(dataloader_wrapper):
                try:  
                    # print(f"rank: {args.rank}: ", idx, f" epoch: {epoch}")
                    # continue
                    if amp:
                        with torch.cuda.amp.autocast():
                            losses = train_step_function (model, batched_inputs, criterion, **train_args, idx = idx)
                    else:
                        losses = train_step_function (model, batched_inputs, criterion, **train_args, idx = idx)
                    # scaler.scale(losses['total_loss']).backward()
                    # scaler.step(optimizer)
                    # scaler.update()

                    optimizer.zero_grad()
                    losses["total_loss"].backward()
                    optimizer.step()

                    # Clamp temperature
                    # with torch.no_grad():
                    #     train_params[-1].clamp_(-1, 1)

      

                    loss = losses["total_loss"]
                    mask_loss = losses["mask_loss"]
                    semantic_loss = losses["semantic_loss"]


                    running_loss += loss.item()
                    running_loss_total += loss.item()
                    running_loss_mask += mask_loss.item() if mask_loss != 0.0 else 0.0
                    running_loss_semantic += semantic_loss.item() if semantic_loss != 0.0 else 0.0

                    # assert False

                    global_step += 1

                    if args.rank == 0:
                        progress_bar.set_description(f"Loss: {(loss):.4f}")
                        progress_bar.update(1)


                    if args.rank == 0 and log_every > 0 and global_step % log_every == 0:
                        writer.add_scalar("train_loss", loss, global_step)
                        writer.add_scalar("train_mask_loss", mask_loss, global_step)
                        writer.add_scalar("train_semantic_loss", semantic_loss, global_step)

                        if args.wandb:
                            wandb.log({
                                "train_loss": running_loss_total/log_every,
                                "train_mask_loss": running_loss_mask/log_every,
                                "train_semantic_loss": running_loss_semantic/log_every,
                            })
                        running_loss_total = 0.0
                        running_loss_mask = 0.0
                        running_loss_semantic = 0.0

                    #if args.rank == 0 and (global_step % args.save_steps == 0 or idx == total_steps - 1):
                    if args.rank == 0 and (global_step % args.save_steps == 0):
                        model_path = os.path.join(args.model_dir, f"model_{global_step}.pt")
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': model.module.state_dict() if isinstance(model, DDP) else model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss': loss,
                            }, model_path)
                        print(f"Saved model: {model_path}")

                    # if (global_step % args.eval_every == 0 or idx == total_steps - 1):
                    if (global_step % args.eval_every == 0):
                        if args.rank == 0 :
                            print("Running evaluation")
                            # val_loss = evaluate(model, val_dataloaders, criterion, text_criterion, device, args, args.eval_steps)
                            # print(f"Val Loss: {(val_loss['total_loss']):.4f}")
                            val_results = evaluate(model, val_dataloaders, train_step_function, criterion, train_args, args, cfg, args.eval_steps)
                            print(json.dumps(val_results, indent = 4))
                            if args.wandb:
                                wandb.log(val_results)
                        if args.world_size > 1:
                            torch.distributed.barrier() 



                except Exception as e: print (str(e))


                
        epoch_loss = running_loss / total_steps
        print(f"Train Epoch Loss: {epoch_loss:.4f}")


    writer.close()

@torch.no_grad()
def evaluate(model, val_dataloaders, train_step_function, criterion, train_args, args,  cfg, eval_steps = 100):
    try:
        summary, scores, avg = seginw.run_seginw_eval(
            model if isinstance(model, FrozenSAM) else model.module,
            **cfg['eval']['seginw_eval_args']
            )
        # breakpoint()

        print(f"#############################################################################")
        print(f"AVERAGE SCORE:::: {avg}")
        print(f"#############################################################################")

        return summary
    
    except Exception as e: 
        print("Evaluation failed")
        print (str(e))
        return {"avg" : 0.0}

    # 
    # dataloader_wrapper = mixed_dataloader (
    #     val_dataloaders, 
    #     eval_steps
    # )

    # running_loss = 0
    # running_loss_total = 0
    # running_loss_mask = 0
    # running_loss_semantic = 0

    # with tqdm.tqdm(total=eval_steps, position=0, leave=True) as progress_bar:
    #     for idx, batched_inputs in enumerate(dataloader_wrapper):
    #     #for idx, _ in enumerate(dataloader_wrapper):
    #         # try:  
    #             #print(f"rank: {args.rank}: ", idx)
    #             # continue
    #             losses = train_step_function (model, batched_inputs, criterion, **train_args, idx = idx)

    #             loss = losses["total_loss"]
    #             mask_loss = losses["mask_loss"]
    #             semantic_loss = losses["semantic_loss"]

    #             if args.rank == 0:
    #                 progress_bar.set_description(f"Loss: {(loss):.4f}")
    #                 progress_bar.update(1)


    #             running_loss += loss.item()
    #             running_loss_total += loss.item()
    #             running_loss_mask += mask_loss.item() if mask_loss != 0.0 else 0.0
    #             running_loss_semantic += semantic_loss.item() if semantic_loss != 0.0 else 0.0

    # return {
    #     "eval" : running_loss / eval_steps,
    #     "eval_loss_total" : running_loss_total / eval_steps,
    #     "eval_loss_mask" : running_loss_mask / eval_steps,
    #     "eval_loss_semantic" : running_loss_semantic / eval_steps,
    # }

if __name__ == '__main__':
    main()
