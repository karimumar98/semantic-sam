





## Code for running evaluation on SEGINW
import os
import sys
import time
import logging
import datetime

# from mpi4py import MPI
import numpy as np

import torch
from detectron2.data import MetadataCatalog
from detectron2.utils.logger import log_every_n_seconds

from itertools import islice

import sys
# TODO: Un-hardcode this
#sys.path.append("os.getenv("XDECODER_DIR")")
sys.path.append("/cluster/project/zhang/umarka/clip_detector/X-Decoder")

## Coco caption has to be installed and added to path
sys.path.append("/utils/coco_caption")
sys.path.append("/coco_caption")


from utils.arguments import load_opt_command
# from utils.distributed import init_distributed, is_main_process, apply_distributed, synchronize
from utils.misc import hook_metadata, hook_switcher, hook_opt
from datasets import build_evaluator, build_eval_dataloader
from xdecoder import build_model
from xdecoder.BaseModel import BaseModel
from xdecoder.utils import get_class_names

logger = logging.getLogger(__name__)
# logging.basicConfig(level = logging.INFO)
# logging.basicConfig(level = logging.WARNING)
logging.basicConfig(level = logging.WARNING)

import matplotlib.pyplot as plt

from torchvision.ops import box_convert, nms, box_iou

from tqdm import tqdm
import cv2

import importlib, segment_anything
importlib.reload(segment_anything)
from eval_with_json.main import evaluate
from segment_anything import predictor, automatic_mask_generator_paper
importlib.reload(predictor)
importlib.reload(automatic_mask_generator_paper)
import torch

from configs.seginw_configs import opt
from segment_anything.eval.metadata import SEGINW_NAMES
from segment_anything.utils import show_anns

import pickle 
import json
import uuid
import shutil

def run_seginw_eval (
    model,
    start = 0,
    end = 100,
    classifier_path = None,
    config = opt,
    bs = 8,
    points_per_side = 10,
    crop_n_layers = 0,
    topk = 20,
    pred_iou_thresh = 0.8,
    stability_score_thresh = 0.8,
    semantic_threshold = 0.9,
    viz_output = True,
    nms_threshold = 0.5,
    max_iter = 10, # Some datasets have thousands of images, making it unpractictal to do eval frequently
    background_scalar = 5.0,
    classifier_reduction = "max",
    crop_n_points_downscale_factor = 1,
    apply_softmax = True,
    area_threshold = 0.8,
    cosine_sim_pred_threshold = 0.2,
    eval_type = "json",
    output_dir = "tmp_json_results",
    scoring_method = "sum",
):

    assert classifier_path != None, "Please provide a directory with classifiers"

    if eval_type == "json":

        run_id = str(uuid.uuid4())
        output_dir = os.path.join(output_dir, run_id)
        os.mkdir(output_dir)
        os.mkdir(os.path.join(output_dir, "jsons"))
        print("saving results to: ", output_dir)

        
        dataset_names = config['DATASETS']['TEST']
        config['SEGINW']['TEST']['BATCH_SIZE_TOTAL'] = bs

        dataloaders = build_eval_dataloader(config)

        # init metadata
        scores = {}
        summary = {}
        dataloaders = dataloaders[start:end+1]
        dataset_names = dataset_names[start:end+1]
        for dataloader, dataset_name in zip(dataloaders, dataset_names):

            try:
                # build evaluator
                evaluator = build_evaluator(config, dataset_name, config['SAVE_DIR'])
                evaluator.reset()
                # Make evaluation only run on single gpu as i don't have enough time to make it distributed
                evaluator._distributed = False

                classifier = pickle.load(
                    open(os.path.join(classifier_path, f"{dataset_name}.pkl"),"rb")
                )

                mask_generator = automatic_mask_generator_paper.SamAutomaticMaskGeneratorPaper(
                    model.sam, 
                    classifier = classifier, 
                    text_decoder = model.text_decoder,
                    points_per_side = points_per_side, 
                    crop_n_layers = crop_n_layers, 
                    background_scalar = background_scalar, 
                    crop_n_points_downscale_factor = crop_n_points_downscale_factor,
                    pred_iou_thresh = pred_iou_thresh,
                    stability_score_thresh = stability_score_thresh,
                    output_mode = "coco_rle"
                )
                # Build predictor from model
                pred = predictor.SamPredictor(model.sam, text_decoder = model.text_decoder)
                mask_generator.predictor = pred

                print(f"Starting Eval on: {dataset_name}")
                with torch.no_grad():
                    # setup model
                    names = get_class_names(dataset_name)
                    eval_type = "seginw"

                    #breakpoint()

                    
                    # setup timer
                    total = len(dataloader)
                    num_warmup = min(5, total - 1)
                    start_time = time.perf_counter()
                    total_data_time = 0
                    total_compute_time = 0
                    total_eval_time = 0
                    start_data_time = time.perf_counter()
                    eval_ids = []

                    l = min(dataloader.__len__(), max_iter)

                    outputs = []

                    with tqdm(total = l) as pbar:

                        #breakpoint()

                        for idx, batch in enumerate(islice(dataloader, 0, l)):
                            batch_ids = [x['image_id'] for x in batch]
                            eval_ids += batch_ids
                            pbar.update(1)
                            # print(len(batch))
                            
                            images = [ cv2.cvtColor(cv2.imread(x['file_name']), cv2.COLOR_BGR2RGB) for x in batch                        ]
                            masks = [mask_generator.generate(image, filter_like_paper = False, apply_softmax = apply_softmax, keep_cuda = False, classifier_reduction = classifier_reduction) for image in images]
                            #breakpoint()
                            # outputs  = postproces_masks(masks)
                            outputs += automatic_mask_generator_paper.postprocess_masks_seginw(masks, 
                                dataset_name, 
                                topk = topk,
                                pred_iou_thresh = pred_iou_thresh,
                                stability_score_thresh = stability_score_thresh,
                                semantic_threshold = semantic_threshold,
                                nms_threshold = nms_threshold,
                                area_threshold = area_threshold,
                                cosine_sim_pred_threshold = cosine_sim_pred_threshold,
                                image_shapes = [x.shape[:2] for x in images],
                                image_ids = batch_ids,
                                scoring_method = scoring_method,
                            )

                            
                        file_name = f"{dataset_name}.json"
                        file_name = os.path.join(os.path.join(output_dir, "jsons"), file_name)
                        json.dump(outputs, open(file_name, "w"))
                        print(f"saved results to {file_name}")
                        
                    
                        
            except Exception as e: 
                print(f"failed: {dataset_name}")
                print(f"Exception: {e}")

        # Now zip all results and call X-Decoder eval code

        sub_path = shutil.make_archive(os.path.join(output_dir, "submission"), 'zip', os.path.join(output_dir, "jsons"))
        print(f"Created submission file: {sub_path}")

        #breakpoint()
        output = evaluate(sub_path)
        
        
        return output

    else:
        # build dataloade
        # evaluation dataset
        dataset_names = config['DATASETS']['TEST']
        config['SEGINW']['TEST']['BATCH_SIZE_TOTAL'] = bs

        dataloaders = build_eval_dataloader(config)

        # init metadata
        scores = {}
        summary = {}
        dataloaders = dataloaders[start:end+1]
        dataset_names = dataset_names[start:end+1]
        for dataloader, dataset_name in zip(dataloaders, dataset_names):

            try:
                # build evaluator
                evaluator = build_evaluator(config, dataset_name, config['SAVE_DIR'])
                evaluator.reset()
                # Make evaluation only run on single gpu as i don't have enough time to make it distributed
                evaluator._distributed = False

                classifier = pickle.load(
                    open(os.path.join(classifier_path, f"{dataset_name}.pkl"),"rb")
                )

                # breakpoint()


                mask_generator = automatic_mask_generator_paper.SamAutomaticMaskGeneratorPaper(
                    model.sam, classifier = classifier, 
                    text_decoder = model.text_decoder,
                    points_per_side = points_per_side, 
                    crop_n_layers = crop_n_layers, 
                    background_scalar = background_scalar, 
                    crop_n_points_downscale_factor = crop_n_points_downscale_factor,
                    pred_iou_thresh = pred_iou_thresh,
                    stability_score_thresh = stability_score_thresh,
                )
                # mask_generator = automatic_mask_generator_paper.SamAutomaticMaskGeneratorPaper(model.sam, classifier = classifier, text_decoder = model.text_decoder, points_per_batch = 512,points_per_side = 10,crop_n_layers = 0,background_scalar = 3)


                print(f"Starting Eval on: {dataset_name}")
                with torch.no_grad():
                    # setup model
                    names = get_class_names(dataset_name)
                    eval_type = "seginw"

                    #breakpoint()

                    # Build predictor from model
                    pred = predictor.SamPredictor(model.sam, text_decoder = model.text_decoder)

                    mask_generator.predictor = pred


                    # setup timer
                    total = len(dataloader)
                    num_warmup = min(5, total - 1)
                    start_time = time.perf_counter()
                    total_data_time = 0
                    total_compute_time = 0
                    total_eval_time = 0
                    start_data_time = time.perf_counter()
                    eval_ids = []

                    l = min(dataloader.__len__(), max_iter)

                    with tqdm(total = l) as pbar:

                        #breakpoint()

                        for idx, batch in enumerate(islice(dataloader, 0, l)):
                            eval_ids += [x['image_id'] for x in batch]
                            pbar.update(1)
                            # print(len(batch))
                            total_data_time += time.perf_counter() - start_data_time
                            if idx == num_warmup:
                                start_time = time.perf_counter()
                                total_data_time = 0
                                total_compute_time = 0
                                total_eval_time = 0
                            start_compute_time = time.perf_counter()



                            images = [ cv2.cvtColor(cv2.imread(x['file_name']), cv2.COLOR_BGR2RGB) for x in batch                        ]
                            masks = [mask_generator.generate(image, filter_like_paper = False, apply_softmax = apply_softmax, keep_cuda = False, classifier_reduction = classifier_reduction) for image in images]
                            breakpoint()
                            # outputs  = postproces_masks(masks)
                            outputs = automatic_mask_generator_paper.postprocess_masks_seginw(masks, 
                                dataset_name, 
                                topk = topk,
                                pred_iou_thresh = pred_iou_thresh,
                                stability_score_thresh = stability_score_thresh,
                                semantic_threshold = semantic_threshold,
                                nms_threshold = nms_threshold,
                                area_threshold = area_threshold,
                                cosine_sim_pred_threshold = cosine_sim_pred_threshold,
                                image_shapes = [x.shape[:2] for x in images]
                            )

                            breakpoint()


                            total_compute_time += time.perf_counter() - start_compute_time
                            start_eval_time = time.perf_counter()
                            
                            evaluator.process(batch, outputs)
                            

                            total_eval_time += time.perf_counter() - start_eval_time

                            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
                            data_seconds_per_iter = total_data_time / iters_after_start
                            compute_seconds_per_iter = total_compute_time / iters_after_start
                            eval_seconds_per_iter = total_eval_time / iters_after_start
                            total_seconds_per_iter = (time.perf_counter() - start_time) / iters_after_start

                            start_data_time = time.perf_counter()
                            
                # evaluate
                # print("before eval")
                results = evaluator.evaluate(img_ids = eval_ids)
                # print("after eval")

                # summary
                summary_keys = []
                if config['MODEL']['DECODER']['TEST']['PANOPTIC_ON']:
                    result_key = 'panoptic_seg'
                    summary_keys += ['PQ', 'SQ', 'RQ']
                if config['MODEL']['DECODER']['TEST']['INSTANCE_ON']:
                    result_key = 'segm'
                    summary_keys += ['AP']
                if config['MODEL']['DECODER']['TEST']['SEMANTIC_ON']:
                    result_key = 'sem_seg'
                    summary_keys += ['mIoU']
                for eval_type in results.keys():
                    for key in results[eval_type]:
                        scores["{}/{}/{}".format(dataset_name, eval_type, key)] = results[eval_type][key]
                        if key in summary_keys:
                            summary["{}/{}/{}".format(dataset_name, eval_type, key)] = results[eval_type][key]
            except Exception as e: 
                print(f"failed: {dataset_name}")
                print(f"Exception: {e}")


        logger = logging.getLogger(__name__)
        logging.basicConfig(level = logging.INFO)

        logger.info(summary)

        # We only care about segmentation
        just_segmentation_summary = {key: summary[key] for key in summary if "segm" in key} 
        # avg = sum(just_segmentation_summary.values()) / (len (just_segmentation_summary) + 0.00000001)
        avg = sum(just_segmentation_summary.values()) / 25.0

        just_segmentation_summary.update({"avg" : avg})

    return just_segmentation_summary, scores, avg


