import sys
from torch.nn.functional import threshold, normalize, pad, interpolate, binary_cross_entropy_with_logits, mse_loss, softmax, cross_entropy
from torchvision.ops import box_convert
import pickle
import webdataset
import numpy as np
import torchvision.transforms as transforms

import torch, torchvision
import random
import os
from PIL import Image
import cv2
import json
import torch.distributed as dist
from pycocotools import mask as mask_utils
from torch.utils.data import Dataset
import os

import torch.utils.data as data
from pycocotools.coco import COCO
import json
import io
import random


from .transforms import ResizeLongestSide
def pad_to_size(in_tensor, target_size = 1024):
    h, w = in_tensor.shape[-2:]
    padh = target_size - h
    padw = target_size - w
    padded = pad(in_tensor, (0, padw, 0, padh))
    return padded



def build_data_loader (cfg, data_loader_args = {}):
    # builds a dataloader for training with cfg
    data_loader_builder_map = {
        "get_multi_node_laion_dataloader" : get_multi_node_laion_dataloader,
        "get_SA_1B_dataloader" : get_SA_1B_dataloader,
        "get_COCO_dataloader" : get_COCO_dataloader,
        "get_SEGINW_dataloader" : get_SEGINW_dataloader
    }
    ##Extract config args to pass to dataloader
    args = {i:cfg[i] for i in cfg if i!='NAME'}
    return data_loader_builder_map[cfg['NAME']](**args, **data_loader_args)

def get_data_loaders (cfg, data_loader_args = {}):
    ##Build all our dataloaders
    train_dataloaders, val_dataloaders = [], []
    for name in cfg["DATA_LOADERS"]:
        train_dataloader, val_dataloader = build_data_loader(cfg["DATA_LOADERS"][name], data_loader_args)
        train_dataloaders.append(train_dataloader)
        val_dataloaders.append(val_dataloader)
    return train_dataloaders, val_dataloaders

def get_shards (url):
    ## Helper function to handle data input soruces depending on it's format. Shards can be passed as:
    # - List of paths or URLS 
    # - List of root directories, each containing many shards
    # - that weird brace expand format eg. "{00000..00100}.tar"

    if "{" in url and ".." in url and "}" in url:
        # Url is in brace format
        ## Expand shards:
        sl = webdataset.SimpleShardList(url)
        shards = [x['url'] for x in sl]
    elif isinstance(url, str) and not url.endswith(".tar") and os.path.isdir(url):
        # Shards are all paths in this directory
        files = [x for x in os.listdir(url) if x.endswith(".tar")]
        shards = [os.path.join(url, f) for f in files]
    elif isinstance(url, list) and os.path.isdir(url[0]):
        # we received a list of directories, we will use all shards in these directories
        shards = []
        
        for base in url:
            shards += [os.path.join(base, x) for x in os.listdir(base) if x.endswith(".tar")]
        # print(shards)
    else:
        # This handles the input being either a list of urls or paths
        shards = url

    return shards

class mixed_dataloader:
    def __init__(self, dataloaders, steps):
        ## Hacky solution to handle mixed datasets across multiple nodes, this version returns a list of batched inputs, in this dataloader version, each iteration has a predefined mix of data sources and requires no coordination between players
        self.dataloaders = [iter(x) for x in dataloaders]
        self.steps = steps
        self.count = 0

    def __iter__(self):
        return self

    def __len__(self):
        return self.steps

    def __next__ (self):

        if self.count > self.steps:
            raise StopIteration
        self.count += 1
        return [next(x) for x in self.dataloaders]
    


def collate_fn_dictionary(batch):
    ## 'Custom' collate to return a list of dicts instead of a single dictionary, DETIC style
    return batch

def get_SEGINW_dataloader (
    root_folder,
    return_dict = False,
    dataset_name = None,
    num_masks_per_image = 8,
    train_split = 1.0,
    batch_size = 4,
    num_workers = 0,
    precomputed_clip_emb_path = None,
    **kwargs):

    if train_split == 1.0:
        coco_dataset = SEGINWDataset(
            root = root_folder,
            return_dict = return_dict,
            dataset_name = dataset_name,
            precomputed_clip_emb_path = "precomputed_seginw_clip_emb.pkl",
            num_masks_per_image = num_masks_per_image
        )
        train_loader = torch.utils.data.DataLoader (
            coco_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn = collate_fn_dictionary,
        )
        return train_loader, None

    else:
        coco_dataset = SEGINWDataset(
            root = root_folder,
            return_dict = return_dict,
            dataset_name = dataset_name,
            precomputed_clip_emb_path = "precomputed_seginw_clip_emb.pkl"
        )

        coco_train_size = int(train_split * len(coco_dataset))
        coco_test_size = len(coco_dataset) - coco_train_size
        train, val = torch.utils.data.random_split(coco_dataset, [coco_train_size, coco_test_size])

        train_loader = torch.utils.data.DataLoader (
            train,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle = True,
            collate_fn = collate_fn_dictionary,
        )
        val_loader = torch.utils.data.DataLoader (
            val,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle = False,
            collate_fn = collate_fn_dictionary
        )

        return train_loader, val_loader

def get_COCO_dataloader (
    root_folder,
    dataType,
    annotations,
    return_dict = False,
    num_masks_per_image = 1,
    train_split=0.8,
    batch_size = 4,
    num_workers = 0,
    precomputed_clip_emb_path = None,
    **kwargs):
    
    image_folder = os.path.join(
        root_folder,
        dataType
    )

    json_path = os.path.join(
        root_folder,
        annotations
    )

    coco_dataset = CocoDataset(
        root = image_folder,
        json = json_path,
        return_dict = return_dict,
        precomputed_clip_emb_path = precomputed_clip_emb_path
    )

    coco_train_size = int(train_split * len(coco_dataset))
    coco_test_size = len(coco_dataset) - coco_train_size
    train, val = torch.utils.data.random_split(coco_dataset, [coco_train_size, coco_test_size])

    train_loader = torch.utils.data.DataLoader (
        train,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle = True,
        collate_fn = collate_fn_dictionary,
    )
    val_loader = torch.utils.data.DataLoader (
        val,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle = False,
        collate_fn = collate_fn_dictionary
    )

    return train_loader, val_loader

def get_SA_1B_dataloader (
    root_folder = None,
    url = None,
    return_dict = False,
    num_masks_per_image = 1,
    train_split=0.8,
    batch_size = 4,
    num_workers = 1,
    **kwargs):

    if root_folder is not None:
        ## SA1B is stored locally, meaning there is a single directory containing all images and all annotations
        sa1b_dataset = SA1B_Dataset(
            root_folder,
            return_dict = return_dict,
            num_masks_per_image = num_masks_per_image
            )
        sa1b_train_size = int(train_split * len(sa1b_dataset))
        sa1b_test_size = len(sa1b_dataset) - sa1b_train_size
        sa1b_train, sa1b_val = torch.utils.data.random_split(sa1b_dataset, [sa1b_train_size, sa1b_test_size])

        sa1b_train_loader = torch.utils.data.DataLoader (
            sa1b_train,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle = True,
            collate_fn = collate_fn_dictionary,
        )
        sa1b_val_loader = torch.utils.data.DataLoader (
            sa1b_val,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle = False,
            collate_fn = collate_fn_dictionary
        )

        return sa1b_train_loader, sa1b_val_loader

    elif url is not None:
        # Handles the case that data is stored in WebDataset style shards, eg:
        #url = ['pipe:s3cmd -q get s3://laion/SA-1B/sa_000020.tar -', 'pipe:s3cmd -q get s3://laion/SA-1B/sa_000022.tar -', 'pipe:s3cmd -q get s3://laion/SA-1B/sa_000023.tar -', 'pipe:s3cmd -q get s3://laion/SA-1B/sa_000024.tar -', 'pipe:s3cmd -q get s3://laion/SA-1B/sa_000025.tar -', 'pipe:s3cmd -q get s3://laion/SA-1B/sa_000029.tar -', 'pipe:s3cmd -q get s3://laion/SA-1B/sa_000130.tar -', 'pipe:s3cmd -q get s3://laion/SA-1B/sa_000132.tar -', 'pipe:s3cmd -q get s3://laion/SA-1B/sa_000133.tar -', 'pipe:s3cmd -q get s3://laion/SA-1B/sa_000134.tar -', 'pipe:s3cmd -q get s3://laion/SA-1B/sa_000135.tar -', 'pipe:s3cmd -q get s3://laion/SA-1B/sa_000137.tar -', 'pipe:s3cmd -q get s3://laion/SA-1B/sa_000138.tar -', 'pipe:s3cmd -q get s3://laion/SA-1B/sa_000139.tar -', 'pipe:s3cmd -q get s3://laion/SA-1B/sa_000250.tar -', 'pipe:s3cmd -q get s3://laion/SA-1B/sa_000251.tar -', 'pipe:s3cmd -q get s3://laion/SA-1B/sa_000252.tar -', 'pipe:s3cmd -q get s3://laion/SA-1B/sa_000254.tar -', 'pipe:s3cmd -q get s3://laion/SA-1B/sa_000256.tar -', 'pipe:s3cmd -q get s3://laion/SA-1B/sa_000257.tar -', 'pipe:s3cmd -q get s3://laion/SA-1B/sa_000258.tar -', 'pipe:s3cmd -q get s3://laion/SA-1B/sa_000259.tar -', 'pipe:s3cmd -q get s3://laion/SA-1B/sa_000340.tar -', 'pipe:s3cmd -q get s3://laion/SA-1B/sa_000341.tar -', 'pipe:s3cmd -q get s3://laion/SA-1B/sa_000342.tar -', 'pipe:s3cmd -q get s3://laion/SA-1B/sa_000343.tar -', 'pipe:s3cmd -q get s3://laion/SA-1B/sa_000346.tar -', 'pipe:s3cmd -q get s3://laion/SA-1B/sa_000347.tar -', 'pipe:s3cmd -q get s3://laion/SA-1B/sa_000348.tar -', 'pipe:s3cmd -q get s3://laion/SA-1B/sa_000349.tar -', 'pipe:s3cmd -q get s3://laion/SA-1B/sa_000494.tar -', 'pipe:s3cmd -q get s3://laion/SA-1B/sa_000587.tar -', 'pipe:s3cmd -q get s3://laion/SA-1B/sa_000719.tar -']
        
        if not isinstance(url, list):
            url = get_shards (url)


        train_size = round(train_split * len(url))
        test_size = len(url) - train_size
        train_shards, val_shards = torch.utils.data.random_split(url, [train_size, test_size])

        sa1b_dataset_train = SA1B_Dataset_WDS(
            nodesplitter=webdataset.split_by_node,
            handler=webdataset.warn_and_continue,
            pipe = train_shards,
            return_dict = return_dict,
            num_masks_per_image = num_masks_per_image
        ).decode("rgb").shuffle(1000)

        print("warn_and_continue")
        sa1b_dataset_val = SA1B_Dataset_WDS(
            nodesplitter=webdataset.split_by_node,
            handler=webdataset.warn_and_continue,
            pipe = val_shards,
            return_dict = return_dict,
            num_masks_per_image = num_masks_per_image
        ).decode("rgb")

        sa1b_train_loader = torch.utils.data.DataLoader (
            sa1b_dataset_train,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn = collate_fn_dictionary,
            pin_memory = True,

        )
        sa1b_val_loader = torch.utils.data.DataLoader (
            sa1b_dataset_val,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn = collate_fn_dictionary,
            pin_memory = True,
        )

        return sa1b_train_loader, sa1b_val_loader

    else:
        assert False, "atleast one source of SA1B data must be provided"



#Only for debugging, to see which shards are assigned to which nodes
def custom_split (src):
    #print(f"[Custom Split function] src: \t {src}")
    for s in src:
        #print(f"[Custom Split function] \t {s}")
        yield s


def split_by_node(src, group=None):
    rank, world_size, worker, num_workers = utils.pytorch_worker_info(group=group)
    if world_size > 1:
        for s in islice(src, rank, None, world_size):
            yield s
    else:
        for s in src:
            yield s


def get_multi_node_laion_dataloader (
    url,
    num_train_samples_per_epoch,
    num_val_samples_per_epoch,
    num_workers = 1,
    batch_size = 4,
    num_masks_per_image = 1,
    image_size = 1024,
    mean = torch.Tensor([[[123.675]], [[116.28 ]],  [[103.53 ]]]),
    std = torch.tensor([[[58.3950]], [[57.1200]], [[57.3750]]]),
    box_input = "cxcywh",
    return_dict = False,
    train_split = 0.8,
    return_url = False,
    return_json = False,
    shuffle = 1000,
    random_mix = False,
    **kwargs): ##Ingore unwanted args

    ## Get laion train + val loaders
    shards = get_shards (url)

    #breakpoint()

    print(f"each node will see {num_train_samples_per_epoch} per epoch")

    train_size = int(train_split * len(shards))
    test_size = len(shards) - train_size
    train_shards, val_shards = torch.utils.data.random_split(shards, [train_size, test_size])

    if not random_mix:
        # Standard case, each shard contains a random mix of data, no shuffling above shard level
        train_dataset = CustomWebDataset(
            [url for url in train_shards],
            nodesplitter=webdataset.split_by_node,
            input_size = image_size,
            pixel_mean = mean,
            pixel_std = std,
            box_input_format = box_input,
            return_dict = return_dict,
            num_masks_per_batch = num_masks_per_image,
            handler=webdataset.warn_and_continue,
            return_json = return_json,
        # ).decode("rgb").repeat(2).with_epoch(4) #.with_epoch(num_train_samples_per_epoch).shuffle(1000)
        ).decode("rgb").with_epoch(num_train_samples_per_epoch) #.shuffle(1000)

        collate_fn = collate_fn_dictionary if return_dict else None
        train_loader = torch.utils.data.DataLoader (
        # train_loader = webdataset.WebLoader (
             train_dataset,
             batch_size=batch_size,
             num_workers=num_workers,
             pin_memory = True,
             collate_fn = collate_fn)
        #train_loader = webdataset.WebLoader (
            #train_dataset,
            #batch_size=batch_size,
            #num_workers=num_workers,
            #pin_memory = True,
            #collate_fn = collate_fn) #.unbatched().shuffle(shuffle).batched(batch_size)
        print("train_shards: ", len([x for x in train_shards]))
        print("val_shards:  ", len([x for x in val_shards]))
        if [x for x in val_shards] != []:
            val_dataset = CustomWebDataset(
                [url for url in val_shards],
                nodesplitter=webdataset.split_by_node,
                input_size = image_size,
                pixel_mean = mean,
                pixel_std = std,
                box_input_format = box_input,
                return_dict = return_dict,
                num_masks_per_batch = num_masks_per_image,
                return_json = return_json,
                handler=webdataset.warn_and_continue,
            # ).decode("rgb").repeat(2).with_epoch(num_train_samples_per_epoch) #.with_epoch(num_val_samples_per_epoch)
            ).decode("rgb").with_epoch(num_val_samples_per_epoch)

            val_loader = torch.utils.data.DataLoader (
            # val_loader = webdataset.WebLoader (
                val_dataset,
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory = True,
                collate_fn = collate_fn)
        else:
            val_loader = None
        return train_loader, val_loader
    else:
        # Non-standard case, each shardlist contains data for a single category type, data from each source is shuffled at train time, not recommended to use this, as it is extremely memory hungry, shuffle data prior to sharding
        train_datasets = [CustomWebDataset(
            url,
            nodesplitter=custom_split,
            input_size = image_size,
            pixel_mean = mean,
            pixel_std = std,
            box_input_format = box_input,
            return_dict = return_dict,
            num_masks_per_batch = num_masks_per_image,
            handler=webdataset.warn_and_continue,
            return_json = return_json).decode("rgb").with_epoch(num_train_samples_per_epoch).shuffle(shuffle) for url in train_shards]

        val_datasets = [CustomWebDataset(
            url,
            nodesplitter=custom_split,
            input_size = image_size,
            pixel_mean = mean,
            pixel_std = std,
            box_input_format = box_input,
            return_dict = return_dict,
            num_masks_per_batch = num_masks_per_image,
            return_json = return_json,
            handler=webdataset.warn_and_continue).decode("rgb").with_epoch(num_val_samples_per_epoch) for url in val_shards]

        # return train_datasets, val_datasets
        train_dataset =  webdataset.RandomMix(train_datasets, longest=True)
        val_dataset =  webdataset.RandomMix(val_datasets, longest=True)


        collate_fn = collate_fn_dictionary if return_dict else None

        train_loader = torch.utils.data.DataLoader (
            train_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory = True,
            collate_fn = collate_fn)


        val_loader = torch.utils.data.DataLoader (
            val_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory = True,
            collate_fn = collate_fn)
        return train_loader, val_loader





## Custom subclass of webdataset to load and transform modified Laion data
class CustomWebDataset(webdataset.WebDataset):

    def __init__(self, pipe,
                 input_size = 1024,
                 pixel_mean = torch.Tensor([[[123.675]], [[116.28 ]],  [[103.53 ]]]),
                 pixel_std = torch.tensor([[[58.3950]], [[57.1200]], [[57.3750]]]),
                 transform=None,
                 random_instance=True,
                 return_raw=False,
                 random_debug = False,
                 return_url = False,
                 box_input_format = "cxcywh",
                 return_all = False,
                 return_dict = False,
                 num_masks_per_batch = 1,
                 return_json = False,
                 return_llm_queries = False,
                 **kwargs):
        super().__init__(pipe, **kwargs)

        self.input_size = input_size
        self.pixel_mean = pixel_mean
        self.pixel_std = pixel_std
        self.resize = ResizeLongestSide(input_size)
        self.transform = transform
        self.random_instance = random_instance
        self.return_raw = return_raw
        self.return_url = return_url
        self.box_input_format = box_input_format
        self.return_all = return_all
        self.return_dict = return_dict
        self.num_masks_per_batch = num_masks_per_batch
        self.return_json = return_json

        # print(f"num_masks_per_batch: {num_masks_per_batch}")

    def __iter__(self):
        for data in  super().__iter__():

            if 'jpg' not in data or 'json' not in data or 'txt' not in data: 
                ## TODO:: Why do i need to do this -> find out why some data is saved bad, only happens very rarely, could be a data streaming issue
                print(f"Data piece not correctly formed: {data['__key__']}, {data['__url__']}")
                continue

            image = (data["jpg"]*255).astype(np.uint8)
            json = data["json"]
            text = data["txt"]
            if "masks.npy" in data:
                masks = data["masks.npy"]
                masks = np.unpackbits(masks, count=json['mask_size']).reshape(json['mask_shape']).view(bool)
            else:
                masks = None
            features = torch.Tensor(data["clip_emb_phrases.npy"]) if "clip_emb_phrases.npy" in data else torch.Tensor(data["text_features.npy"])
            caption_feature  = torch.Tensor(data["clip_caption.npy"]) if "clip_caption.npy" in data else None
            bboxes = torch.Tensor(json["boxes"]) if "boxes" in json else torch.Tensor(json["bboxes"])
            caption = json["phrases"]
            logits = json['logits']

            # If sample has no bounding boxes, skip it
            if bboxes.shape[0] == 0:
                continue

            image_size = image.shape[:2]
            image_size = json['original_width'], json['original_width']
            num_masks = min(
                self.num_masks_per_batch,
                features.shape[0]
            )

            if self.return_all:
                ## Return all masks per batch
                mask_indices = [l for l in range(num_masks)]
            elif self.random_instance:
                ## Should we choose a random mask or always the first mask?
                ## sample a random subset of
                mask_indices = random.sample([l for l in range(features.shape[0])], k=num_masks)
            else:
                ## Non random, pick first num_masks_per_batch masks
                mask_indices = [l for l in range(num_masks)]

            ## masks is now a num_masks_per_batch x 1024 x 1024 tensor
            if masks is not None:
                masks = masks[mask_indices]
            bboxes = bboxes[mask_indices]


            image = image_raw = self.resize.apply_image(image)

            ## Fix for boxes being bad
            image_size = image.shape[:2]
            image_scaler = torch.tensor([image_size[1],image_size[0],image_size[1],image_size[0]]) / torch.tensor([1024.0] * 4)
            bboxes = bboxes * image_scaler
            image = torch.as_tensor(image, dtype=torch.float32).permute(2, 0, 1).contiguous()
            image = transforms.Normalize(self.pixel_mean, self.pixel_std)(image)

            # Pad image to 1024x1024
            h, w = image.shape[-2:]
            padh = 1024 - h
            padw = 1024 - w
            image = pad(image, (0, padw, 0, padh))

            if self.box_input_format == "cxcywh":
                #TODO: This is probably wrong, but not used so fix in future
                # If boxes are stored in cxcywh, then we have to convert them first
                bboxes = bboxes * torch.Tensor([w, h, w, h])
                bboxes = box_convert(boxes=bboxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()

            bboxes = torch.as_tensor(bboxes, dtype=torch.float32)

            ## Hack to use built in resize tooling of segment anything
            if masks is not None:
                masks = np.array(masks).astype(np.uint8)
                masks = [self.resize.apply_image(mask) for mask in masks]
                masks = [torch.as_tensor(mask, dtype=torch.uint8) for mask in masks]
                masks = [pad(mask, (0, padw, 0, padh)) for mask in masks]
                masks = [mask.bool() for mask in masks]
                masks = torch.stack(masks)

            captions = [caption[mask_index] for mask_index in mask_indices]
            logits = [logits[mask_index] for mask_index in mask_indices]
            features = features[mask_indices]


            if self.return_json:
                if self.return_dict:
                    yield {
                        "image" : image,
                        "boxes" : bboxes,
                        "masks" : masks,
                        "captions" : captions,
                        "image_size" : image_size,
                        "semantic_targets": features,
                        "image_level_features": caption_feature,
                        "ground_truth_type" : "semantic",
                        "json" : json,
                        "logits" : logits
                    }
                else:
                    yield image, bboxes, masks, captions, image_size, features, json, #data['__url__']
            elif self.return_raw:
                raise NotImplementedError()
                yield image, bboxes, masks, captions, image_size, features, img_raw
            else:
                if self.return_dict:
                    yield {
                        "image" : image,
                        "boxes" : bboxes,
                        "masks" : masks,
                        "captions" : captions,
                        "image_size" : image_size,
                        "semantic_targets": features,
                        "image_level_features": caption_feature,
                        "ground_truth_type" : "semantic",
                        "logits" : logits
                    }
                else:
                    # yield image, bboxes, masks, captions, image_size, features
                    yield image, bboxes, masks, captions, image_size, features, caption_feature, "semantic"


class SA1B_Dataset_WDS (webdataset.WebDataset):
    ## LOADS SA1B data if it stored in webdataset format
    def __init__ (
        self,
        pipe,
        transform = None,
        target_transform = None,
        random_instance=True,
        input_size = 1024,
        pixel_mean = torch.Tensor([[[123.675]], [[116.28 ]],  [[103.53 ]]]),
        pixel_std = torch.tensor([[[58.3950]], [[57.1200]], [[57.3750]]]),
        return_all_masks = False,
        return_dict = False,
        paths = [],
        num_masks_per_image = 1,
        **kwargs):

        super().__init__(pipe, **kwargs)

        self.pipe = pipe
        self.transform = transform
        self.target_transform = target_transform
        self.random_instance = random_instance
        self.return_all_masks = return_all_masks
        self.return_dict = return_dict

        self.input_size = input_size
        self.pixel_mean = pixel_mean
        self.pixel_std = pixel_std
        self.num_masks_per_image = num_masks_per_image

        self.resize = ResizeLongestSide(input_size)


    def __iter__(self):
        for data in  super().__iter__():        

            if 'jpg' not in data or 'json' not in data:
                # Should not happen if Meta SA-1B is not weird
                print("Data malformed !!!")
                print(data)
                continue
            
            image = (data['jpg']*255).astype(np.uint8)
            anns_ = data["json"]
            anns = anns_["annotations"]

            num_masks = min(
                self.num_masks_per_image,
                len(anns)
            )


            if self.random_instance:
                ## Should we choose a random mask or always the first mask?
                ## sample a random subset of
                mask_indices = random.sample([l for l in range(len(anns))], k=num_masks)
            else:
                ## Non random, pick first num_masks_per_batch masks
                mask_indices = [l for l in range(num_masks)]

            bboxes = [anns[i]['bbox'] for i in mask_indices]
            masks = [mask_utils.decode(anns[i]['segmentation']) for i in mask_indices]


            image = self.resize.apply_image(image)
            image = torch.as_tensor(image, dtype=torch.float32).permute(2, 0, 1).contiguous()
            image = transforms.Normalize(self.pixel_mean, self.pixel_mean)(image)

            h, w = image.shape[-2:]
            padh = 1024 - h
            padw = 1024 - w
            image = pad(image, (0, padw, 0, padh))




            ## We need to resize the box to the 1024 x 1024 the model expects
            image_size = (anns_["image"]["width"], anns_["image"]["height"])
            w, h = image_size[0], image_size[1]
            k = max(w,h)


            bboxes = torch.as_tensor(bboxes, dtype=torch.float32)
            bboxes = box_convert(boxes=bboxes, in_fmt="xywh", out_fmt="xyxy")

            bboxes /= k #torch.Tensor([k, k, k, k])
            bboxes *= 1024

            ## Hack to use built in resize tooling
            masks = np.array(masks).astype(np.uint8)#.transpose(1,2,0)
            masks = [self.resize.apply_image(mask) for mask in masks]
            masks = [torch.as_tensor(mask, dtype=torch.uint8) for mask in masks]
            masks = torch.stack([pad(mask, (0, padw, 0, padh)) for mask in masks])
            masks = masks.bool()

            if self.return_dict:
                yield {
                    "image" : image,
                    "boxes" : bboxes,
                    "masks" : masks,
                    "ground_truth_type" : "mask"
                }
            else:
                yield image, bboxes, masks, image_size

    def __len__(self):
        return len(self.imgs)


class SA1B_Dataset(Dataset):
    ## LOADS SA1B data if it stored in the format generated by https://github.com/KKallidromitis/SA-1B-Downloader

    def __init__ (
        self,
        root_folder,
        transform = None,
        target_transform = None,
        random_instance=True,
        input_size = 1024,
        pixel_mean = torch.Tensor([[[123.675]], [[116.28 ]],  [[103.53 ]]]),
        pixel_std = torch.tensor([[[58.3950]], [[57.1200]], [[57.3750]]]),
        return_all_masks = False,
        return_dict = False,
        paths = [],
        num_masks_per_image = 1,
        **kwargs):

        super().__init__(**kwargs)

        self.root_folder = root_folder
        self.transform = transform
        self.target_transform = target_transform
        self.random_instance = random_instance
        self.return_all_masks = return_all_masks
        self.return_dict = return_dict

        self.input_size = input_size
        self.pixel_mean = pixel_mean
        self.pixel_std = pixel_std
        self.num_masks_per_image = num_masks_per_image

        self.resize = ResizeLongestSide(input_size)



        self.imgs = [os.path.join(root_folder + "images", x) for x in os.listdir(os.path.join(root_folder, "images"))]
        self.image_map = {a: b for a, b in zip(range(len(self.imgs)), self.imgs)}

    def __getitem__(self, index):
        
        path = self.image_map[index] # discard automatic subfolder labels
        anns_ = json.load(open(path.replace("jpg", "json").replace("/images/", "/annotations/")))
        anns = anns_["annotations"]

        num_masks = min(
            self.num_masks_per_image,
            len(anns)
        )


        if self.random_instance:
            ## Should we choose a random mask or always the first mask?
            ## sample a random subset of
            mask_indices = random.sample([l for l in range(len(anns))], k=num_masks)
        else:
            ## Non random, pick first num_masks_per_batch masks
            mask_indices = [l for l in range(num_masks)]

        bboxes = [anns[i]['bbox'] for i in mask_indices]
        masks = [mask_utils.decode(anns[i]['segmentation']) for i in mask_indices]


        image = cv2.imread(self.image_map[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = self.resize.apply_image(image.astype(np.uint8))
        image = torch.as_tensor(image, dtype=torch.float32).permute(2, 0, 1).contiguous()
        image = transforms.Normalize(self.pixel_mean, self.pixel_mean)(image)

        h, w = image.shape[-2:]
        padh = 1024 - h
        padw = 1024 - w
        image = pad(image, (0, padw, 0, padh))




        ## We need to resize the box to the 1024 x 1024 the model expects
        image_size = (anns_["image"]["width"], anns_["image"]["height"])
        w, h = image_size[0], image_size[1]
        k = max(w,h)


        bboxes = torch.as_tensor(bboxes, dtype=torch.float32)
        bboxes = box_convert(boxes=bboxes, in_fmt="xywh", out_fmt="xyxy")

        bboxes /= k #torch.Tensor([k, k, k, k])
        bboxes *= 1024

        ## Hack to use built in resize tooling
        masks = np.array(masks).astype(np.uint8)#.transpose(1,2,0)
        masks = [self.resize.apply_image(mask) for mask in masks]
        masks = [torch.as_tensor(mask, dtype=torch.uint8) for mask in masks]
        masks = torch.stack([pad(mask, (0, padw, 0, padh)) for mask in masks])
        masks = masks.bool()

        if self.return_dict:
            return {
                "image" : image,
                "boxes" : bboxes,
                "masks" : masks,
                "ground_truth_type" : "mask"
            }

        return image, bboxes, masks, image_size

    def __len__(self):
        return len(self.imgs)






class CocoDataset(Dataset):
    """COCO Custom Dataset compatible with torch.utils.data.DataLoader."""
    def __init__(
        self,
        root,
        json,
        transform=None,
        return_dict = False,
        precomputed_clip_emb_path = "precomputed_coco_clip_emb.pkl",
        pixel_mean = torch.Tensor([[[123.675]], [[116.28 ]],  [[103.53 ]]]),
        pixel_std = torch.tensor([[[58.3950]], [[57.1200]], [[57.3750]]]),
    ):

        """Set the path for images, captions and vocabulary wrapper.

        Args:
            root: image directory.
            json: coco annotation file path.
            transform: image transformer.
        """
        self.root = root
        self.coco = COCO(json)
        self.ids = list(self.coco.anns.keys())
        self.transform = transform

        self.image_size = 1024

        self.resize = ResizeLongestSide(1024)
        self.to_tensor = torchvision.transforms.ToTensor()

        self.return_dict = return_dict

        self.precomputed_clip_emb = pickle.load(open(precomputed_clip_emb_path, "rb"))
        
        self.pixel_mean = pixel_mean
        self.pixel_std = pixel_std


    def __getitem__(self, index):
        """Returns one data pair (image and caption)."""
        coco = self.coco
        ann_id = self.ids[index]
        img_id = coco.anns[ann_id]['image_id']
        path = coco.loadImgs(img_id)[0]['file_name']


        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        
        image = cv2.imread(os.path.join(self.root, path))
        #image = Image.open(os.path.join(self.root, path)) #.convert('RGB')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #print(len(anns))
        width, height = image.size

        #numpy_image = np.array(image)
        #numpy_image = self.resize.apply_image(numpy_image)
        image = self.resize.apply_image(image.astype(np.uint8))
        #torch_image = self.to_tensor(numpy_image)
        image = torch.as_tensor(image, dtype=torch.float32).permute(2, 0, 1).contiguous()
        torch_image = transforms.Normalize(self.pixel_mean, self.pixel_mean)(image)
        ## Now resize masks to size
        # breakpoint()
        masks = torch.stack([torch.tensor(coco.annToMask(x)) for x in anns])
        masks = interpolate(masks.unsqueeze(dim=1), size = torch_image.shape[1:]).squeeze(dim=1)

        torch_image = transforms.Normalize(self.pixel_mean, self.pixel_mean)(torch_image)
        torch_image = pad_to_size(torch_image, self.image_size)
        masks = pad_to_size(masks, self.image_size)


        # Get boxes and resize
        bboxes = torch.tensor([x["bbox"] for x in anns])
        bboxes = torchvision.ops.box_convert(bboxes, in_fmt = "xywh", out_fmt = "xyxy")
        bboxes = self.resize.apply_boxes_torch(bboxes, original_size = (height, width))

        # print(anns[0].keys())

        categories = [x["category_id"] for x in anns]
        category_names = [x["name"] for x in self.coco.loadCats(categories)]
        a = [x in self.precomputed_clip_emb for x in categories]
        semantic_target = torch.stack([self.precomputed_clip_emb[x["category_id"]] for x in anns])


        if self.transform is not None:
            image = self.transform(image)
        #breakpoint()
        if self.return_dict:
            return {
                "image" : torch_image,
                "masks" : masks,
                "categories" : categories,
                "category_names" : category_names,
                "semantic_targets" : semantic_target,
                "boxes" : bboxes
            }
        return image, target_masks, categories, semantic_target, bboxes

    def __len__(self):
        return len(self.ids)


FOLDER_NAME_MAP = {
     'Watermelon': "seginw_Watermelon_val",
     'HouseHold-Items': 'seginw_HouseHold-Items_val',
     'Cows':'seginw_Cows_val',
     'Hand' : 'seginw_Hand_val',
     'Airplane-Parts': 'seginw_Airplane-Parts_val',
     'Poles' :'seginw_Poles_val',
     'Puppies' : 'seginw_Puppies_val', 
     'Fruits' : 'seginw_Fruits_val',
     'Strawberry': 'seginw_Strawberry_val',
     'Nutterfly-Squireel' : 'seginw_Nutterfly-Squireel_val',
     'Rail' : 'seginw_Rail_val',
     'Garbage' : 'seginw_Garbage_val', 
     'Toolkits' : 'seginw_Toolkits_val',
     'Tablets' : 'seginw_Tablets_val',
     'Hand-Metal' : 'seginw_Hand-Metal_val',
     'Phones' : 'seginw_Phones_val',
     'Bottles' : 'seginw_Bottles_val',
     'Elephants' : 'seginw_Elephants_val',
     'Chicken' : 'seginw_Chicken_val',
     'Electric-Shaver' :'seginw_Electric-Shaver_val',
     'Ginger-Garlic' : 'seginw_Ginger-Garlic_val',
     'Salmon-Fillet' : 'seginw_Salmon-Fillet_val',
     'Brain-Tumor' : 'seginw_Brain-Tumor_val',
     'House-Parts' : 'seginw_House-Parts_val',
     'Trash' : 'seginw_Trash_val'
}

NAME_FOLDER_MAP = {'seginw_Watermelon_val': 'Watermelon',
 'seginw_HouseHold-Items_val': 'HouseHold-Items',
 'seginw_Cows_val': 'Cows',
 'seginw_Hand_val': 'Hand',
 'seginw_Airplane-Parts_val': 'Airplane-Parts',
 'seginw_Poles_val': 'Poles',
 'seginw_Puppies_val': 'Puppies',
 'seginw_Fruits_val': 'Fruits',
 'seginw_Strawberry_val': 'Strawberry',
 'seginw_Nutterfly-Squireel_val': 'Nutterfly-Squireel',
 'seginw_Rail_val': 'Rail',
 'seginw_Garbage_val': 'Garbage',
 'seginw_Toolkits_val': 'Toolkits',
 'seginw_Tablets_val': 'Tablets',
 'seginw_Hand-Metal_val': 'Hand-Metal',
 'seginw_Phones_val': 'Phones',
 'seginw_Bottles_val': 'Bottles',
 'seginw_Elephants_val': 'Elephants',
 'seginw_Chicken_val': 'Chicken',
 'seginw_Electric-Shaver_val': 'Electric-Shaver',
 'seginw_Ginger-Garlic_val': 'Ginger-Garlic',
 'seginw_Salmon-Fillet_val': 'Salmon-Fillet',
 'seginw_Brain-Tumor_val': 'Brain-Tumor',
 'seginw_House-Parts_val': 'House-Parts',
 'seginw_Trash_val': 'Trash'}


class SEGINWDataset(torch.utils.data.IterableDataset):
    """COCO Custom Dataset compatible with torch.utils.data.DataLoader."""
    def __init__(
        self,
        root,
        dataset_name = None, ## If none, load from all sources
        transform=None,
        split = "train",
        return_dict = False,
        num_masks_per_image = 1,
        precomputed_clip_emb_path = "precomputed_seginw_clip_emb.pkl",
        pixel_mean = torch.Tensor([[[123.675]], [[116.28 ]],  [[103.53 ]]]),
        pixel_std = torch.tensor([[[58.3950]], [[57.1200]], [[57.3750]]]),
    ):


        self.root = root
        self.transform = transform
        self.image_size = 1024
        self.resize = ResizeLongestSide(1024)
        self.to_tensor = torchvision.transforms.ToTensor()
        self.return_dict = return_dict
        self.precomputed_clip_emb = pickle.load(open(precomputed_clip_emb_path, "rb"))
        self.pixel_mean = pixel_mean
        self.pixel_std = pixel_std
        self.dataset_name = dataset_name
        self.split = split
        self.num_masks_per_image = num_masks_per_image

        # Now load all the seperate COCO annotation files
        if dataset_name == None:
            jsons = [(x, os.path.join(*[root, x, split, "_annotations_min1cat.coco.json"])) for x in os.listdir(root)]
            print(jsons)
            self.cocos = [(a, COCO(b)) for a,b in jsons] 

        else:
            jsons = [(x, os.path.join(*[root, x, split, "_annotations_min1cat.coco.json"])) for x in [NAME_FOLDER_MAP[dataset_name]]]
            self.cocos = [(a, COCO(b)) for a,b in jsons] 
            print(jsons)


    def __iter__(self):
        for i in range(100000):
            try:
                # Get random dataset and random index:
                folder, coco = random.choice(self.cocos)
                index =  random.choice(list(coco.imgs.keys()))

                img_id = coco.imgs[index]['id']
                path = coco.loadImgs(img_id)[0]['file_name']


                path = os.path.join (
                    *[self.root, folder, self.split, path]
                )

                ann_ids = coco.getAnnIds(imgIds=img_id)
                anns = coco.loadAnns(ann_ids)
                image = cv2.imread(path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                width, height, _ = image.shape

                image = self.resize.apply_image(image.astype(np.uint8))
                image = torch.as_tensor(image, dtype=torch.float32).permute(2, 0, 1).contiguous()
                torch_image = transforms.Normalize(self.pixel_mean, self.pixel_mean)(image)

                ## Now resize masks to size
                anns = random.sample(anns, min(len(anns), self.num_masks_per_image))
                masks = torch.stack([torch.tensor(coco.annToMask(x)) for x in anns])
                masks = interpolate(masks.unsqueeze(dim=1), size = torch_image.shape[1:]).squeeze(dim=1)

                torch_image = pad_to_size(torch_image, self.image_size)
                masks = pad_to_size(masks, self.image_size)

                # Get boxes and resize
                bboxes = torch.tensor([x["bbox"] for x in anns])
                bboxes = torchvision.ops.box_convert(bboxes, in_fmt = "xywh", out_fmt = "xyxy")
                bboxes = self.resize.apply_boxes_torch(bboxes, original_size = (height, width))


                categories = [x["category_id"] for x in anns]
                category_names = [x["name"] for x in coco.loadCats(categories)]
                a = [x in self.precomputed_clip_emb for x in categories]
                semantic_target = [self.precomputed_clip_emb[FOLDER_NAME_MAP[folder]][x["category_id"]] for x in anns]
                #breakpoint()
                semantic_target = torch.stack(semantic_target)


                if self.transform is not None:
                    image = self.transform(image)
                if self.return_dict:
                    yield {
                        "image" : torch_image,
                        "masks" : masks,
                        "categories" : categories,
                        "category_names" : category_names,
                        "semantic_targets" : semantic_target,
                        "boxes" : bboxes
                    }
                else:
                    yield torch_image, masks, categories, semantic_target, bboxes, category_names
            except Exception as e: 
                pass
                #print (str(e))

    def __len__(self):
        return len(self.ids)