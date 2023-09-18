import sys
sys.path.append("..")
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
from segment_anything.utils.config import configurable
from segment_anything.modeling.mask_decoder import MLP
from torch import nn
import torch
import numpy as np
from torch.nn.functional import threshold, normalize, pad, interpolate, binary_cross_entropy_with_logits, mse_loss, softmax, cross_entropy
from typing import Dict, List, Optional, Tuple


class Semantic_SAM(nn.Module):
    # Essentially just a wrapper around SAM with a modified mask decoder
    @configurable
    def __init__(
        self, 
        model_type, 
        checkpoint, 
        clip_dim = 768, 
        device = "cuda", 
        text_encoder = False,
        use_semantic_tokens = True,
        use_semantic_hyper_networks = False,
        image_encoder_grad = False
        ):

        super(Semantic_SAM, self).__init__()
        ## Do not supply checkpoint, as the keys will not match
        self.sam = sam_model_registry[model_type](checkpoint=None)
        self.sam.mask_decoder.use_semantic_hyper_networks = use_semantic_hyper_networks

        embed_dim = self.sam.prompt_encoder.embed_dim
        if text_encoder:
            self.text_encoder = MLP(clip_dim, embed_dim, embed_dim, 3).to(device)
        self.text_decoder = MLP(embed_dim, embed_dim, clip_dim, 3)

        self.temperature = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.device = device
        self.load_checkpoint (checkpoint, device)
        self.to(device)
        self.inference = False
        self.clip_dim = clip_dim
        self.use_semantic_tokens = use_semantic_tokens
        self.image_encoder_grad = image_encoder_grad

    @classmethod
    def from_config (cls, cfg):

        print("Initializing Model from Config")
        model_cfg = cfg["model"]

        # These args must be present
        checkpoint = model_cfg["checkpoint"]
        model_type = model_cfg["model_type"]

        clip_dim = model_cfg.get("clip_dim", 768)
        device = model_cfg.get("device", "cuda")
        text_encoder = model_cfg.get("text_encoder", False)
        use_semantic_tokens = model_cfg.get("use_semantic_tokens", False)
        use_semantic_hyper_networks = model_cfg.get("use_semantic_hyper_networks", False)


        try:
            image_encoder_grad = cfg["training"]['train_params']['sam_image_encoder']
        except KeyError:
            image_encoder_grad = False  # or whatever

        
        return {
            "model_type" : model_type, 
            "checkpoint" : checkpoint, 
            "clip_dim": clip_dim, 
            "device" : device, 
            "text_encoder" : text_encoder,
            "use_semantic_tokens" : use_semantic_tokens,
            "use_semantic_hyper_networks" : use_semantic_hyper_networks,
            "image_encoder_grad" : image_encoder_grad,
        }



    def load_checkpoint (self, checkpoint, device_id):
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f, map_location=device_id)

        if "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        try:
            print(self.load_state_dict(state_dict))
        except Exception as e:
            #print(e)
            print("Skipping loading of unmatched parameters")
            if "sam_vit_" in checkpoint:
                ## Handle cam checkpoints
                print(self.sam.load_state_dict(state_dict, strict=False))
            else:
                print(self.load_state_dict(state_dict, strict=False))

    def encode_image(self, images):
        image_embedding = self.sam.image_encoder(images)
        return image_embedding

    def encode_prompt(self, boxes):
        sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
            points=None,
            boxes=boxes,
            masks=None,
        )
        return sparse_embeddings, dense_embeddings

    def decode(self, image_embedding, sparse_embeddings, dense_embeddings, text_embeddings=None, multimask_output = False, normalize_text_embeddings = True):
        
        low_res_masks, iou_predictions, semantic_tokens, cosine_predictions = self.sam.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=self.sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
            use_semantic_tokens = self.use_semantic_tokens,
        )

        semantic_features = self.text_decoder(semantic_tokens)
        if normalize_text_embeddings:
            #breakpoint()
            semantic_features = normalize(semantic_features, dim = 2)

        masks = interpolate(
            low_res_masks,
            (self.sam.image_encoder.img_size, self.sam.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)

        return masks, semantic_features, iou_predictions, cosine_predictions

    def _move_to_current_device(self, x):
        return move_device_like(x, self.pixel_mean)

    def preprocess_image(self, batched_inputs):
        # images = [self._move_to_current_device(x["image"]) for x in batched_inputs]
        images = torch.stack([x["image"] for x in batched_inputs]).to(self.device)
        return images
        ## Maybe: do some preprocessing here instead of in dataloader, resize, normalize etc

    def forward(
            self,
            batched_inputs: List[Dict[str, torch.Tensor]],
            multimask_output = False):

        if "image" in batched_inputs[0]:
            images = self.preprocess_image(batched_inputs)
            if self.image_encoder_grad:
                image_embeddings = self.encode_image(images)
            else:
                with torch.no_grad():
                    image_embeddings = self.encode_image(images)

        else:
            # Inference, image features pre-computed
            image_embeddings = [x["image_features"] for x in batched_inputs]
            
        outputs = []
        for image_record, curr_embedding in zip(batched_inputs, image_embeddings):
            if "point_coords" in image_record:
                # breakpoint()
                points = image_record["point_coords"].to(self.device), image_record["point_labels"].to(self.device)
            else:
                points = None

            if "box_proompts" in image_record:
                boxes=image_record.get("box_proompts", None).to(self.device)
            else:
                boxes = None

            if "semantic_prompt" in image_record:
                #breakpoint()
                text_embeddings = image_record['semantic_prompt'].to(self.device).float()
                text_embeddings = self.text_encoder(text_embeddings).unsqueeze(1)

            else:
                text_embeddings = None

            if "mask_prompt" in image_record:
                mask_inputs = image_record.get("mask_prompt").to(self.device)
            else:
                mask_inputs = None

            sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
                points=points,
                boxes=boxes,
                masks=mask_inputs,
                text_embeddings = text_embeddings
            )

            pred_masks, pred_semantic, iou_predictions, cosine_predictions = self.decode(
                curr_embedding, 
                sparse_embeddings, 
                dense_embeddings, 
                text_embeddings = text_embeddings,
                multimask_output = multimask_output)
            outputs.append(
                {
                    "masks": pred_masks,
                    "iou_predictions": iou_predictions,
                    "semantic_features" : pred_semantic,
                    "cosine_predictions" : cosine_predictions
                }
            )

        return outputs, self.temperature.exp()