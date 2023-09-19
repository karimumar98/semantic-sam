


## Create a simple zero shot classfier on top of the coco classes
import requests
from torch.nn.functional import threshold, normalize
import torch
import pickle
from pycocotools.coco import COCO
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from groundingdino.util.inference import annotate
from torchvision.ops.boxes import box_convert

## Random stuff for visualizing the results

def annotate_and_draw (
    image,
    filtered_masks
):
    mask =  torch.tensor(np.array([x['segmentation'] for x in filtered_masks]))
    cls    =  [x['category'] for x in filtered_masks]
    boxes = torch.tensor([x['bbox'] for x in filtered_masks])
    boxes = box_convert(boxes=boxes, in_fmt="xywh", out_fmt="cxcywh")/ torch.tensor([image.shape[1], image.shape[0]] * 2)
    logits =  [x['score'] for x in filtered_masks]

    annotated_frame = annotate(
        image_source=np.asarray(image), 
        boxes=boxes, 
        logits=logits, 
        phrases=cls)

    annotated_frame = show_masks(
        np.asarray(mask), 
        annotated_frame)
    return annotated_frame

## Helper method to annotate images
def caption_image(img, caption):
    w, h = img.size
    im2 = Image.new("RGBA", (w + 10, h + (h//10)), "white")
    im2.paste(img, (5,5,(w + 5), (h + 5)))
    font = ImageFont.load_default()
    
    
    w_c, h_c = font.getsize(caption)
    draw = ImageDraw.Draw(im2)
    # Add Text to an image
    draw.text(
        (
            ((w-w_c)//2, 
            (h+((h/10)-h_c)//2))
        ), 
        caption, font=font, fill ="black")
    return im2

def show_masks(masks, image, color=None):
    annotated_frame_pil = Image.fromarray(image).convert("RGBA")
    composite_image = annotated_frame_pil
    mask_images = []
    for mask in masks:
        if not isinstance(color, np.ndarray) and color == None:
            color_ = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)

        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color_.reshape(1, 1, -1)

        mask_image_pil = Image.fromarray((mask_image * 255).astype(np.uint8)).convert("RGBA")
        mask_images += [mask_image_pil]

        composite_image = Image.alpha_composite(
            composite_image,
            mask_image_pil
        )
    return np.array(composite_image)

def show_mask(mask, ax, color=None):
    if not isinstance(color, np.ndarray) and color == None:
        color = np.concatenate([np.random.random(3), np.array([0.8])], axis=0)
        # else:
        #     color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax, edgecolor = "green"):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor=edgecolor, facecolor=(0,0,0,0), lw=2, label='Label'))    
    

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = max([x.size[0] for x in imgs]), max([x.size[1] for x in imgs])
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid


def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    polygons = []
    color = []
    for ann in sorted_anns:
        m = ann['segmentation']
        img = np.ones((m.shape[0], m.shape[1], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]
        color_mask = [0.9, 0.9, 0.9]
        for i in range(3):
            img[:,:,i] = color_mask[i]
        ax.imshow(np.dstack((img, m*0.9)))

