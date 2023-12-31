TEMPLATES_1 = [
    lambda c: f"{c}"
]

TEMPLATES_2 = [
    lambda c: f"a {c}",
    lambda c: f"{c}"
]

TEMPLATES_5 = [
    lambda c: f"an image of a {c}",
    lambda c: f"a {c}",
    lambda c: f"{c}",
    lambda c: f"A toy {c}",
    lambda c: f"several {c}",
]

# Copied from: https://github.com/mlfoundations/open_clip/blob/f692ec95e1bf30d50aeabe2fd32008cdff53ef5e/src/open_clip/zero_shot_metadata.py#L4
OPENAI_IMAGENET_TEMPLATES = (
    lambda c: f'a bad photo of a {c}.',
    lambda c: f'a photo of many {c}.',
    lambda c: f'a sculpture of a {c}.',
    lambda c: f'a photo of the hard to see {c}.',
    lambda c: f'a low resolution photo of the {c}.',
    lambda c: f'a rendering of a {c}.',
    lambda c: f'graffiti of a {c}.',
    lambda c: f'a bad photo of the {c}.',
    lambda c: f'a cropped photo of the {c}.',
    lambda c: f'a tattoo of a {c}.',
    lambda c: f'the embroidered {c}.',
    lambda c: f'a photo of a hard to see {c}.',
    lambda c: f'a bright photo of a {c}.',
    lambda c: f'a photo of a clean {c}.',
    lambda c: f'a photo of a dirty {c}.',
    lambda c: f'a dark photo of the {c}.',
    lambda c: f'a drawing of a {c}.',
    lambda c: f'a photo of my {c}.',
    lambda c: f'the plastic {c}.',
    lambda c: f'a photo of the cool {c}.',
    lambda c: f'a close-up photo of a {c}.',
    lambda c: f'a black and white photo of the {c}.',
    lambda c: f'a painting of the {c}.',
    lambda c: f'a painting of a {c}.',
    lambda c: f'a pixelated photo of the {c}.',
    lambda c: f'a sculpture of the {c}.',
    lambda c: f'a bright photo of the {c}.',
    lambda c: f'a cropped photo of a {c}.',
    lambda c: f'a plastic {c}.',
    lambda c: f'a photo of the dirty {c}.',
    lambda c: f'a jpeg corrupted photo of a {c}.',
    lambda c: f'a blurry photo of the {c}.',
    lambda c: f'a photo of the {c}.',
    lambda c: f'a good photo of the {c}.',
    lambda c: f'a rendering of the {c}.',
    lambda c: f'a {c} in a video game.',
    lambda c: f'a photo of one {c}.',
    lambda c: f'a doodle of a {c}.',
    lambda c: f'a close-up photo of the {c}.',
    lambda c: f'a photo of a {c}.',
    lambda c: f'the origami {c}.',
    lambda c: f'the {c} in a video game.',
    lambda c: f'a sketch of a {c}.',
    lambda c: f'a doodle of the {c}.',
    lambda c: f'a origami {c}.',
    lambda c: f'a low resolution photo of a {c}.',
    lambda c: f'the toy {c}.',
    lambda c: f'a rendition of the {c}.',
    lambda c: f'a photo of the clean {c}.',
    lambda c: f'a photo of a large {c}.',
    lambda c: f'a rendition of a {c}.',
    lambda c: f'a photo of a nice {c}.',
    lambda c: f'a photo of a weird {c}.',
    lambda c: f'a blurry photo of a {c}.',
    lambda c: f'a cartoon {c}.',
    lambda c: f'art of a {c}.',
    lambda c: f'a sketch of the {c}.',
    lambda c: f'a embroidered {c}.',
    lambda c: f'a pixelated photo of a {c}.',
    lambda c: f'itap of the {c}.',
    lambda c: f'a jpeg corrupted photo of the {c}.',
    lambda c: f'a good photo of a {c}.',
    lambda c: f'a plushie {c}.',
    lambda c: f'a photo of the nice {c}.',
    lambda c: f'a photo of the small {c}.',
    lambda c: f'a photo of the weird {c}.',
    lambda c: f'the cartoon {c}.',
    lambda c: f'art of the {c}.',
    lambda c: f'a drawing of the {c}.',
    lambda c: f'a photo of the large {c}.',
    lambda c: f'a black and white photo of a {c}.',
    lambda c: f'the plushie {c}.',
    lambda c: f'a dark photo of a {c}.',
    lambda c: f'itap of a {c}.',
    lambda c: f'graffiti of the {c}.',
    lambda c: f'a toy {c}.',
    lambda c: f'itap of my {c}.',
    lambda c: f'a photo of a cool {c}.',
    lambda c: f'a photo of a small {c}.',
    lambda c: f'a tattoo of the {c}.',
)


# COCO 2017 Classes
COOC_2017_CLASS_NAMES = [
'person',
 'bicycle',
 'car',
 'motorcycle',
 'airplane',
 'bus',
 'train',
 'truck',
 'boat',
 'traffic light',
 'fire hydrant',
 'stop sign',
 'parking meter',
 'bench',
 'bird',
 'cat',
 'dog',
 'horse',
 'sheep',
 'cow',
 'elephant',
 'bear',
 'zebra',
 'giraffe',
 'backpack',
 'umbrella',
 'handbag',
 'tie',
 'suitcase',
 'frisbee',
 'skis',
 'snowboard',
 'sports ball',
 'kite',
 'baseball bat',
 'baseball glove',
 'skateboard',
 'surfboard',
 'tennis racket',
 'bottle',
 'wine glass',
 'cup',
 'fork',
 'knife',
 'spoon',
 'bowl',
 'banana',
 'apple',
 'sandwich',
 'orange',
 'broccoli',
 'carrot',
 'hot dog',
 'pizza',
 'donut',
 'cake',
 'chair',
 'couch',
 'potted plant',
 'bed',
 'dining table',
 'toilet',
 'tv',
 'laptop',
 'mouse',
 'remote',
 'keyboard',
 'cell phone',
 'microwave',
 'oven',
 'toaster',
 'sink',
 'refrigerator',
 'book',
 'clock',
 'vase',
 'scissors',
 'teddy bear',
 'hair drier',
 'toothbrush'
]

# Maps index in classifier to COCO_CLASS
COCO_2017_ID_MAP = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]

SEGINW_NAMES = {'seginw_Elephants_val': ['elephant', 'background'],
 'seginw_Hand-Metal_val': ['hand', 'metal', 'background'],
 'seginw_Watermelon_val': ['watermelon', 'background'],
 'seginw_House-Parts_val': ['aluminium door',
  'aluminium window',
  'cellar window',
  'mint cond roof',
  'plaster',
  'plastic door',
  'plastic window',
  'plate fascade',
  'wooden door',
  'wooden fascade',
  'wooden window',
  'worn cond roof',
  'background'],
 'seginw_HouseHold-Items_val': ['bottle',
  'mouse',
  'perfume',
  'phone',
  'background'],
 'seginw_Strawberry_val': ['R_strawberry', 'people', 'background'],
 'seginw_Fruits_val': ['apple',
  'lemon',
  'orange',
  'pear',
  'strawberry',
  'background'],
 'seginw_Nutterfly-Squireel_val': ['butterfly', 'squirrel', 'background'],
 'seginw_Hand_val': ['Hand-Segmentation', 'hand', 'background'],
 'seginw_Garbage_val': ['bin', 'garbage', 'pavement', 'road', 'background'],
 'seginw_Chicken_val': ['chicken', 'background'],
 'seginw_Rail_val': ['rail', 'background'],
 'seginw_Airplane-Parts_val': ['Airplane',
  'Body',
  'Cockpit',
  'Engine',
  'Wing',
  'background'],
 'seginw_Brain-Tumor_val': ['tumor', 'background'],
 'seginw_Poles_val': ['poles', 'background'],
 'seginw_Electric-Shaver_val': ['caorau', 'background'],
 'seginw_Bottles_val': ['bottle', 'can', 'label', 'background'],
 'seginw_Toolkits_val': ['Allen-key',
  'block',
  'gasket',
  'plier',
  'prism',
  'screw',
  'screwdriver',
  'wrench',
  'background'],
 'seginw_Trash_val': ['Aluminium foil',
  'Cigarette',
  'Clear plastic bottle',
  'Corrugated carton',
  'Disposable plastic cup',
  'Drink Can',
  'Egg Carton',
  'Foam cup',
  'Food Can',
  'Garbage bag',
  'Glass bottle',
  'Glass cup',
  'Metal bottle cap',
  'Other carton',
  'Other plastic bottle',
  'Paper cup',
  'Plastic bag - wrapper',
  'Plastic bottle cap',
  'Plastic lid',
  'Plastic straw',
  'Pop tab',
  'Styrofoam piece',
  'background'],
 'seginw_Salmon-Fillet_val': ['Salmon_fillet', 'background'],
 'seginw_Puppies_val': ['puppy', 'background'],
 'seginw_Tablets_val': ['tablets', 'background'],
 'seginw_Phones_val': ['phone', 'background'],
 'seginw_Cows_val': ['cow', 'background'],
 'seginw_Ginger-Garlic_val': ['garlic', 'ginger', 'background']}



