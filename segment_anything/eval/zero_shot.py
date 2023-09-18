
import torch
from .metadata import COOC_2017_CLASS_NAMES, COCO_2017_ID_MAP, SEGINW_NAMES
import numpy




def get_class_probs (classifier, feature_vector, class_names = COOC_2017_CLASS_NAMES, k = 1, id_map = COCO_2017_ID_MAP, apply_softmax = True, background_scalar = 1.0, reduction = "max", logits_scale = numpy.e):

    # background_scalar: Experimental, scale background score in an attempt to reduce false positives

    weights = classifier['weights']
    class_names = classifier['classes']

    n_classes, n_templates = weights.shape[:2]
    if isinstance(feature_vector, torch.Tensor):    
        feature_vector = feature_vector.to(weights.device)
    elif isinstance(feature_vector, numpy.ndarray):
        feature_vector = torch.tensor(feature_vector).to(weights.device)
    
    logits = (logits_scale * weights.flatten(0,1) @ torch.nn.functional.normalize(feature_vector, dim = 0).T).reshape(n_classes, n_templates)
    #logits = (weights.flatten(0,1) @ torch.nn.functional.normalize(feature_vector, dim = 0).T).reshape(n_classes, n_templates)

    if reduction == "max":
        ## Choose max overlap with each class
        logits, _ = logits.max(dim = 1)
    elif reduction == "mean":
        #print(logits)
        ## Choose max overlap with each class
        logits = logits.mean(dim=1)
        #print(logits)
    else:
        raise NotImplementedError()

    if apply_softmax:
        # print(logits)
        for i,j in zip(class_names, logits):
            if i == "background":
                j *= background_scalar
            # print(i, j)
        # print(logits)
        probs, ind = logits.softmax(dim=0).topk(k = k)
    else:
        probs, ind = logits.topk(k = k)

    classes = [class_names[i] for i in ind]

    return probs, classes, [class_names.index(x) for x in classes]