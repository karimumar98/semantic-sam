# Semantic Segment Anything

Semantic SAM extends SAM by also producing a semantic prediction for each mask. Semantic-SAM can be used as an open-vocabulary instance segmentation model.

![SAM design](assets/model_overview.jpg?raw=true)


## Semantic SAM Training Data

Semantic SAM was trained in a weakly supervised approach. We build our dataset by generating annotations for LAION image-text pairs. Contrary to previous approaches, we expand the number of candidate pseudo-labels by leveraging a language model to produce more candidates for objects that may be visible in the image but not explicitly mentioned in the caption. This bridges the gap of image captions describing an image as a whole and thus making them not ideal to derive pseudo labels. 


![SAM design](assets/data_overview.jpg?raw=true)


**Semantic-SAM** was trained to maintain **SAM** high quality mask outputs


## Installation

The code requires `python>=3.8`, as well as `pytorch>=1.7` and `torchvision>=0.8`. Please follow the instructions [here](https://pytorch.org/get-started/locally/) to install both PyTorch and TorchVision dependencies. Installing both PyTorch and TorchVision with CUDA support is strongly recommended.

Install all requirements using:

```
pip install -r requirements.txt
```



## <a name="GettingStarted"></a>Getting Started

First download a [model checkpoint](#model-checkpoints). Then follow the ssam_playground.iypnb for examples on how to use semantic-SAM.

## License
The model is licensed under the [Apache 2.0 license](LICENSE).


## TODO
Upload code for data generation
