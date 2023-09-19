# Semantic Segment Anything

Semantic SAM extends SAM by also producing a semantic prediction for each mask. Semantic-SAM can be used as an open-vocabulary instance segmentation model.

![SAM design](assets/model_overview.jpg?raw=true)


## Semantic-SAM Training Data

Semantic-SAM was trained in a weakly supervised approach. We build our dataset by generating annotations for LAION image-text pairs. Contrary to previous approaches, we expand the number of candidate pseudo-labels by leveraging a language model to produce more candidates for objects that may be visible in the image but not explicitly mentioned in the caption. This bridges the gap of image captions describing an image as a whole and thus making them not ideal to derive pseudo labels. Semantic-SAM was trained without any human labeled data.


![SAM design](assets/data_overview.jpg?raw=true)


**Semantic-SAM** was trained to maintain **SAM's** high quality mask outputs


## Installation

The code requires `python>=3.8`, as well as `pytorch>=1.7` and `torchvision>=0.8`. Please follow the instructions [here](https://pytorch.org/get-started/locally/) to install both PyTorch and TorchVision dependencies. Installing both PyTorch and TorchVision with CUDA support is strongly recommended.

For conda first perform:

```
conda install -y pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia conda-forge
```

For PIP:
```
pip3 install torch torchvision torchaudio
```



Install all requirements using:

```
pip install -r requirements.txt
```



## <a name="GettingStarted"></a>Getting Started

First download a [model checkpoint](#model-checkpoints). Then follow the ssam_playground.iypnb for examples on how to use semantic-SAM.

## License
The model is licensed under the [Apache 2.0 license](LICENSE).


## acknowledgement
Big thanks to the people @ Snorkel AI for their support and making this possible.


## Notes
This work was done as part of my MSc. Thesis.
[This](https://github.com/karimumar98/semantic-sam/blob/main/short_report.pdf) report is a very short overview of the project, while [this](https://github.com/karimumar98/semantic-sam/blob/main/MSc_Thesis_Karim_Umar_ETH.pdf) report goes into more depth.


## TODO
Upload code for data generation
