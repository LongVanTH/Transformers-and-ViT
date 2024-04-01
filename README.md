# Assignment 3: Transformers and Vision Transformers (ViT)

This repository contains my solutions for [Assignment 3](https://github.com/visual-learning/transformers/tree/0810e29a8513b4b0a9ce6c3249bbbdb520b49b5c) from the course **16-824: Visual Learning and Recognition** offered in Spring 2024 at Carnegie Mellon University.

## Setup

Most of the dependencies can be installed via `pip install -r requirements.txt`. You can also install them manually using the following commands:

`pip install torch torchvision numpy==1.23.0 h5py imageio matplotlib`

## Usage

Follow the instructions in the READMEs of the respective directories [`transformers_captioning/`](https://github.com/LongVanTH/Transformers-and-ViT/tree/main/transformer_captioning) and [`vit_classification/`](https://github.com/LongVanTH/Transformers-and-ViT/tree/main/vit_classification) to run the code (simply run `run.py`).

These instructions are reproduced below.

# Image Captioning with Transformers

Please attempt this question sequentially, as the parts build upon previous sections. We will build a transformer decoder and use it for image captioning. Please read 
["Attention Is All You Need"](https://arxiv.org/abs/1706.03762), which introduced transformers.

## Dataset

Download the dataset by running `./get_coco_captioning.sh` in the `transformer_captioning/datasets` directory.

## Q.1.a : Attention 

Complete the TODOs in the `AttentionLayer` class in the file `transformer.py`

Given query $q$, key $k$ and value $v$, these are first projected into the same embedding dimension using separate linear projections. 

The attention output is then given by : 

$$Y = \text{dropout}\bigg(\text{softmax}\bigg(\frac{Q.K^\top + M}{\sqrt{d}}\bigg)\bigg)V$$

where Q, K and V are matrices containing rows of projected queries, keys and values respectively. M is an additive mask, which is used to restrict where attention is applied.

## Q.1.b : Multi-head Attention 

Complete the TODOs in the `MultiHeadAttentionLayer` class in the file `transformer.py`

For the model to have more expressivity, we can add more heads to allow it to attend to different parts of the input. 
For this we split the query, key and value matrices Q,K,V along the embedding dimension, and attention is performed on each of these separately. 
For the ith head, the output is given by : 

$$Y_i = \text{dropout}\bigg(\text{softmax}\bigg(\frac{Q_i.K_i^\top + M}{\sqrt{d/h}}\bigg)\bigg)V_i$$

where $Y_i\in\mathbb{R}^{\ell \times d/h}$, where $\ell$ is our sequence length.

These are then concatenated and projected to the embedding dimension to obtain the overall output:
$$Y = [Y_1;\dots;Y_h]A$$

## Q.1.c : Positional Encoding 

Complete the TODOs in the `PositionalEncoding` class in the file `transformer.py`

While transformers can aggregate information from across the sequence, we need to provide information about the ordering of the tokens. This can be done using a special code for each token, which is precomputed and fixed, and added to the sequence. 

## Q.1.d : Transformer Decoder Layer

Complete the TODOs in the `SelfAttentionBlock`,  `CrossAttentionBlock` and `FeedForwardBlock` classes in the file `transformer.py`

A transformer decoder layer consists of three blocks - a masked self-attention block, a cross-attention block that uses conditioning features (no mask), and a feedforward block, as described in the paper  ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762). The structure of these blocks is explained in the code comments, and also in the paper. The transformer decoder is formed by stacking a number of such layers together. 

## Q.1.e : Transformer Decoder \& Captioning

Most of the implementation for this class has been provided, including auto-regressively predicting caption tokens. Please fill out remaining TODOs in `TransformerDecoder` in the file `transformer.py`, relating to projecting the captions and features into `embed_dim` dimensions, and getting the causal mask. Also fill out the TODOs in `Trainer` in the file `trainer.py` for computing the loss between predictions and labels. 

Now, run `run.py` with the following configurations -
1) num_heads : 2, num_layers : 2, learning_rate : 1e-4
2) num_heads : 4, num_layers : 6, learning_rate : 1e-4
3) num_heads : 4, num_layers : 6, learning_rate : 1e-3

Include loss plot and 2 images from the training set for each at the end of 100 epochs. These models don't perform well on the validation set since we're training on a small subset of the training data.

# Classification with Vision Transformers

The previous question used attention over image features provided in the COCO dataset. But what if we want to use attention directly over the image? Recent works have explored this, and in this question we will implement a Vision Transformer (ViT), as described in this [paper](https://arxiv.org/pdf/2010.11929.pdf), for the task of image classification. 

## Q.2.a : Initialization 

Complete the TODOs in the `__init__` function in the file `vit.py`. In addition to projection and encoding layers, this includes a class token. This is a learnable parameter, which you will use in part c. 

## Q.2.b : Patchification

Complete the TODOs in the `patchify` function in the file `vit.py`

The vision transformer breaks the image into a set of patches, each of which is then projected into a corresponding token to be attended over. This allows the transformer to learn representations using attention directly from pixels. 

## Q.2.c : ViT Forward Pass

Complete the TODOs in the `forward` function in the file `vit.py`

This includes utlizing the class token. This should be included at the beginning of the sequence before being passed to the transformer. The class prediction only uses the first token from the output sequence, which corresponds to the class token. 

## Q.2.d : Loss \& Classification

Complete the TODO in the `loss` function in the file `trainer.py` 

After this, train the model on CIFAR10 using `run.py`. Include the train and test accuracy, and the training loss in your homework pdf submission. Note that on datasets of this small size, training a ViT from scratch as we have done here does not yield better results than using a convolutional network. 
