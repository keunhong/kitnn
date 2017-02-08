import numpy as np
import torch
from torch import nn
from kitnn.utils import load_module_npy, to_imagenet, make_batch


LAYERS = ['conv1_1', 'pool1', 'pool2', 'pool3', 'pool4', 'pool5']


def make_gram_matrix(activations):
    """
    Activations should be of size o x m x n where o is the output dimension
    and m x n is the output size of the kernel.
    """
    dim_out = activations.size(0)
    dim_h = activations.size(1)
    dim_w = activations.size(2)
    m = dim_h * dim_w
    activations = activations.view(dim_out, m) # Vectorize.
    G = activations.mm(activations.t()) / m
    return G


def make_gram_features(activations):
    G = make_gram_matrix(activations)
    return G.view(G.size(0) * G.size(1))


def gram_features(feat_dict):
    n_inputs = list(feat_dict.values())[0].size(0)
    input_gvecs = []
    for i in range(n_inputs):
        G_vecs = []
        for resp in feat_dict.values():
            G = make_gram_features(resp[i])
            G_vecs.append(G)
        input_gvecs.append(torch.cat(G_vecs).view(1, -1))
    return torch.cat(input_gvecs, 0)
