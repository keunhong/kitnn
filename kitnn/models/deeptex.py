import numpy as np
import torch
from torch import nn
from kitnn.utils import load_module_npy, to_imagenet, make_batch
from kitnn.models.vgg import VGG19


class DeepTex(nn.Module):
    FEAT_LAYERS = ['conv1_1', 'pool1', 'pool2', 'pool3', 'pool4', 'pool5']

    def __init__(self, features):
        super().__init__()
        self.features = features

    @staticmethod
    def _gram_matrix(x):
        F = x.view(x.size(0), -1)
        M = F.size(1)
        G = F @ F.t() / M
        return G

    def forward(self, x, selection=FEAT_LAYERS):
        features = self.features(x, selection=selection)
        input_gvecs = []
        for i in range(x.size(0)):
            G_vecs = []
            for resp in features:
                G = self._gram_matrix(resp[i])
                G = G.view(G.size(0) * G.size(1))
                G_vecs.append(G)
            input_gvecs.append(torch.cat(G_vecs).view(1, -1))
        return torch.cat(input_gvecs, 0)

    def load_npy(self, path):
        with open(path, 'rb') as f:
            data = np.load(f)[()]
        load_module_npy(self.features, data)


def compute_feats(patches, texnet, use_cuda=True):
    patches = [to_imagenet(p) for p in patches]
    batch = make_batch(patches)
    if use_cuda:
        batch = batch.cuda()
    result = texnet(batch, selection=['conv1_1', 'pool1', 'pool2', 'pool3', 'pool4'])
    del batch
    return result.cpu().data.numpy()
