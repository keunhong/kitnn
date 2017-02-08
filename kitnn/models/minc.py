import logging
from collections import OrderedDict

import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from skimage.color import rgb2lab
from pydensecrf import densecrf
from toolbox.images import resize
from kitnn.modules import SelectiveSequential
from kitnn.utils import (SerializationMixin, load_module_npy, make_batch,
                         softmax2d)
from kitnn.models.vgg import VGG16
from kitnn.models.alexnet import AlexNet


logger = logging.getLogger(__name__)


SUBSTANCES = [
    'brick',
    'carpet',
    'ceramic',
    'fabric',
    'foliage',
    'food',
    'glass',
    'hair',
    'leather',
    'metal',
    'mirror',
    'other',
    'painted',
    'paper',
    'plastic',
    'polishedstone',
    'skin',
    'sky',
    'stone',
    'tile',
    'wallpaper',
    'water',
    'wood'
]
REMAPPED_SUBSTANCES = [
    'fabric',
    'leather',
    'wood',
    'shiny',
    'background',
]
SUBST_MAPPING = OrderedDict([
    ('fabric', 'fabric'),
    ('leather', 'leather'),
    ('wood', 'wood'),
    ('metal', 'shiny'),
    ('plastic', 'shiny'),
])
RT2 = np.sqrt(2)


def compute_remapped_probs(probs: np.ndarray):
    bg_idx = REMAPPED_SUBSTANCES.index('background')
    remapped = np.zeros((*probs.shape[:2], len(REMAPPED_SUBSTANCES)))
    for subst_old, subst_new in SUBST_MAPPING.items():
        old_idx = SUBSTANCES.index(subst_old)
        new_idx = REMAPPED_SUBSTANCES.index(subst_new)
        remapped[:, :, new_idx] += probs[:, :, old_idx]
    #bg_thres = np.median(remapped.max(axis=2))
    # bg_thres = np.percentile(remapped.max(axis=2), 33)
    remapped[remapped[:,:,:4].max(axis=2) < 0.2, bg_idx] = 1.0
    remapped = softmax2d(remapped)
    return remapped


def preprocess_image(image):
    if image.max() > 1.0:
        logger.warning("Image has values larger than 1.0, this function"
                       " expects all images to be single between 0.0 and 1.0!")
    processed = image.astype(np.float32) * 255.0
    processed = processed[:, :, [2, 1, 0]]
    processed[:, :, 0] -= 104.0
    processed[:, :, 1] -= 117.0
    processed[:, :, 2] -= 124.0
    return processed


def resize_image(image, scale=1.0, l_size=256, l_frac=0.233, order=2):
    small_dim_len = l_size / l_frac
    scale_mult = scale * small_dim_len / min(image.shape[:2])
    scale_shape = (int(image.shape[0] * scale_mult),
                   int(image.shape[1] * scale_mult))
    return resize(image, scale_shape, order=order)


def combine_probs(prob_maps, image, remap=False):
    substances = REMAPPED_SUBSTANCES if remap else SUBSTANCES
    map_scale = 550 / min(image.shape[:2])
    map_sum = np.zeros((int(image.shape[0] * map_scale),
                        int(image.shape[1] * map_scale),
                        len(substances)))
    for prob_map in prob_maps:
        if remap:
            prob_map = compute_remapped_probs(prob_map)
        prob_map = resize(prob_map, map_sum.shape[:2], order=3)
        map_sum += prob_map
    return softmax2d(map_sum / len(prob_maps))


def compute_probs_multiscale(image, mincnet,
                             scales=list([RT2, 1.0, 1 / RT2]),
                             use_cuda=True):
    prob_maps = []
    feat_dicts = []
    for scale in scales:
        image_scaled = resize_image(image, scale=scale)
        logger.info("\tProcessing scale={:.4}, shape={}"
                    .format(scale, image_scaled.shape))
        batch_arr = make_batch([image_scaled])
        if use_cuda:
            batch_arr = batch_arr.cuda()
        batch = Variable(batch_arr, volatile=True)
        prob_map, sel_dict = mincnet(batch, selection=['fc8-20', 'softmax'])
        prob_map_numpy = prob_map.cpu().data.numpy()[0].transpose((1, 2, 0))
        prob_maps.append(prob_map_numpy)
        feat_dicts.append(sel_dict)
    return prob_maps, feat_dicts


def compute_probs_crf(
        image, prob_map, theta_p=0.1, theta_L=20.0, theta_ab=5.0):
    image_lab = rgb2lab(resize(image, prob_map.shape[:2]))

    p_y, p_x = np.mgrid[0:image_lab.shape[0], 0:image_lab.shape[1]]

    feats = np.zeros((5, *image_lab.shape[:2]), dtype=np.float32)
    d = min(image_lab.shape[:2])
    feats[0] = p_x / (theta_p * d)
    feats[1] = p_y / (theta_p * d)
    feats[2] = image_lab[:, :, 0] / theta_L
    feats[3] = image_lab[:, :, 1] / theta_ab
    feats[4] = image_lab[:, :, 2] / theta_ab
    crf = densecrf.DenseCRF2D(*prob_map.shape)
    unary = np.rollaxis(
        -np.log(prob_map), axis=-1).astype(dtype=np.float32, order='c')
    crf.setUnaryEnergy(np.reshape(unary, (prob_map.shape[-1], -1)))
    crf.addPairwiseEnergy(np.reshape(feats, (feats.shape[0], -1)),
                          compat=3)

    Q = crf.inference(20)
    Q = np.array(Q).reshape((-1, *prob_map.shape[:2]))
    return np.rollaxis(Q, 0, 3)


class MincAlexNet(nn.Module, SerializationMixin):
    def __init__(self):
        super().__init__()
        self.features = AlexNet().features
        self.classifier = SelectiveSequential(OrderedDict([
            ('fc6', nn.Conv2d(256, 4096, kernel_size=6, stride=1, padding=3)),
            ('relu6', nn.ReLU(inplace=True)),
            ('fc7', nn.Conv2d(4096, 4096, kernel_size=1, stride=1)),
            ('relu7', nn.ReLU(inplace=True)),
            ('fc8-20', nn.Conv2d(4096, 23, kernel_size=1, stride=1)),
            ('softmax', nn.Softmax2d())
        ]))

    def forward(self, x):
        features = self.features(x, selection=['pool5'])[0]
        softmax = self.classifier(features, ['softmax'])[0]
        return softmax

    def load_npy(self, path):
        with open(path, 'rb') as f:
            data = np.load(f)[()]
        load_module_npy(self.features, data)
        load_module_npy(self.classifier, data)


class MincVGG(nn.Module, SerializationMixin):
    def __init__(self):
        super().__init__()
        self.features = VGG16().features
        self.classifier = SelectiveSequential(OrderedDict([
            ('fc6', nn.Conv2d(512, 4096, kernel_size=7, stride=1, padding=3)),
            ('relu6', nn.ReLU(True)),
            ('fc7', nn.Conv2d(4096, 4096, kernel_size=1, stride=1)),
            ('relu7', nn.ReLU(True)),
            ('fc8-20', nn.Conv2d(4096, 23, kernel_size=1, stride=1)),
            ('softmax', nn.Softmax2d())
        ]))

    def forward(self, x, selection=list(['softmax'])):
        feature_sel = {'pool5'}
        classifier_sel = set()
        for sel in selection:
            if sel in self.features.modules_dict:
                feature_sel.add(sel)
            elif sel in self.classifier.modules_dict:
                classifier_sel.add(sel)
        x, features = self.features(x, selection=feature_sel)
        x, classifier = self.classifier(x, selection=classifier_sel)
        return x, {**features, **classifier}

    def load_npy(self, path):
        with open(path, 'rb') as f:
            data = np.load(f)[()]
        load_module_npy(self.features, data)
        load_module_npy(self.classifier, data)
