from collections import OrderedDict

import numpy as np
from torch import nn
from kitnn.modules import SelectiveSequential
from kitnn.utils import SerializationMixin, load_module_npy
from kitnn.models.vgg import VGG16
from kitnn.models.alexnet import AlexNet


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


SUBST_REMAPPED = [
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


def _softmax2d(x):
    e_x = np.exp(x - np.max(x, axis=0)[None, :, :])
    return e_x / e_x.sum(axis=0) # only difference


def remap_softmax2d(softmax2d: np.ndarray):
    remapped = np.zeros((len(SUBST_REMAPPED), *softmax2d.shape[1:]))
    for subst_old, subst_new in SUBST_MAPPING.items():
        old_idx = SUBSTANCES.index(subst_old)
        new_idx = SUBST_REMAPPED.index(subst_new)
        remapped[new_idx] += softmax2d[old_idx]
    remapped[SUBST_REMAPPED.index('background'), remapped[:4].max(axis=0) < 0.2] = 1.0
    return _softmax2d(remapped)


def preprocess_image(image):
    processed = image.astype(np.float32)
    processed = processed[:, :, [2, 1, 0]]
    processed[:, :, 0] -= 104
    processed[:, :, 1] -= 117
    processed[:, :, 2] -= 124
    return processed


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
        features = self.features(x, selection=['pool5'])[0]
        return self.classifier(features, selection=selection)

    def load_npy(self, path):
        with open(path, 'rb') as f:
            data = np.load(f)[()]
        load_module_npy(self.features, data)
        load_module_npy(self.classifier, data)
