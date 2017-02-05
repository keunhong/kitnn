from collections import OrderedDict
import numpy as np
from torch import nn
from kitnn.modules import SelectiveSequential, LRN
from kitnn.utils import load_module_npy, SerializationMixin


class AlexNet(nn.Module, SerializationMixin):
    def __init__(self):
        super().__init__()
        self.features = SelectiveSequential(OrderedDict([
            ('conv1', nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0)),
            ('relu1', nn.ReLU(inplace=True)),
            ('norm1', LRN(5, 0.0001, 0.75, 1)),
            ('pool1', nn.MaxPool2d(kernel_size=3, stride=2)),

            ('conv2', nn.Conv2d(96, 256, kernel_size=5, padding=2, groups=2)),
            ('relu2', nn.ReLU(inplace=True)),
            ('norm2', LRN(5, 0.0001, 0.75, 1)),
            ('pool2', nn.MaxPool2d(kernel_size=3, stride=2)),

            ('conv3', nn.Conv2d(256, 384, kernel_size=3, padding=1)),
            ('relu3', nn.ReLU(inplace=True)),

            ('conv4', nn.Conv2d(384, 384, kernel_size=3, padding=1, groups=2)),
            ('relu4', nn.ReLU(inplace=False)),

            ('conv5', nn.Conv2d(384, 256, kernel_size=3, padding=1, groups=2)),
            ('relu5', nn.ReLU(inplace=True)),
            ('pool5', nn.MaxPool2d(kernel_size=3, stride=2)),
        ]))

    def forward(self, x, selection=list()):
        return self.features(x, selection)

    def load_npy(self, path):
        with open(path, 'rb') as f:
            data = np.load(f)[()]
        load_module_npy(self.features, data)
