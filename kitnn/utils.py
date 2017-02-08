import numpy as np
import torch
from torch.autograd import Variable


IMAGENET_MEAN = np.array([0.40760392, 0.45795686, 0.48501961])


class SerializationMixin:
    def load_npy(self, path):
        raise NotImplementedError()

    def save_pth(self, path):
        with open(path, 'wb') as f:
            torch.save(self, path)


def make_batch(images):
    batch = torch.from_numpy(np.stack(images, axis=3)
                             .transpose((3, 2, 0, 1))
                             .astype(dtype=np.float32))
    return batch


def load_module_npy(module, data):
    for name, child in module._modules.items():
        if name in data:
            print("Loading {} => {}".format(name, child))
            weight_shape = tuple(child.weight.size())
            weights = data[name]['weights']
            if weight_shape != weights.shape:
                print("\tReshaping weight {} => {}"
                      .format(weights.shape, weight_shape))
                weights = weights.reshape(weight_shape)
            weights = torch.from_numpy(weights)
            bias = data[name]['biases']
            bias = torch.from_numpy(bias)
            child.weight.data.copy_(weights)
            child.bias.data.copy_(bias)


def to_imagenet(image):
    image = image.astype(np.float32)
    image = image[:, :, [2, 1, 0]]
    image -= IMAGENET_MEAN[None, None, :]
    image *= 255.0
    return image


def from_imagenet(image):
    image += IMAGENET_MEAN[None, None, :]
    image = image[:, :, [2, 1, 0]]
    return image


def softmax2d(x):
    e_x = np.exp(x - np.max(x, axis=-1)[:, :, None])
    return e_x / e_x.sum(axis=-1)[:, :, None]
