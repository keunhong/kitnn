import math
import logging
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

from kitnn.functions import StablePow

logger = logging.getLogger(__name__)


IMAGENET_MEAN = np.array([0.40760392, 0.45795686, 0.48501961])
SOBEL_KERNEL_X = Variable(torch.from_numpy(
    np.array([(1, 0, -1),
              (2, 0, -2),
              (1, 0, -1)]).astype(dtype=np.float32)),
                          requires_grad=False)
SOBEL_KERNEL_X = SOBEL_KERNEL_X.view(1, 1, *SOBEL_KERNEL_X.size())
SOBEL_KERNEL_Y = Variable(torch.from_numpy(
    np.array([(1, 2, 1),
              (0, 0, 0),
              (-1, -2, -1)]).astype(dtype=np.float32)),
                          requires_grad=False)
SOBEL_KERNEL_Y = SOBEL_KERNEL_Y.view(1, 1, *SOBEL_KERNEL_Y.size())


class SerializationMixin:
    def load_npy(self, path):
        raise NotImplementedError()

    def save_pth(self, path):
        with open(path, 'wb') as f:
            torch.save(self, path)


def make_batch(images, flatten=False):
    if len(images[0].shape) == 2:
        images = [i[:, :, None] for i in images]
    batch = np.stack(images, axis=3) \
        .transpose((3, 2, 0, 1)) \
        .astype(dtype=np.float32)
    batch = torch.from_numpy(batch).contiguous()
    if flatten:
        batch.resize_(*batch.size()[:2], batch.size(2) * batch.size(3))
    return batch


def load_module_npy(module, data):
    for name, child in module._modules.items():
        if name in data:
            logger.info("Loading {} => {}".format(name, child))
            weight_shape = tuple(child.weight.size())
            weights = data[name]['weights']
            if weight_shape != weights.shape:
                logger.info("\tReshaping weight {} => {}"
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


def batch_to_images(batch):
    if len(batch.size()) == 4:
        return batch.cpu().data.numpy().reshape(
            batch.size(0), 3, batch.size(-2), batch.size(-1))\
            .transpose((0, 2, 3, 1))
    else:
        return batch.cpu().data.numpy().reshape(
            batch.size(0), batch.size(-2), batch.size(-1)).transpose((0, 1, 2))


def gradient_image(batch):
    grady = F.conv2d(batch, SOBEL_KERNEL_Y.cuda())
    gradx = F.conv2d(batch, SOBEL_KERNEL_X.cuda())
    return grady, gradx


def normalize_batch(batch):
    if torch.is_tensor(batch):
        mean = batch.mean()
        std = batch.std()
    else:
        mean = batch.mean().view(*(1 for _ in batch.size())).expand(batch.size())
        std = batch.std().view(*(1 for _ in batch.size())).expand(batch.size())
    batch = (batch - mean) / std
    return batch


def batch_frobenius_norm(batch):
    batch = batch.view(batch.size(0), batch.size(1), -1)
    return (batch ** 2).sum(dim=2).squeeze().sqrt()


def hist_kern_normal(x, mu, sigma):
    return 1.0 / (sigma * math.sqrt(2*math.pi)) * torch.exp(-(x - mu)**2 / (2*sigma**2))


def hist_kern_l1(batch, bin_val, span):
    k = 1 - torch.abs(batch - bin_val) / span
    k = (k >= 0).float() * k
    return k


def hist_kern_l2(batch, bin_val, span):
    return 1 - (batch - bin_val).pow(2)


def hist_kern_const(batch, bin_val, span):
    return ((batch - bin_val) <= span / 2) * ((batch - bin_val) >= -span / 2)


def batch_histogram(batch, bins=32, mask_batch=None,
                    kern_func=hist_kern_const):
    batch_size = (*batch.size()[:2], batch.size(2) * batch.size(3))
    hists = Variable(torch.zeros(batch.size(0), 3, bins).cuda())
    binvals = Variable(torch.linspace(0, 1, bins).cuda())
    # Expand values so we compute histogram in parallel.
    binvals = binvals.view(1, 1, 1, bins).expand(*batch_size, bins)
    batch = batch.view(*batch_size, 1).expand(*batch_size, bins)
    hist_responses = kern_func(batch, binvals).float()
    if mask_batch is not None:
        mb_size = (mask_batch.size(0), mask_batch.size(1), mask_batch.size(2) * mask_batch.size(3))
        mask_batch = mask_batch.view(*mb_size, 1).expand(*batch_size, bins)
        hist_responses = hist_responses * mask_batch
    hist = hist_responses.sum(dim=2)[:, :, 0, :]
    # L1 normalize.
    hist /= hist.sum(dim=2).expand(hist.size())
    return hist
