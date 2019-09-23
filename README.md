# kitnn
Various neural network models for PyTorch. Weights were converted from Caffe models using my fork of caffe-tensorflow.

## Weights

Weights are stored in an order consistent with PyTorch. Refer to kitnn.utils to see how to read them in. They can be downloaded here: https://github.com/keunhong/kitnn/releases/tag/0.1

### Material in Context 
```
Bell, Sean, et al. "Material recognition in the wild with the materials in context database." Proceedings of the IEEE conference on computer vision and pattern recognition. 2015.
```

* MINC AlexNet
* MINC VGG-16
* MINC GoogLeNet

#### Example Usage

```python
import skimage
import skimage.io
import numpy as np
from kitnn.models import minc
import toolbox.images


mincnet = minc.MincVGG()
mincnet.load_npy('path/to/model.npy')
mincnet = mincnet.cuda()

image = skimage.img_as_float32(skimage.io.imread('path/to/image.png'))

subst_map, _ = compute_substance_map(mincnet, image)
subst_map = toolbox.images.resize(
    subst_map, shape=IMAGE_SHAPE, order=0)
    

def compute_substance_map(mincnet, image, fg_mask=None):
    if fg_mask is None:
        fg_mask = np.ones(image.shape[:2], dtype=bool)
    processed_image = minc.preprocess_image(image)

    prob_maps, feat_dicts = minc.compute_probs_multiscale(
        processed_image, mincnet, use_cuda=True)
    prob_map_avg = minc.combine_probs(prob_maps, processed_image,
                                      remap=True, fg_mask=fg_mask)

    prob_map_crf = minc.compute_probs_crf(image, prob_map_avg)
    prob_map_crf = toolbox.images.resize(
        prob_map_crf, processed_image.shape[:2], order=3)
    subst_id_map = np.argmax(prob_map_crf, axis=-1)
    return subst_id_map, prob_map_crf
```

### VGG

* VGG-19 (normalized)
