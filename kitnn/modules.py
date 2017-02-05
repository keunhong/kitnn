from torch import nn
from torch.autograd import Variable
from torch.legacy.nn import SpatialCrossMapLRN


class SelectiveSequential(nn.Module):
    def __init__(self, modules_dict):
        super().__init__()
        for key, module in modules_dict.items():
            self.add_module(key, module)

    def forward(self, x, selection=list()):
        selection_dict = {}
        for name, module in self._modules.items():
            x = module(x)
            if name in selection:
                selection_dict[name] = x
        return [selection_dict[m] for m in selection]


class LRN(nn.Module):
    def __init__(self, size, alpha=1e-4, beta=0.75, k=1):
        super().__init__()
        self.lrn = SpatialCrossMapLRN(size, alpha, beta, k)

    def forward(self, x):
        self.lrn.clearState()
        return Variable(self.lrn.updateOutput(x.data))
