from __future__ import print_function
import torch
from torch import is_tensor
from torch.tensor import Tensor
class GetStarted(object):
    def Tensors(self):
        x1 = torch.rand(5, 3)
        x2 = torch.empty(5, 3)
        is_t=is_tensor(x1)
        is_t2=isinstance(x2,Tensor)
        x3 = torch.zeros(5, 3, dtype=torch.long)
        x4 = torch.tensor([5.5, 3])
        is_f=torch.is_floating_point(x4)
        x5 = x4.new_ones(5, 3, dtype=torch.double)      # new_* methods take in sizes
        is_f1=torch.is_floating_point(x5)

        x6 = torch.randn_like(x5, dtype=torch.float)    # override dtype!
        is_f2=torch.is_floating_point(x6)
        keys = locals().keys()
        for key in list(keys):
            exec('self.'+key+'=locals().get(key)')