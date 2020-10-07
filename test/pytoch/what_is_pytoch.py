from __future__ import print_function
import torch
from torch import is_tensor
from torch.tensor import Tensor
x1 = torch.rand(5, 3)
x2 = torch.empty(5, 3)
is_t=is_tensor(x1)
is_t2=isinstance(x2,Tensor)