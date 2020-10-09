from __future__ import print_function
import torch
from torch import is_tensor
from torch.tensor import Tensor
class GetStarted(object):
    def __init__(self,loca=None):
        if loca:
            self.setSelf(loca)
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
        return GetStarted(locals())
    def setSelf(self,loca):
        keys = loca.keys()
        for key in list(keys):
            if key!='self':
                if hasattr(self,key):
                    raise ValueError('the key {0} already exists'.format(key))
                else:
                    exec('self.'+key+'=loca.get(key)')
        return self
    def Operations(self):
        y = torch.rand(5, 3)
        add=self.x6+y
        result = torch.empty(5, 3)
        torch.add(self.x6, y, out=result)
        result1=y.add_(self.x6)
        x7 = torch.randn(4, 4)
        y1 = x7.view(16)
        z =x7.view(-1, 8)  # the size -1 is inferred from other dimensions
        x8 = torch.randn(1)
        is_t3=is_tensor(x8)
        is_f3=torch.is_floating_point(x8)
        return self.setSelf(locals())
    def numpy_bridge(self):
        a = torch.ones(5)
        b = a.numpy()
        a.add_(1)
        return self.setSelf(locals())
    def from_numpy(self,a):
        return  torch.from_numpy(a)
    def CUDA_Tensors(self):
        # We will use ``torch.device`` objects to move tensors in and out of GPU
        if torch.cuda.is_available():
            device = torch.device("cuda")          # a CUDA device object
            y = torch.ones_like(x, device=device)  # directly create a tensor on GPU
            x = x.to(device)                       # or just use strings ``.to("cuda")``
            z = x + y
            print(z)
            print(z.to("cpu", torch.double))       # ``.to`` can also change dtype together!
        return self
    def autograd(self):
        x = torch.ones(2, 2, requires_grad=True)
        a1 = torch.randn(2, 2)
        a1 = ((a1 * 3) / (a1 - 1))
        y2 = x + 2
        z1 = y2 * y2 * 3
        out = z1.mean()
        out.backward()
        return self.setSelf(locals())
    # a tensor divided by itself such as: (((a + offset) * times)**power) / a *total_offset. e.g. ((a+2)*(a+2)*3)/4/a. offset=2, times=3,power=2,total_offset==1/4
    def divide_self(self,tensor,offset,times,power,total_offset):
        backprop_scalar=tensor.data[0][0]
        χι_plus_offset = (backprop_scalar+offset)*(power-1)
        coef = offset*total_offset*times
        αο_by_αχι=χι_plus_offset*coef
        return αο_by_αχι,'αο/αχι|χι={0} = {1}'.format(backprop_scalar,αο_by_αχι)
    def is_tensor(self,tens):
        return is_tensor(tens)