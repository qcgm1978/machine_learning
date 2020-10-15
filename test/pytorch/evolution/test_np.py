import unittest
import numpy as np
import math
from functools import reduce
import torch
import matplotlib.pyplot as plt
from utilities import saveAndShow
from .np_f_b import NP


class TDD_NP(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.foo = 1
    
    def test_np(self):
        n = NP()
        self.assertEqual(n.x.shape, (64, 1000))
        
        self.assertEqual(n.h.shape, (n.N, n.H))
        self.assertEqual(np.maximum([.5, 0, -.5], 0).tolist(), [.5, 0, 0])
        self.assertEqual((n.y_pred - n.y).shape, n.y.shape)
        self.assertEqual(n.x.ndim, n.w1.ndim, 2)
        self.assertEqual(n.y_pred.shape, (n.N,n.D_out),(64,10))
        self.assertEqual(n.h_relu.shape, (n.N,n.H),(64,100))
        self.assertEqual(n.h_relu.T.shape, (n.H,n.N),(100,64))
        self.assertEqual(n.grad_w2.shape, (n.H,n.D_out))
        self.assertEqual(n.grad_h_relu.shape, (n.N,n.H))
        self.assertEqual(n.grad_w1.shape, (n.D_in,n.H))
    def test_ReLU(self):
        torch.manual_seed(2)
        r = torch.nn.ReLU()
        randn = torch.randn(20)
        output = map(lambda item: item >= 0, r(randn))
        s = sorted(randn)
        plt.plot(s, r(torch.tensor(s)))
        _l = r(randn)
        s = list(map(lambda item: item.tolist(), randn))
        plt.scatter(s, _l)
        saveAndShow(plt)
        self.assertTrue(all(output))
        input = randn.unsqueeze(0)
        output = torch.cat((r(input), r(-input)))
        self.assertEqual(output.tolist(), [np.maximum(
            randn, 0).tolist(), np.maximum(-randn, 0).tolist()])

    

if __name__ == '__main__':
    unittest.main()
