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
        N, D_in, H, D_out=(10, 1000, 100, 10)
        # N, D_in, H, D_out=(64, 1000, 100, 10)
        cls.n = NP((N, D_in, H, D_out))
    def test_np(self):
        n=self.n
        N, D_in, H, D_out=(10, 1000, 100, 10)
        n.f_b()
        self.assertEqual(n.x.shape, (N, D_in))
        
        self.assertEqual(n.h.shape, (n.N, n.H))
        self.assertEqual(np.maximum([.5, 0, -.5], 0).tolist(), [.5, 0, 0])
        self.assertEqual((n.y_pred - n.y).shape, n.y.shape)
        self.assertEqual(n.x.ndim, n.w1.ndim, 2)
        self.assertEqual(n.y_pred.shape, (n.N,n.D_out),(N,D_out))
        self.assertLess(n._l[-1],1e-5)
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

    def test_chain_rule(self):
        n=self.n
        self.assertAlmostEqual(n.get_change_rate(10),699,0)
        self.assertEqual(n.cal_units(['°C','1/km','km','1/hour']).units,'kelvin / hour')
        self.assertEqual(n.chain_rule([-6.5,  2.5],['°C','1/km','km','1/hour'],'kelvin / hour' ), -16.25 )
if __name__ == '__main__':
    unittest.main()
