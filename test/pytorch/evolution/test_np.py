import unittest,numpy as np
import torch
from .np_f_b import NP
class TDD_NP(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.foo = 1
    def test_np(self):
        n=NP()
        self.assertEqual(n.x.shape,(64,1000))
        self.assertEqual(np.maximum([2, 3, 4], [1, 5, 2]).tolist(),[2, 5, 4])
        eye=np.maximum(np.eye(2), [0.5, 2]) # broadcasting
        self.assertEqual(eye.tolist(),[[ 1. ,  2. ],
            [ 0.5,  2. ]])
        m=np.maximum([np.nan, 0, np.nan], [0, np.nan, np.nan])
        self.assertTrue(all(list(map(lambda num:num!=num,m))))
        self.assertEqual(np.maximum(np.Inf, 1).tolist(),float('inf'))
        self.assertEqual(n.h.shape,(n.N,n.H))
        self.assertEqual(np.maximum([.5, 0, -.5], 0).tolist(), [.5, 0, 0])
    def test_ReLU(self):
        torch.manual_seed(2)
        m = torch.nn.ReLU()
        randn = torch.randn(2)
        m(randn)
        input = randn
        output = map(lambda item:item>=0, m(input))
        self.assertTrue(all(output))
        m = torch.nn.ReLU()
        input = randn.unsqueeze(0)
        output = torch.cat((m(input),m(-input)))
        self.assertEqual(output.tolist(),[np.maximum(randn,0).tolist(),np.maximum(-randn,0).tolist()])
if __name__ == '__main__':
    unittest.main()
                