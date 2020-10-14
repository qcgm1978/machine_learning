import unittest,numpy as np,math
from functools import reduce
import torch
import matplotlib.pyplot as plt
from .np_f_b import NP
from utilities import saveAndShow
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
        r = torch.nn.ReLU()
        randn = torch.randn(20)
        output = map(lambda item:item>=0, r(randn))
        s=sorted(randn)
        plt.plot(s,r(torch.tensor(s)))
        l = r(randn)
        s=list(map(lambda item:item.tolist(),randn))
        # print(NP.h_relu.T)
        plt.scatter(s,l)
        saveAndShow(plt)
        self.assertTrue(all(output))
        input = randn.unsqueeze(0)
        output = torch.cat((r(input),r(-input)))
        self.assertEqual(output.tolist(),[np.maximum(randn,0).tolist(),np.maximum(-randn,0).tolist()])
        # print(NP.loss)
    def test_T(self):
        m = np.matrix('[1, 2; 3, 4]')
        self.assertEqual(m.tolist(),[[1, 2],
        [3, 4]])
        self.assertEqual(m.getT().tolist(),[[1, 3],
        [2, 4]])
        self.assertEqual(m.getT().tolist(),m.T.tolist())
    def test_sum(self):
        l=[0.5, 1.5]
        self.assertEqual(np.sum(l),sum(l),2)
        l1 = [0.5, 0.7, 0.2, 1.5]
        self.assertEqual(np.sum(l1, dtype=np.int32),sum(map(lambda item:math.floor(item),l1)),1)
        l2 = [[0, 1], [0, 5]]
        s = sum(map(lambda item:sum(item),l2))
        self.assertEqual(np.sum(l2),s,6)
        self.assertEqual(np.sum(l2, axis=None),s,6)
        self.assertEqual(np.sum(l2, axis=0).tolist(),list(map(lambda item:sum(item),zip(*l2))),[0,6])
        self.assertEqual(np.sum([[0, 1], [0, 5]], axis=1).tolist(),list(map(lambda item:sum(item),l2)),[1,5])
        l3 = [[0, 1], [np.nan, 5]]
        self.assertEqual(np.sum(l3, where=[False, True], axis=1).tolist(),list(map(lambda item:reduce(lambda acc,it:acc+(0 if np.isnan(it) else it),item,0),l3)),[1,5])
        # If the accumulator is too small, overflow occurs:
        for item in (127,128,129):
            self.assertEqual(np.ones(item, dtype=np.int8).sum(dtype=np.int8),item if item<128 else item-128*2)
        # You can also start the sum with a value other than zero:
        ini=5
        l4 = [10,3]
        self.assertEqual(np.sum(l4, initial=ini),reduce(lambda acc,item:acc+item,l4,ini),18)
if __name__ == '__main__':
    unittest.main()
