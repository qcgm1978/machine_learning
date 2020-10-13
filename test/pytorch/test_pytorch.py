import unittest,numpy as np
from .what_is_pytorch import GetStarted
from .nn import Net
from .classifier import Classifier
class TDD_PYTORCH(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        original=GetStarted()
        cls.gs=original.Tensors().Operations().numpy_bridge().CUDA_Tensors().autograd().vector_Jacobian_product().detach()
    def test_syntactic_sugar(self):
        l=[1,2,3]
        l[:2]=[4 for _ in range(2)]
        self.assertEqual(l,[4,4,3])
    def test_pytoch(self):
        gs=self.gs
        self.assertTrue(gs.is_t)
        self.assertTrue(gs.is_t2)
        self.assertEqual(gs.x3.shape,(5,3))
        flat_list = [not item for sublist in gs.x3 for item in sublist]
        self.assertTrue(all(flat_list))
        self.assertTrue(gs.is_f)
        self.assertTrue(gs.is_f1)
        self.assertTrue(gs.is_f2)
        self.assertEqual(gs.x5.shape,(5,3))
        self.assertEqual(gs.x6.shape,(5,3))
        self.assertEqual(list(gs.x6.size()),[5,3])
    def test_operations(self):
        gs=self.gs
        flat=[(item1,item2) for sub1,sub2 in zip(gs.add,gs.x6) for item1,item2 in zip(sub1,sub2)]
        greater=map(lambda item:item[0]>item[1],flat)
        self.assertTrue(all(greater))
        flat=[(item1,item2,item3) for sub1,sub2,sub3 in zip(gs.add,gs.result,gs.result1) for item1,item2,item3 in zip(sub1,sub2,sub3)]
        equal=map(lambda item:item[0]==item[1]==item[2],flat)
        self.assertTrue(all(equal))
        self.assertEqual(list(gs.x6[:,1].shape),[5])
        self.assertEqual((gs.x7.size(), gs.y1.size(), gs.z.size()),((4,4),(16,),(2,8)))
        self.assertTrue(gs.is_t3 and gs.is_f3 and isinstance(gs.x8.item(),float))
    def test_numpy_bridge(self):
        gs=self.gs
        self.assertIsInstance(gs.b,np.ndarray)
        self.assertEqual(list(gs.a),[2,2,2,2,2])
        self.assertEqual(list(gs.b),[2,2,2,2,2])
        a = np.ones(5)
        b=gs.from_numpy(a)
        np.add(a, 1, out=a)
        self.assertEqual(list(a),[2,2,2,2,2])
        self.assertEqual(list(b),[2,2,2,2,2])
    def test_autograd(self):
        gs=self.gs
        self.assertTrue(gs.is_tensor(gs.x))
        y=gs.x+2
        self.assertTrue(hasattr(y,'grad_fn'))
        z=y*y*3
        out = z.mean()
        data=map(lambda item:list(item),z.data)
        self.assertEqual(list(data),[[27,27],[27,27]])
        self.assertEqual(out,27)
        g=map(lambda item:list(item),gs.x.grad)
        self.assertEqual(list(g),[[4.5,4.5],[4.5,4.5]])
        self.assertEqual(list(map(lambda item:list(item),gs.x.data)),[[1,1],[1,1]])
        ratio_equality=gs.divide_self(gs.x,2,3,2,1/4)
        self.assertEqual(ratio_equality[0],4.5)
        self.assertEqual(ratio_equality[1], 'αο/αχι|χι=1.0 = 4.5')
    def test_Jacobian_matrix(self):
        gs = self.gs
        y=gs.vector_Jacobian_product_y
        self.assertTrue(gs.is_tensor(y))
        self.assertTrue(callable(y.grad_fn))
        self.assertFalse(gs.vector_Jacobian_product_z)
        self.assertFalse(gs.detach_z)
        self.assertTrue(gs.detach_z1)
    def test_netural_networks(self):
        net = Net()
        self.assertEqual(net.__class__,Net)
        self.assertIsInstance(net,Net)
        net.get_parameters().input_32().zero_backprops().loss().backprop().update()
        self.assertEqual(net.len,10)
        self.assertEqual(list(net.size),[6,1,3,3])
        self.assertTrue(callable(net.out.grad_fn))
        is_tensor=net.torch().is_tensor
        self.assertTrue(is_tensor(net.loss))
        self.assertEqual(str(net.acc.__class__).split('\'')[1],'AccumulateGrad')
        self.assertTrue(is_tensor(net.g))
        self.assertFalse(any(net.g_befor_backward))
        self.assertTrue(all(net.g))
        self.assertFalse(any(net.g_after_zero))
    def test_training_classifier(self):
        net =Net(3)
        size=net.interpret_size('3x32x32')
        self.assertEqual(size,{'channel':3,'width':32,'height':32})
        c=Classifier(net)
        # c.DataLoader().show_random_imgs()
        # c.DataLoader().feed_inputs(False).save().display_test_img()
        # net.load(c.get_path()).predict(c.images, c.classes)
        # net.DataLoader().load_dict(c.get_path()).perform_dataset().class_difference(c.classes)
        net.CUDA()
if __name__ == '__main__':
    unittest.main()