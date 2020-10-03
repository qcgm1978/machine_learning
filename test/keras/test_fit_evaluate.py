import unittest
# from utilities import getPath,parseNumber
import keras
from .fit_evaluate import FitEvaluate
class TDD_TEST_FIT_EVALUATE(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.topDense=10
        cls.epochs=2
        cls.k = FitEvaluate(cls.topDense,[keras.optimizers.RMSprop,keras.losses.SparseCategoricalCrossentropy,keras.metrics.SparseCategoricalAccuracy],2)
    def test_test_fit_evaluate(self):
        K=self.k.getKeras()
        self.assertRaises(TypeError,lambda: K['optimizers'])
        self.assertRaises(TypeError,lambda: K.optimizers['RMSprop'])
        self.assertIsInstance(self.k.model,K.Model)
        history=self.k.fit()
        # self.assertCountEqual(list(history.history.keys()),['loss','sparse_categorical_accuracy','val_loss','val_sparse_categorical_accuracy'])
        self.assertAlmostEqual(history.history['loss'][-1],.02,2)
        p=self.k.test()
        self.assertEqual(p.shape,(3,self.topDense))
if __name__ == '__main__':
    unittest.main()

                