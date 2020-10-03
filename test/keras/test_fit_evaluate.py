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
        self.assertCountEqual(list(history.history.keys()),['loss','categorical_true_positives'])
        self.assertAlmostEqual(history.history['loss'][-1],.04,2)
        p=self.k.test()
        self.assertEqual(p.shape,(3,self.topDense))
        history=self.k.getHistory()
        self.assertAlmostEqual(history.history['loss'][0],2.5,1)
        history=self.k.handle_matrics()
        self.assertCountEqual(list(history.history.keys()),['loss','std_of_activation'])
        history=self.k.handle_matrics_function()
        self.assertCountEqual(list(history.history.keys()),['loss','std_of_activation'])
        history=self.k.compile_no_loss()
        self.assertCountEqual(list(history.history.keys()),['loss','accuracy'])
if __name__ == '__main__':
    unittest.main()

                