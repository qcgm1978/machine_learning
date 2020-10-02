import unittest

from tensorflow.python.keras.backend import dtype
from utilities import getPath, parseNumber
from .humans import Dataset

from .custom_model import CustomModel
class TDD_TEST_KERAS(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.d = Dataset()
        cls.tf = cls.d.getTf()
        cls.np = cls.d.getNp()

    def test_test_keras(self):
        d = self.d
        dataset = d.dataset
        self.assertIsInstance(dataset, d.getType())
        l = d.getShapeDtype()
        self.assertEqual(len(l), 1)
        l0 = l[0]
        self.assertCountEqual(
            list(l0.keys()), ['data', 'labels'])
        self.assertEqual(l0['data'].dtype, self.np.float32)
        self.assertEqual(l0['data'].shape, [4, 200, 200, 3])
        self.assertEqual(l0['labels'].dtype, self.np.int32)
        self.assertEqual(l0['labels'].shape, [4])
        lt = d.getShapeDtype(d.datasetTxt)
        lt0 = lt[0]
        self.assertEqual(lt0['data'].dtype, self.tf.string)
        self.assertEqual(lt0['data'].shape, [2])
        self.assertEqual(lt0['labels'].dtype, self.np.int32)
        self.assertEqual(lt0['labels'].shape, [2])

    def test_preprocess_layer(self):
        d = self.d
        tf = self.tf
        l = [["This is the 1st sample."], ["And here's the 2nd sample."]]
        seq = d.str2sequence(l, isBigrams=False)
        self.assertIsInstance(seq, tf.Tensor)
        self.assertEqual(seq.dtype, tf.int64)
        self.assertEqual(seq.shape, (2, 5))
        val = tf.get_static_value(seq)
        self.assertIsInstance(val, self.np.ndarray)
        self.assertIsInstance(val[0], self.np.ndarray)
        self.assertEqual(list(val[0]), [4, 5, 2, 9, 3])
        self.assertEqual(list(val[1]), [7, 6, 2, 8, 3])
        seq = d.str2sequence(l)
        self.assertIsInstance(seq, tf.Tensor)
        self.assertEqual(seq.dtype, tf.float32)
        self.assertEqual(seq.shape, (2, 17))
        val = tf.get_static_value(seq)
        self.assertIsInstance(val, self.np.ndarray)
        self.assertIsInstance(val[0], self.np.ndarray)
        self.assertEqual(list(val[0]), [0, 1, 1, 1, 1,
                                        0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1])
        self.assertEqual(list(val[1]), [0, 1, 1, 0, 0,
                                        1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0])

    def test_normalize_features(self):
        size = (64, 200, 200, 3)
        data = self.d.setImgData(size).normalizeFeatures()
        self.assertAlmostEqual(self.np.var(data), 1.0000, 4)
        self.assertAlmostEqual(self.np.mean(data), -0.0000, 4)
        height = 150
        width = 160
        output_data = self.d.setImgData(
            size).rescaleCenterCrop(height=height, width=width)
        self.assertEqual(output_data.shape, (size[0], height, width, size[-1]))
        self.assertEqual(self.np.min(output_data), 0)
        self.assertEqual(self.np.max(output_data), 1)

    def test_functional_API(self):
        m = self.d.get16maps()
        self.assertIsInstance(m, self.tf.python.keras.layers.core.Dense)
        numClasses = 10
        outputs = self.d.layerTransform(num_classes=numClasses)
        self.assertEqual(outputs.shape.as_list(), [None, numClasses])
        model = self.d.getModel(outputs)
        self.assertIsInstance(model, self.d.getKeras().Model)
        size = (64, 200, 200, 3)
        data = self.np.random.randint(
            0, 256, size=size).astype("float32")
        processed_data = model(data)
        self.assertEqual(processed_data.shape, (size[0], numClasses))
        summary = self.d.summary()
        self.assertRegex(summary[0], 'Model: "functional_')

    def test_fit(self):
        toy = self.d.toy(Model=CustomModel)
        self.assertAlmostEqual(toy.history.history['loss'][0], 2, 0)
        self.assertLess(toy.history.history['acc'][0], .6)
        self.assertLess(toy.history.history['val_acc'][0], .3)
        self.assertLess(toy.history.history['val_loss'][0], 3)
        loss,acc=self.d.evaluate()
        self.assertAlmostEqual(loss, 2, 0)
        self.assertLess(acc, .4)
        self.assertEqual(self.d.predict().shape,(10000,10))
        self.assertEqual(len(self.d.predict()[0]),10)
if __name__ == '__main__':
    unittest.main()
