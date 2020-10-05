import time,math
t0=time.time()
import unittest
# from tensorflow._api.v2.data import Dataset
from utilities import getPath,parseNumber,dump_json
from .Intro_m import Intro
from .functional_API import *
class TDD_INTRO(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.intro = Intro([])
    # Prepare your data before training a model (by turning it into either NumPy arrays or tf.data.Dataset objects).
    def test_prepare_data(self):
        self.assertIsInstance(self.intro,self.intro.getNp().ndarray)
        D=self.intro.getDataset()
        dataset = D.from_tensor_slices([1, 2, 3])
        sequence=self.intro.getSequence()()
        self.assertIsInstance(dataset,D)
        self.assertTrue(self.intro.isValidData(self.intro))
        self.assertTrue(self.intro.isValidData(dataset))
        self.assertTrue(self.intro.isValidData(sequence))
    # Do data preprocessing, for instance feature normalization or vocabulary indexing.
    # Build a model that turns your data into useful predictions, using the Keras Functional API.
    def test_functional_API(self):
        intro = self.intro
        K = intro.get_keras()
        layer=K.layers.Dense(units=16)
        self.assertTrue(intro.isLayer(layer))
        inputs = K.Input(shape=(None, None, 3))
        self.assertEqual(inputs.shape.as_list(),[None,None,None,3])
        num_classes = 10
        outputs=get_outputs(inputs,num_classes=num_classes)
        self.assertTrue(isinstance(outputs,intro.get_tf().Tensor))
        model = K.Model(inputs=inputs, outputs=outputs)
        first = 64
        data = intro.getNp().random.randint(0, 256, size=(first, 200, 200, 3)).astype("float32")
        processed_data = model(data)
        self.assertEqual(processed_data.shape,(first,num_classes))
    def test_api(self):
        chain=Chain()
        chain.setInputs().setDense().setX().setX(chain.x).outputs().createModel().summary().plot(show_shapes=True).t_e_i().save().delete().load()
        self.assertEqual(chain.inputs.shape.as_list(),[None,784])
        intro = self.intro
        self.assertEqual(chain.inputs.dtype,intro.get_tf().float32)
        self.assertTrue(intro.is_tensor(chain.x))
        self.assertTrue(intro.is_model(chain.model))
        self.assertLess(chain.loss,.3)
        self.assertAlmostEqual(math.floor(chain.accuracy*10)/10,.9,1)
    def test_same_layers(self):
        chain=Chain()
        chain.encoder().decoder()
        self.assertEqual(chain.encoder_input.shape.as_list(),chain.decoder_output.shape.as_list())
    # Train your model with the built-in Keras fit() method, while being mindful of checkpointing, metrics monitoring, and fault tolerance.
    # Evaluate your model on a test data and how to use it for inference on new data.
    # Customize what fit() does, for instance to build a GAN.
    # Speed up training by leveraging multiple GPUs.
    # Refine your model through hyperparameter tuning.
p = '/Users/zhanghongliang/Documents/ml/test/keras/json-dump.json'
duration=time.time()-t0
data = dump_json(p,duration)
print( data[-2:],'\n',data[-1]-data[-2])
if __name__ == '__main__':
    unittest.main()
