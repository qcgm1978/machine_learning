import unittest

from tensorflow._api.v2.data import Dataset
# from utilities import getPath,parseNumber
import numpy as np
from .Intro_m import Intro
class TDD_INTRO(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.intro = Intro([])
    def test_intro(self):
        self.assertIsInstance(self.intro,np.ndarray)
        dataset = Dataset.from_tensor_slices([1, 2, 3])
        self.assertIsInstance(dataset,Dataset)
        self.assertTrue(self.intro.isValidData(self.intro))
        self.assertTrue(self.intro.isValidData(dataset))
if __name__ == '__main__':
    unittest.main()

                