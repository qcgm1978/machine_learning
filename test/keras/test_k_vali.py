import unittest
from utilities import getPath,parseNumber
class TDD_TEST_K_VALI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.foo = 1
    def test_test_k_vali(self):
        # todo Automatically setting apart a validation holdout set https://keras.io/guides/training_with_built_in_methods/
        self.assertEqual(self.foo,1)
if __name__ == '__main__':
    unittest.main()

                