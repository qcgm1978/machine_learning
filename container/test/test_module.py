
import unittest
class TDD_TEST_MODULE(unittest.TestCase):
    def test_test_module(self):
        self.assertIsNone(__package__)
        self.assertEqual(__name__,'__main__')
if __name__ == '__main__':
    unittest.main()

                