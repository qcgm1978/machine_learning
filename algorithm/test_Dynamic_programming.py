import unittest
# from utilities import getPath,parseNumber,update_json
from dy_program import *
class TDD_DYNAMIC_PROGRAMMING(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.foo = 1
    def test_Dynamic_programming(self):
        self.assertEqual(fib(5),5)
if __name__ == '__main__':
    unittest.main()
                