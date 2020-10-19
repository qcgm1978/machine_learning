import unittest
# from utilities import getPath,parseNumber,update_json
from dy_program import *


class TDD_DYNAMIC_PROGRAMMING(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.foo = 1

    def test_Dynamic_programming(self):
        d = DP()
        fib = d.fib
        self.assertEqual(fib(5), 5)
        self.assertEqual(d.two_count,1)
        self.assertEqual(fib(4) + fib(3), fib(5), 5)
        self.assertEqual((fib(3) + fib(2)) + (fib(2) + fib(1)), fib(5), 5)
        self.assertEqual(((fib(2) + fib(1)) + (fib(1) + fib(0))) +
                         ((fib(1) + fib(0)) + fib(1)), fib(5), 5)
        self.assertEqual((((fib(1) + fib(0)) + fib(1)) + (fib(1) +
                                                          fib(0))) + ((fib(1) + fib(0)) + fib(1)), fib(5), 5)

if __name__ == '__main__':
    unittest.main()
