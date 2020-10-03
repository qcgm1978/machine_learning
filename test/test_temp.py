import unittest,re
# from utilities import getPath,parseNumber
class TDD_TEST_TEMP(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.foo = 1
    def test_temp(self):
        s="<class 'tensorflow.python.keras.optimizer_v2.rmsprop.RMSprop'>"
        g = self.conver2str(s)
        self.assertEqual(g,'rmsprop')
        s="<class 'tensorflow.python.keras.losses.SparseCategoricalCrossentropy'>"
        g=self.conver2str(s)
        self.assertEqual(g,'sparse_categorical_crossentropy')

    def conver2str(self, s):
        def addUnderscore(matchobj):
            s=matchobj.group(0)
            if re.search(r'[A-Z]',s): 
                return '_'+s.lower()
            else: 
                return s
        g=re.search(r'[A-Z].*(?=.>)',s).group()
        g=re.sub(r'[A-Z]+[a-z]+',addUnderscore,g)
        g=''.join(list(g))[1:]
        return g
if __name__ == '__main__':
    unittest.main()

                