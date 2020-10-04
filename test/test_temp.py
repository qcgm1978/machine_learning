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
    def test_sun(self):
        # p=σ*T**4 Stefan-Boltzmann law
        p=6e7
        σ=5.67e-8
        T=(p/σ)**(1/4)
        C=T+273.15
        self.assertAlmostEqual(T,5700,-3)
        self.assertAlmostEqual(C,5977,0)
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

                