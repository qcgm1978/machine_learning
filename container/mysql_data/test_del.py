from mysqlOp import MysqlOp
import unittest
class TDD_DEL(unittest.TestCase):
    def setUp(self):
        val = [{'name':"John",'address': "Mountain 21"},{'name':'Joe','address':"Yellow Garden 2"}]
        self.m = MysqlOp('foo',val,unique='address')
        self.m1 = MysqlOp('data',val,db='machine_learning')
        return super().setUp()
    def test_del(self):
        ret=self.m.delete({'address' : 'Mountain 21'})
        self.assertEqual(ret,1 )
        ret1=self.m.delete({'address' : "Yellow Garden 2"})
        self.assertEqual(ret1,1 )
    def test_drop_table(self):
        # self.assertTrue(self.m.dropTable(['customers','foo','users','products']))
        pass
        # self.assertTrue(self.m1.dropTable(['data']))
if __name__ == '__main__':
    unittest.main()
