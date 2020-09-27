from mysqlOp import MysqlOp
import unittest
class TDD_UPDATE(unittest.TestCase):
    def setUp(self):
        self.m = MysqlOp('customers',{'name': 'VARCHAR(255)', 'address': 'VARCHAR(255)'})
        self.m.updateField({'field':'address','from':'Valley 345'  ,'to':'Canyon 123'})
        return super().setUp()
    def test_update(self):
        count=self.m.where({'address':'Canyon 123'})
        self.assertEqual(count,1)
        count=self.m.updateField({'field':'address','from': 'Canyon 123' ,'to':'Valley 345'})
        f=self.m.where({'address' :'Valley 345'})
        # self.assertEqual(count,1)
        # self.assertEqual(len(f),1)
if __name__ == '__main__':
    unittest.main()
