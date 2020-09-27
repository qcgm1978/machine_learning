import unittest,mysql
from mysqlOp import MysqlOp
class TDD_TEST_MYSQL(unittest.TestCase):
    def setUp(self):
        self.m = MysqlOp('customers',{'name': 'VARCHAR(255)', 'address': 'VARCHAR(255)'})
        return super().setUp()
    def test_key(self):
        self.assertRaises(mysql.connector.errors.ProgrammingError, lambda: self.m.createPrimaryKey())
        try:
            self.m.addPrimaryKey()
        except mysql.connector.errors.ProgrammingError:
            pass
    def test_unique(self):
        self.assertIsNone(self.m.unique('address'))
    def test_insert(self):
        val = ("John", "Highway 21")
        r=self.m.insertInto(val)
        self.assertEqual(self.m.mycursor.rowcount, 1 if r else -1)
        val = [
        ('Peter', 'Lowstreet 4'),
        ('Amy', 'Apple st 652'),
        ('Hannah', 'Mountain 21'),
        ('Michael', 'Valley 345'),
        ('Sandy', 'Ocean blvd 2'),
        ('Betty', 'Green Grass 1'),
        ('Richard', 'Sky st 331'),
        ('Susan', 'One way 98'),
        ('Vicky', 'Yellow Garden 2'),
        ('Ben', 'Park Lane 38'),
        ('William', 'Central st 954'),
        ('Chuck', 'Main Road 989'),
        ('Viola', 'Sideway 1633')
        ]
        r=self.m.executemany(val)
        self.assertEqual(self.m.mycursor.rowcount, len(val) if r else -1)
    def test_insert_id(self):
        sql = "INSERT INTO customers (name, address) VALUES (%s, %s)"
        val = ("Michelle", "Blue Village")
        self.m.insertInto( val)
        self.assertIsInstance( self.m.mycursor.lastrowid,int)
if __name__ == '__main__':
    unittest.main()
