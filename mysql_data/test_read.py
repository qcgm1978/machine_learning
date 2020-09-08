import unittest
from mysqlOp import MysqlOp
from decorators_func import singleton 
class TDD_READ(unittest.TestCase):
    @singleton
    def setUp(self):
            val = [{'name':"John",'address': "Mountain 21"},{'name':'Joe','address':"Yellow Garden 2"},{'name':'Ben','address' :'Park Lane 38'},{'name':'John', 'address':'Highway 21'}]
            self.__class__.m = MysqlOp('customers',val,unique='address')
            return super().setUp()
    def test_table(self):
        self.m.showTables()
        self.assertIsInstance(self.m.mycursor, object)
        for k in self.m.mycursor:
            self.assertTrue(k in [('foo',),('customers',),('users',),('products',)])
    def test_select(self):
        s = self.m.select()
        self.assertEqual(len(s),self.m.mycursor.rowcount)
        s1 = self.m.select(limit=1)
        self.assertEqual(len(s1),1)
        self.assertEqual(len(s),self.m.getRowsCount())
        first=('John', 'Mountain 21')
        self.assertEqual(s[0][:2],first)
        s1 = self.m.selectColumns(['name', 'address'])
        self.assertEqual(s1[0],first)
        self.assertEqual(self.m.fetchOne()[:2], first)
    def test_filter(self):
        f=self.m.where({'address' :'Park Lane 38'})
        self.assertEqual(f[0][:2],('Ben', 'Park Lane 38'))
    def test_wild(self):
        w=self.m.wild({'address':'way'})
        self.assertEqual(w[0][:2],('John', 'Highway 21'))
    def test_escape(self):
        adr = ("Yellow Garden 2", )
        e=self.m.escape(adr)
        self.assertEqual(e[0][:2],('Joe', 'Yellow Garden 2'))
    def test_sort(self):
        s=self.m.sort('name')
        s1=self.m.sort('name',True)
        self.assertEqual(s[0][0],'Ben')
        self.assertEqual(s[-1][0],'John')
        self.assertEqual(s1[-1][0],'Ben')
        self.assertEqual(s1[-1][0],s[0][0])
        self.assertEqual(s1[0][0], s[-1][0])
#     def test_join(self):
#         MysqlOp('users',[
#             { 'id': 1, 'name': 'John', 'fav': 154},
#             { 'id': 2, 'name': 'Peter', 'fav': 154},
#             { 'id': 3, 'name': 'Amy', 'fav': 155},
#             { 'id': 4, 'name': 'Hannah', 'fav':0},
#             { 'id': 5, 'name': 'Michael', 'fav':0}
#         ])
#         MysqlOp('products',[
#             { 'id': 154, 'name': 'Chocolate Heaven' },
#             { 'id': 155, 'name': 'Tasty Lemons' },
#             { 'id': 156, 'name': 'Vanilla Dreams' }
#         ])
    
#         j=self.m.join()
#         j1=self.m.join(isLeft=True)
#         j2=self.m.join(isRight=True)
#         self.assertEqual(j,[('John', 'Chocolate Heaven'),
# ('Peter', 'Chocolate Heaven'),
# ('Amy', 'Tasty Lemons'),
# ('John', 'Chocolate Heaven'),
# ('Peter', 'Chocolate Heaven'),
# ('Amy', 'Tasty Lemons')])
#         self.assertEqual(j1,[('John', 'Chocolate Heaven'), ('Peter', 'Chocolate Heaven'), ('Amy', 'Tasty Lemons'), ('Hannah', None), ('Michael', None), ('John', 'Chocolate Heaven'), ('Peter', 'Chocolate Heaven'), ('Amy', 'Tasty Lemons'), ('Hannah', None), ('Michael', None)])
#         self.assertEqual(j2,[(None, 'Chocolate Heaven'), (None, 'Tasty Lemons'), (None, 'Vanilla Dreams'), (None, 'Chocolate Heaven'), (None, 'Tasty Lemons'), (None, 'Vanilla Dreams'), ('John', 'Chocolate Heaven'), ('Peter', 'Chocolate Heaven'), ('John', 'Chocolate Heaven'), ('Peter', 'Chocolate Heaven'), ('Amy', 'Tasty Lemons'), ('Amy', 'Tasty Lemons'), (None, 'Vanilla Dreams')])
if __name__ == '__main__':
    unittest.main()
