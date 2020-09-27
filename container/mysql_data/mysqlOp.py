from functools import reduce
import mysql.connector, json
class MysqlOp(object):
    def __init__(self, table, cols=None, val=None, db="mydatabase", unique=None):
        self.table = table
        self.db = db
        self.mydb = mysql.connector.connect(
            host="localhost", user="root", password="test@2018", database=self.db
        )
        self.mycursor = self.mydb.cursor(buffered=True)
        if isinstance(cols, dict):
            self.fields = cols
            self.createTable(cols)
            self.addPrimaryKey()
            unique and self.unique(unique)
            self.fields = cols
            if isinstance(val, list):
                self.executemany(val)
        elif isinstance(cols, list):
            fields = cols[0]
            if isinstance(fields, dict):
                d = self.getFields(fields)
                self.fields = d
                self.createTable(d)
                self.addCols()
                self.addPrimaryKey()
                unique and self.unique(unique)
                self.executemany(cols)
    def __del__(self):
        try:
            self.mycursor.close()
            self.mydb.close()
        except ReferenceError:
            pass
        except AttributeError:
            pass
    def getFields(self, d):
        r = {}
        for key, val in d.items():
            if isinstance(val, int):
                r[key] = "INT"
            elif isinstance(val, str):
                r[key] = "VARCHAR(255)"
            elif isinstance(val, list):
                r[key] = "JSON"
        return r
    def showTables(self):
        self.mycursor.execute("SHOW TABLES")
    def addCols(self):
        for key, val in self.fields.items():
            sql = "SELECT COUNT(*) FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME = '{0}' AND COLUMN_NAME = '{1}'".format(
                self.table, key
            )
            self.mycursor.execute(sql)
            RESULT = self.mycursor.fetchall()
            if RESULT == [(0,)]:
                sql = "ALTER TABLE {0} ADD COLUMN {1} {2}".format(self.table, key, val)
                self.mycursor.execute(sql)
    def createTable(self, d):
        def getKey(acc, key):
            return acc + "{0} {1},".format(key, d[key])
        s = reduce(getKey, d, "")
        self.mycursor.execute(
            "CREATE TABLE IF NOT EXISTS {0} ({1})".format(self.table, s[:-1])
        )
    def createPrimaryKey(self):
        self.mycursor.execute(
            "CREATE TABLE customers (id INT AUTO_INCREMENT PRIMARY KEY, name VARCHAR(255), address VARCHAR(255))"
        )
    def addPrimaryKey(self):
        if "id" in self.fields:
            alter = """
                ALTER TABLE {0}  
                ADD PRIMARY KEY (id)
            """.format(
                self.table
            )
        else:
            alter = "ALTER TABLE {0} ADD id INT AUTO_INCREMENT PRIMARY KEY".format(
                self.table
            )
        try:
            self.mycursor.execute(alter)
            return True
        except mysql.connector.errors.ProgrammingError:
            return False
        finally:
            self.fields.pop("id", None)
    def insertInto(self, val):
        if isinstance(val,tuple) :
            return self.executemany(val, isMany=False) 
        else:
            s = ",".join(val.keys())
            placeholder = ", ".join(["%s"] * len(val))
            sql = "INSERT INTO {2} ({0}) VALUES  ({1})".format(s, placeholder, self.table)
            try:
                vals = list(map(self.valConvert, [val]))
                func = self.mycursor.execute
                func(sql, vals[0])
                self.mydb.commit()
                return self.mycursor.rowcount
            except mysql.connector.errors.IntegrityError:
                return 0
    def valConvert(self, item):
        if isinstance(item, dict):
            val = item.values()
            l = list(val)[0]
            if isinstance(l, list):
                s = json.dumps(l)
                ret = (s,)
            else:
                ret = tuple(val)
        else:
            ret = item
        return ret
    def executemany(self, val, isMany=True):
        def pop(item):
            isinstance(item, dict) and item.pop("id", None)
            return item
        if isMany:
            val = list(map(pop, val))
        else:
            val = pop(val)
        fields = self.fields
        s = ",".join(fields.keys())
        placeholder = ", ".join(["%s"] * len(fields))
        sql = "INSERT INTO {2} ({0}) VALUES  ({1})".format(s, placeholder, self.table)
        try:
            vals = list(map(self.valConvert, val))
            func = self.mycursor.executemany if isMany else self.mycursor.execute
            func(sql, vals)
            self.mydb.commit()
            return 1
        except mysql.connector.errors.IntegrityError:
            return 0
    def select(self, limit=None, offset=None):
        o = "OFFSET " + str(offset) if offset else ""
        s = "LIMIT " + str(limit) if limit else ""
        self.mycursor.execute("SELECT * FROM {0} {1} {2}".format(self.table, s, o))
        myresult = self.mycursor.fetchall()
        return myresult
    def getRowsCount(self):
        self.select()
        return self.mycursor.rowcount
    def selectColumns(self, cols):
        s = reduce(lambda acc, item: acc + ", " + item, cols)
        self.mycursor.execute("SELECT {0} FROM customers".format(s))
        return self.mycursor.fetchall()
    def fetchOne(self):
        self.mycursor.execute("SELECT * FROM {0}".format(self.table))
        return self.mycursor.fetchone()
    def where(self, d, offset=0):
        s = reduce(lambda acc, key: "{0} = '{1}'".format(key, d[key]), d, "")
        sql = "SELECT * FROM {1} WHERE {0}".format(s,self.table)
        self.mycursor.execute(sql)
        return self.mycursor.rowcount
    def ignore(self, colName):
        sql = """delete from {0} 
where ID in (
select * from (select ID from {0} where ID not in (
select max(ID) from {0} group by {1}
) ORDER BY ID
) AS p
)""".format(
            self.table, colName
        )
        self.mycursor.execute(sql)
    def unique(self, colName):
        if isinstance(colName, str):
            colName = [colName]
        for i in colName:
            self.ignore(i)
            sql = "ALTER  TABLE {0} ADD UNIQUE unique_index  ({1} (8000))".format(
                self.table, i
            )
            try:
                self.mycursor.execute(sql)
            except mysql.connector.errors.ProgrammingError:
                return None
    def wild(self, d):
        s = self.getStr(d, "LIKE", True)
        sql = "SELECT * FROM {0} WHERE {1}".format(self.table, s)
        self.mycursor.execute(sql)
        return self.mycursor.fetchall()
    def getStr(self, d, relation=" ", isWild=False):
        s = "%" if isWild else ""
        return reduce(
            lambda acc, key: "{0} {2} '{3}{1}{3}'".format(key, d[key], relation, s),
            d,
            "",
        )
    def escape(self, adr, field="address"):
        sql = "SELECT * FROM {0} WHERE {1} = %s".format(self.table, field)
        self.mycursor.execute(sql, adr)
        return self.mycursor.fetchall()
    def sort(self, field, isDESC=False):
        sql = "SELECT * FROM {0} ORDER BY {1} {2}".format(
            self.table, field, "DESC" if isDESC else ""
        )
        self.mycursor.execute(sql)
        return self.mycursor.fetchall()
    def delete(self, d):
        s = reduce(lambda acc, key: "{0} = %s".format(key), d, "")
        sql = "DELETE FROM {0} WHERE {1}".format(self.table, s)
        self.mycursor.execute(sql, tuple(d.values()))
        self.mydb.commit()
        return self.mycursor.rowcount
    def dropTable(self, l=None):
        if l is None:
            l = [self.table]
        for val in l:
            sql = "DROP TABLE  IF EXISTS  {0}".format(val)
            self.mycursor.execute(sql)
        return True
    def updateField(self, d):
        fro = d["from"] if "from" in d else self.list
        if "field" in d:
            d1 = {d["field"]: fro}
        count = self.where(d1)
        to = d["to"]
        if count:
            sql = "UPDATE {0} SET {1} = %s WHERE {1} = %s".format(
                self.table, d["field"]
            )
            if isinstance(to, list):
                to = json.dumps(to)
                fro = json.dumps(fro)
            self.mycursor.execute(sql, (to, fro))
            self.mydb.commit()
        else:
            self.insertInto({d["field"]: to})
        if isinstance(to, list):
            self.list=to
        return self.mycursor.rowcount
    def join(self, isLeft=False, isRight=False):
        if isLeft:
            sql = "SELECT \
  users.name AS user, \
  products.name AS favorite \
  FROM users \
  LEFT JOIN products ON users.fav = products.id"
        elif isRight:
            sql = "SELECT \
  users.name AS user, \
  products.name AS favorite \
  FROM users \
  RIGHT JOIN products ON users.fav = products.id"
        else:
            sql = "SELECT \
  users.name AS user, \
  products.name AS favorite \
  FROM users \
  INNER JOIN products ON users.fav = products.id"
        self.mycursor.execute(sql)
        return self.mycursor.fetchall()
