from os.path import dirname
import sys,os

print(sys.argv)


def main():
    l = sys.argv[1].split("/")
    alist = [
        [
            l[-1].upper(),
            l[-1],
            ','+sys.argv[2] if len(sys.argv)>=3 else ''
        ]
        ]
    dirName = '/'.join(l[:-1])
    print(dirName,sys.argv[1],dirName+'/test_'+l[-1])
    # Create target directory & all intermediate directories if don't exists
    try:
        print("Directory " , dirName ,  " Created ")
        os.makedirs(dirName)
    except FileExistsError:
        print("Directory " , dirName ,  " already exists")
    for item in alist:
        f = open(dirName+'/test_'+l[-1] + ".py", "w+")
        # for i in range(10):
        #      f.write("This is line %d\r\n" % (i+1))
        content = """import os,time
t0=time.time()
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import unittest{2}
# from utilities import getPath,parseNumber
class TDD_{0}(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.foo = 1
    def test_{1}(self):
        self.assertEqual(self.foo,1)
    @classmethod
    def tearDownClass(cls):
        p = '/Users/zhanghongliang/Documents/ml/test/tf/json-update.json'
        duration=time.time()-t0
        data = update_json(p,duration)
        print( 'Previouse two duration: {0}'.format(data[-2:]),'\n','Difference: {0}'.format(data[-1]-data[-2]))
if __name__ == '__main__':
    unittest.main()

                """
        f.write(content.format(item[0], item[1],item[2],dirName))
        f.close()
    # create json if not exists
    file_name = dirName+'/json-update.json'
    f = open(file_name, 'a+')  # open file in append mode
    f.write('[0]')
    f.close()

if __name__ == "__main__":
    main()
