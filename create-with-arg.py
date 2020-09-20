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
    # Create target directory & all intermediate directories if don't exists
    try:
        print("Directory " , dirName ,  " Created ")
        os.makedirs(dirName)
    except FileExistsError:
        print("Directory " , dirName ,  " already exists")
    for item in alist:
        f = open(sys.argv[1] + ".py", "w+")
        # for i in range(10):
        #      f.write("This is line %d\r\n" % (i+1))
        content = """import unittest{2}
from graphic.plot import Plot
from mysql_data.decorators_func import singleton
from utilities import getPath,parseNumber
class TDD_{0}(unittest.TestCase):
    @singleton
    def setUp(self):
        class PlotAI(Plot):
            def __init__(self,arg=None):
                Plot.__init__(self)
        self.__class__.p = Plot()
    def test_{1}(self):
        self.assertEqual(True,1)
if __name__ == '__main__':
    unittest.main()

                """
        f.write(content.format(item[0], item[1],item[2]))
        f.close()


if __name__ == "__main__":
    main()
