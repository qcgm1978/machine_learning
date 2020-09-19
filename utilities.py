import  os
import re
def parseNumber(value, as_int=False):
    try:
        number = float(re.sub('[^.\-\d]', '', value))
        if as_int:
            return int(number + 0.5)
        else:
            return number
    except ValueError:
        return float('nan')  # or None if you wish
def getPath(file,chdir=None):
    chdir and os.chdir(chdir)
    pre = os.getcwd()
    file = os.path.join(pre, file)
    return file