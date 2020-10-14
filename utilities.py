import  os,traceback
import re,math
import json
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
def update_json(p,duration):
    with open(p, "r") as jsonFile:
        try:
            data = json.load(jsonFile)
        except json.decoder.JSONDecodeError:
            data=[]
    data.append(round(duration))
    with open(p, "w") as jsonFile:
        json.dump(data, jsonFile)
    return data
def truncate(n,fixedTo=0):
    return math.floor(n * 10 ** fixedTo) / 10**fixedTo
def get_method_name():
        name=traceback.extract_stack(None, 2)[0][2]
        return name
def saveAndShow(plt,name='demo',enableShow=False):
        file='img/{0}.png'.format(name)
        plt.savefig(file)
        enableShow and plt.show()