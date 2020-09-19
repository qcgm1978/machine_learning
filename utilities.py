import  os
def getPath(file,chdir=None):
    os.chdir(chdir)
    pre = os.getcwd()
    file = os.path.join(pre, file)
    return file