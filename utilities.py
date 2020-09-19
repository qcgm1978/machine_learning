import  os
def getPath(file,isUpperLevel=False):
    isUpperLevel and os.chdir('..')
    pre = os.getcwd()
    file = os.path.join(pre, file)
    return file