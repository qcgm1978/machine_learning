import numpy as np
class DoNumpy(object):
    def poly1d(self, x, y):
        return np.poly1d(np.polyfit(x, y, 4))    