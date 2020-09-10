from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt
import numpy as np

x = np.arange(4)
y = x**2


def MyTicks(x, pos):
    'The two args are the value and tick position'
    if pos is not None:
        tick_locs=ax.yaxis.get_majorticklocs()      # Get the list of all tick locations
        str_tl=str(tick_locs).split()[1:-1]         # convert the numbers to list of strings
        p=max(len(i)-i.find('.')-1 for i in str_tl) # calculate the maximum number of non zero digit after "."
        p=max(1,p)                                  # make sure that at least one zero after the "." is displayed
        return "pos:{0}/x:{1:1.{2}f}".format(pos,x,p)

formatter = FuncFormatter(MyTicks)

fig, ax = plt.subplots()
ax.yaxis.set_major_formatter(formatter)
plt.plot(x,y,'--o')
plt.show()