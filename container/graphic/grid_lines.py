import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.collections as coll
wid = 10
hei = 10
nrows = 400
ncols = 230
inbetween = 10
xx = np.arange(0, ncols, (wid))
yy = np.arange(0, nrows, (hei))
fig = plt.figure()
ax = plt.subplot(111, aspect='equal')
pat = []
for ind,xi in enumerate(xx):
    for index,yi in enumerate(yy):
        sq = patches.Rectangle((xi, yi), wid, hei, fill=True,color='white' if (index+ind)%2 else 'gray')
        ax.add_patch(sq)
ax.relim()
ax.autoscale_view()
# plt.axis('off')
# plt.savefig('test.png', dpi=90)
plt.savefig("img/SD.png")
# plt.show()