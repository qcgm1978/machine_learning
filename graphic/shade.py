import numpy as np
import matplotlib.pyplot as plt

# x = [1, 2, 3, 4, 5]
x=np.linspace(-4,12,30)
# y1 = [1, 1, 2, 3, 5]
y1=np.random.normal(800,10,30)
# y2 = [0, 4, 2, 6, 8]
y2=np.linspace(1010,0,30)
# y3 = [1, 3, 5, 7, 9]

y = np.vstack([y1, y2])

labels = ["Fibonacci ", "Evens"]

fig, ax = plt.subplots()
ax.stackplot(x, y1, y2, labels=labels)
ax.legend(loc='upper left')
# plt.show()

fig, ax = plt.subplots()
ax.stackplot(x, y)
# plt.show()

plt.savefig('img/shade.png')