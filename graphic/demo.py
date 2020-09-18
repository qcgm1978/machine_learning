import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import math

mu = 0
variance = 1
limit=4
sigma = math.sqrt(variance)
x = np.linspace(mu - limit*sigma, mu + limit*sigma, 100)
x1 = np.linspace(-limit, limit, 100)
y = stats.norm.pdf(x, mu, sigma)
plt.plot(x, y,c='black',linewidth=.5)
plt.fill_between(x, y, color='#0080CF', alpha=0.3)

plt.savefig("/Users/zhanghongliang/Documents/machine_learning/img/demo.png")
