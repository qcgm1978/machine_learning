import matplotlib.pyplot as plt
XVals = ['10-Dec-18', '11-Dec-18']
YVals = [[0.88, 0.78, 0.92, 0.98, 0.91],[0.88, 0.78, 0.92, 0.98]]
X = [XVals[i] for i, data in enumerate(YVals) for j in range(len(data))]
Y = [val for data in YVals for val in data]
plt.scatter(X, Y)
plt.show()