import matplotlib.pyplot as plt
fig, ax = plt.subplots()
# We need to draw the canvas, otherwise the labels won't be positioned and 
# won't have values yet.
fig.canvas.draw()
labels = [item.get_text() for item in ax.get_xticklabels()]
# labels[1] = 'Testing'
ax.set_xticklabels(labels)
plt.show()