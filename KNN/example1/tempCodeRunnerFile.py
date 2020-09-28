fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(feature[:, 0], feature[:, 1], 15.0*np.array(classLabel), 15.0*np.array(classLabel))
plt.show()