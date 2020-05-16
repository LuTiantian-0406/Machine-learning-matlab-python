import matplotlib.pyplot as plt
import numpy as np


def file2matrix(filename):
    data = open(filename)
    numberOfLines = len(data.readlines())
    dataSet = np.zeros((numberOfLines, 2))
    Result = []
    index = 0
    data = open(filename)
    for line in data.readlines():
        line = line.strip()
        listfromLine = line.split('\t')
        dataSet[index, :] = listfromLine[0:2]
        Result.append(float(listfromLine[-1]))
        index += 1
    return dataSet, Result


[dataSet, Result] = file2matrix('F:\\python\\Linear_regression\\LeastSquare\\ex0.txt')
# print(dataSet)
# print(Result)

# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.scatter(dataSet[:, 1], Result, s=10)
# plt.show()


def Learst_square(Xal, yal):
    xMat = np.mat(Xal)
    yMat = np.mat(yal).T
    xTx = xMat.T * xMat
    if np.linalg.det(xTx) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    theta = xTx.I * (xMat.T*yMat)
    return theta


theta = Learst_square(dataSet, Result)
xMat = np.mat(dataSet)
fig = plt.figure()
ax = fig.add_subplot(111)
xCopy = xMat.copy()
xCopy.sort(0)
yHat = xCopy * theta
ax.scatter(dataSet[:, 1], Result, s=10)
ax.plot(xCopy[:, 1], yHat)
plt.show()
