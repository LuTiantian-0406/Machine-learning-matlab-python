import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def file2matrix(filename):
    data = open(filename)
    numberOfLines = len(data.readlines())
    returnMat = np.zeros((numberOfLines, 3))
    classLabelVector = []
    index = 0
    data = open(filename)
    for line in data.readlines():
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat, classLabelVector


def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = np.zeros(np.shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - np.tile(minVals, (m, 1))
    normDataSet = normDataSet / np.tile(ranges, (m, 1))
    return normDataSet, ranges, minVals


[feature, classLabels] = file2matrix('KNN\\example1\\DatingTestSet2.txt')
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(feature[:, 0], feature[:, 1], 15.0*np.array(classLabels), 15.0*np.array(classLabels))
plt.show()


def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.items(), key=lambda item: item[1], reverse=True)
    return sortedClassCount[0][0]


def datingClassTest():
    hoRatio = 0.1
    [feature, classLabels] = file2matrix('KNN\\example1\\DatingTestSet2.txt')
    [DataSet, ranges, minVals] = autoNorm(feature)
    m = DataSet.shape[0]
    numTestVecs = int(m * hoRatio)
    print('numTestVecs=', numTestVecs)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(DataSet[i, :], DataSet[numTestVecs:m, :], classLabels[numTestVecs:m], 3)
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, classLabels[i]))
        if (classifierResult != classLabels[i]):
            errorCount += 1.0
    print("the total error rate is: %f" % (errorCount / float(numTestVecs)))
    print(errorCount)


if __name__ == '__main__':
    datingClassTest()
