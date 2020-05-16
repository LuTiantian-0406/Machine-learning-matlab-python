import numpy as np
# import matplotlib
# import matplotlib.pyplot as plt


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


# [feature, classLabel] = file2matrix('F:\\python\\KNN\\DatingTestSet2.txt')
# print(returnMat)


def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = np.zeros(np.shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - np.tile(minVals, (m, 1))
    normDataSet = normDataSet / np.tile(ranges, (m, 1))
    return normDataSet, ranges, minVals


# [DataSet, ranges, minVals] = autoNorm(feature)


# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.scatter(feature[:, 0], feature[:, 1], 15.0*np.array(classLabel), 15.0*np.array(classLabel))
# plt.show()


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


def anti_autoNorm(value, ranges, minVals):
    inX = (value + minVals) * ranges
    return inX


def Knnclassify(data, filename):
    [feature, classLabel] = file2matrix(filename)
    [DataSet, ranges, minVals] = autoNorm(feature)
    Inx = anti_autoNorm(data, ranges, minVals)
    predict = classify0(Inx, DataSet, classLabel, 20)
    return predict


handsome_man = [25000, 5, 1]
pred = Knnclassify(handsome_man, 'F:\\python\\KNN\\DatingTestSet2.txt')
print(pred)
