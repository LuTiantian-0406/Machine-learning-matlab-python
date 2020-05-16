import numpy as np
import os


def img2vector(filename):
    returnvect = np.zeros((1, 1024))
    data = open(filename)
    for i in range(32):
        lineStr = data.readline()
        for j in range(32):
            returnvect[0, 32 * i + j] = int(lineStr[j])
    return returnvect


# [returnvect] = img2vector('F:\\python\\KNN\\example2\\trainingDigits\\0_0.txt')
# print(returnvect[0:32])
# print(returnvect[32:64])


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
    sortedClassCount = sorted(classCount.items(),
                              key=lambda item: item[1],
                              reverse=True)
    return sortedClassCount[0][0]


def handwritingClassTest():
    #1. 导入训练数据
    hwLabels = []
    trainingFileList = os.listdir('F:\\python\\KNN\\example2\\trainingDigits')
    m = len(trainingFileList)
    trainingMat = np.zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i, :] = img2vector(
            'F:\\python\\KNN\\example2\\trainingDigits\\%s' % fileNameStr)

    # 2. 导入测试数据
    testFileList = os.listdir('F:\\python\\KNN\\example2\\testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('F:\\python\\KNN\\example2\\testDigits\\%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr))
        if (classifierResult != classNumStr):
            errorCount += 1.0
    print("\nthe total number of errors is: %d" % errorCount)
    print("\nthe total error rate is: %f" % (errorCount / float(mTest)))


handwritingClassTest()
