import numpy as np


def loadDataSet(fileName):
    fr = open(fileName)
    stringArr = [line.strip().split(' ') for line in fr.readlines()]
    datArr = [line for line in stringArr]
    datArr = np.mat(datArr)
    datArr = datArr.astype(float)
    return datArr


def replaceNanWithMean(fileName):
    datMat = loadDataSet(fileName)
    numFeat = np.shape(datMat)[1]     # 数据的特征数量(也就是矩阵的列数)
    for i in range(numFeat):
        meanVal = np.mean(datMat[np.nonzero(~np.isnan(datMat[:, i].A))[0], i])
        datMat[np.nonzero(np.isnan(datMat[:, i].A))[0], i] = meanVal
    return datMat


dataMat = replaceNanWithMean('F:\\python\\PCA\\secom.data')
# print(data)


def pca(dataMat, topNfeat=9999999):
    meanVals = np.mean(dataMat, axis=0)
    meanRemoved = dataMat - meanVals
    # cov协方差=[(x1-x均值)*(y1-y均值)+(x2-x均值)*(y2-y均值)+...+(xn-x均值)*(yn-y均值)+]/(n-1)
    covMat = np.cov(meanRemoved, rowvar=0)
    eigVals, eigVects = np.linalg.eig(np.mat(covMat))    # eigVals为特征值， eigVects为特征向量
    eigValInd = np.argsort(eigVals)
    eigValInd = eigValInd[:-(topNfeat+1):-1]
    redEigVects = eigVects[:, eigValInd]
    lowDDataMat = meanRemoved * redEigVects
    reconMat = (lowDDataMat * redEigVects.T) + meanVals
    return lowDDataMat, reconMat, eigVals


[lowDDataMat, reconMat, eigVals] = pca(dataMat, 300)
# lowDDataMat表示低维特征空间
# reconMat表示原来的特征空间
# eigVals表示特征值
# erro表示重构值，重构值越大说明对原来数据影响越小
erro = sum(eigVals[0:5])/sum(eigVals)
print(erro)
