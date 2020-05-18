import numpy as np
# import matplotlib.pyplot as plt


def file2matrix(filename):
    data = open(filename)
    numberOfLines = len(data.readlines())
    dataset = np.zeros((numberOfLines, 21))
    classLabelVector = []
    index = 0
    data = open(filename)
    for line in data.readlines():
        line = line.strip()
        listFromLine = line.split('\t')
        dataset[index, :] = listFromLine[0:-1]
        classLabelVector.append(float(listFromLine[-1]))
        index += 1
    return dataset, classLabelVector


[dataset, classLabel] = file2matrix('F:\\python\\Logistic_regression\\example2\\horseColicTraining.txt')
# print(dataset)
# print(classLabel)
dataset = np.column_stack((np.ones((len(dataset), 1)), dataset))


# # fig = plt.figure()
# # ax = fig.add_subplot(111)
# # ax.scatter(dataset[:, 0], dataset[:, 1], c=classLabel)
# # plt.show()


def sigmoid(x):
    x = np.array(x)
    m = np.size(x, 0)
    n = np.size(x, 1)
    for i in range(m):
        for j in range(n):
            if x[i, j] >= 0:      # 对sigmoid函数的优化，避免了出现极大的数据溢出
                x[i, j] = 1.0/(1+np.exp(-x[i, j]))
            else:
                x[i, j] = np.exp(x[i, j])/(1+np.exp(x[i, j]))
    return x


def compute_cost(theta, xal, yal):
    yal = np.array(yal)
    y1 = np.log(sigmoid(np.dot(theta, xal.T)) + 1e-5)
    y2 = np.log(1 - sigmoid(np.dot(theta, xal.T)) + 1e-5)
    cost = np.sum(y1 * yal + y2.T * (1-yal))
    return cost


def step_grad_desc(current_theta, alpha, xal, yal):
    sum_grad_theta = 0
    M = len(xal)
    sum_grad_theta = np.dot((sigmoid(np.dot(current_theta, xal.T)) - yal), xal)
    # 用公式求当前梯度
    grad_theta = 2/M * sum_grad_theta
    # 梯度下降，更新当前的theta
    updated_theta = current_theta - alpha * grad_theta
    return updated_theta


def grad_desc(xal, yal, initial_theta, alpha, iternum):
    theta = initial_theta
    cost_list = []  # 定义一个list保存所有的损失函数值，用来显示下降过程。
    for i in range(iternum):
        cost_list.append(compute_cost(theta, xal, yal))
        theta = step_grad_desc(theta, alpha, xal, yal)
    return [theta, cost_list]


[theta, cost_list] = grad_desc(dataset, classLabel, np.random.rand(1, 22), 0.1, 100)
[testset, testclassLabel] = file2matrix('F:\\python\\Logistic_regression\\example2\\horseColicTest.txt')
mTest = len(testset)
testset = np.column_stack((np.ones((len(testset), 1)), testset))
# print(theta)
pre = sigmoid(np.dot(theta, testset.T))
classifierResult = []
errorCount = 0
for i in range(np.size(pre, 1)):
    if pre[0, i] > 0.5:
        Result = 1.0
    else:
        Result = 0.0
    classifierResult.append(Result)
    print("the classifier came back with: %d, the real answer is: %d" % (Result, testclassLabel[i]))
    if (Result != testclassLabel[i]):
        errorCount += 1.0
print("\nthe total number of errors is: %d" % errorCount)
print("\nthe total error rate is: %f" % (errorCount / float(mTest)))
