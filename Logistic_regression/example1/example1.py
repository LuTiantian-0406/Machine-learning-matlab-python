import numpy as np
import matplotlib.pyplot as plt


def file2matrix(filename):
    data = open(filename)
    numberOfLines = len(data.readlines())
    dataset = np.zeros((numberOfLines, 2))
    classLabelVector = []
    index = 0
    data = open(filename)
    for line in data.readlines():
        line = line.strip()
        listFromLine = line.split('\t')
        dataset[index, :] = listFromLine[0:-1]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return dataset, classLabelVector


[dataset, classLabel] = file2matrix('F:\\python\\Logistic_regression\\example1\\testSet.txt')
# print(dataset)
# print(classLabelVector)
dataset = np.column_stack((np.ones((len(dataset), 1)), dataset))

# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.scatter(dataset[:, 0], dataset[:, 1], c=classLabel)
# plt.show()


def sigmoid(x):
    x = np.array(x)
    return 1 / (1 + np.exp(-x))


def compute_cost(theta, xal, yal):
    yal = np.array(yal)
    y1 = np.log(sigmoid(np.dot(theta, xal.T)))
    y2 = np.log(1 - sigmoid(np.dot(theta, xal.T)))
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


[theta, cost_list] = grad_desc(dataset, classLabel, np.random.rand(1, 3), 0.1, 500)
print(theta)
xMat = np.mat(dataset)
fig = plt.figure()
ax = fig.add_subplot(111)
xCopy = 0.1 * np.arange(-30, 30, 1)
yHat = (-theta[0, 0]-theta[0, 1]*xCopy)/theta[0, 2]
ax.scatter(dataset[:, 1], dataset[:, 2], c=classLabel)
ax.plot(xCopy, yHat)
plt.show()
