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


[dataSet, Result] = file2matrix('F:\\python\\Linear_regression\\Gradient_descent\\ex0.txt')
# print(dataSet)
# print(Result)

# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.scatter(dataSet[:, 1], Result, s=10)
# plt.show()


def compute_cost(w, b, xal, yal):
    cost = sum((np.dot(w, dataSet.T) + b - Result)**2)
    return cost


def step_grad_desc(current_w, current_b, alpha, xal, yal):
    sum_grad_w = 0
    sum_grad_b = 0
    M = len(xal)
    sum_grad_w = sum(np.dot((np.dot(current_w, xal.T) + current_b - yal), xal))
    sum_grad_b = sum((np.dot(current_w, xal.T) + current_b - yal))
    # 用公式求当前梯度
    grad_w = 2/M * sum_grad_w
    grad_b = 2/M * sum_grad_b
    # 梯度下降，更新当前的w和b
    updated_w = current_w - alpha * grad_w
    updated_b = current_b - alpha * grad_b
    return updated_w, updated_b


def grad_desc(xal, yal, initial_w, initial_b, alpha, iternum):
    w = initial_w
    b = initial_b
    cost_list = []  # 定义一个list保存所有的损失函数值，用来显示下降过程。
    for i in range(iternum):
        cost_list.append(compute_cost(w, b, xal, yal))
        [w, b] = step_grad_desc(w, b, alpha, xal, yal)
    return [w, b, cost_list]


[w, b, cost_list] = grad_desc(dataSet, Result, [1, 2], 4, 0.1, 200)
print(w, b)
xMat = np.mat(dataSet)
fig = plt.figure()
ax = fig.add_subplot(111)
xCopy = 0.1 * np.arange(10)
yHat = xCopy * w[1] + w[0] + b
ax.scatter(dataSet[:, 1], Result, s=10)
ax.plot(xCopy, yHat)
plt.show()
