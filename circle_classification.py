"""
圆形决策边界数据分类

参考文献：
1. Mitarai, Kosuke, et al. Quantum circuit learning. Physical Review A 98.3 (2018): 032309.
2. Farhi, Edward, and Hartmut Neven. Classification with quantum neural networks on near term processors. arXiv preprint arXiv:1802.06002 (2018).
3. Schuld, Maria, et al. Circuit-centric quantum classifiers. Physical Review A 101.3 (2020): 032308.
4. Schuld, Maria. Supervised quantum machine learning models are kernel methods. arXiv preprint arXiv:2101.11020 (2021).
"""


from matplotlib import pyplot as plt

from wuyue.register.quantumregister import QuantumRegister
from wuyue.register.classicalregister import ClassicalRegister
from wuyue.circuit.circuit import QuantumCircuit
from wuyue.element.gate import RY, RZ, CNOT
from wuyue_machine_learning.utils import numpy as np
from wuyue_machine_learning.optimizer import Adam
from wuyue_machine_learning.qnn import QNN
from wuyue_machine_learning.backend.observable import *


# 参数设置
train_num = 200  # 训练集大小
test_num = 100  # 测试集大小
boundary_gap = 0.5  # 决策边界的宽度
seed_data = 5  # 固定随机种子
qubits_num = 4  # 量子比特数量
batch_size = 20  # 训练batch大小
epochs = 20 # 训练轮数
np.random.seed(42)  # 设置随机种子用以初始化各种参数


# 用以可视化生成的数据集
def data_point_plot(data, label):
    num_samples, _ = np.shape(data)
    plt.figure(1)
    for i in range(num_samples):
        if label[i] == 0:
            plt.plot(data[i][0], data[i][1], color="g", marker="o")
        elif label[i] == 1:
            plt.plot(data[i][0], data[i][1], color="b", marker="o")
    plt.show()


# 圆形决策边界两分类数据集生成器
def circle_data_point_generator(train_num, test_num, boundary_gap):
    # 取前train_num个为训练集，后test_num个为测试集
    train_x, train_y = [], []
    num_samples, seed_para = 0, 0
    while num_samples < train_num + test_num:
        data_point = np.random.rand(2) * 2 - 1  # 生成[-1, 1]范围内二维向量

        # 如果数据点的模小于(0.7 - gap)，标为0
        if np.linalg.norm(data_point) < 0.7 - boundary_gap / 2:
            train_x.append(data_point)
            train_y.append(0.)
            num_samples += 1

        # 如果数据点的模大于(0.7 + gap)，标为1
        elif np.linalg.norm(data_point) > 0.7 + boundary_gap / 2:
            train_x.append(data_point)
            train_y.append(1.)
            num_samples += 1
        else:
            seed_para += 1

    train_x = np.array(train_x).astype("float64")
    train_y = np.array(train_y).astype("float64")

    return train_x[0:train_num], train_y[0:train_num], train_x[train_num:], train_y[train_num:]


# 生成自己的数据集
train_x, train_y, test_x, test_y = circle_data_point_generator(train_num, test_num, boundary_gap)

# 数据可视化调用
# 训练数据可视化
data_point_plot(train_x, train_y)
# 测试数据可视化
data_point_plot(test_x, test_y)

# 构建量子神经网络
class MyQNN(QNN):
    def forward(self, params, x):
        qubit = QuantumRegister(qubits_num)
        cbit = ClassicalRegister(qubits_num)
        circuit = QuantumCircuit(qubit, cbit)
        # 角度编码
        circuit.add(RY, qubit[0], paras=np.arcsin(x[:,0])).add(RZ, qubit[0],paras=np.arccos(x[:,0] ** 2))
        circuit.add(RY, qubit[1], paras=np.arcsin(x[:,1])).add(RZ, qubit[1],paras=np.arccos(x[:,1] ** 2))
        circuit.add(RY, qubit[2], paras=np.arcsin(x[:,0])).add(RZ, qubit[2],paras=np.arccos(x[:,0] ** 2))
        circuit.add(RY, qubit[3], paras=np.arcsin(x[:,1])).add(RZ, qubit[3],paras=np.arccos(x[:,1] ** 2))
        # 参数化量子线路
        circuit.add(RZ,qubit[0],paras=params[0])
        circuit.add(RY,qubit[0],paras=params[1])
        circuit.add(RZ,qubit[0],paras=params[2])
        circuit.add(RZ,qubit[1],paras=params[3])
        circuit.add(RY,qubit[1],paras=params[4])
        circuit.add(RZ,qubit[1],paras=params[5])
        circuit.add(RZ,qubit[2],paras=params[6])
        circuit.add(RY,qubit[2],paras=params[7])
        circuit.add(RZ,qubit[2],paras=params[8])
        circuit.add(RZ,qubit[3],paras=params[9])
        circuit.add(RY,qubit[3],paras=params[10])
        circuit.add(RZ,qubit[3],paras=params[11])
        circuit.add(CNOT,qubit[1],qubit[0])
        circuit.add(CNOT,qubit[2], qubit[1])
        circuit.add(CNOT,qubit[3], qubit[2])
        circuit.add(CNOT,qubit[0], qubit[3])
        circuit.add(RY,qubit[0],paras=params[12])
        circuit.add(RY,qubit[1],paras=params[13])
        circuit.add(RY,qubit[2],paras=params[14])
        circuit.add(RY,qubit[3],paras=params[15])

        return self.backend.expectation(circuit, PauliZ(0))
    # 预测
    def predict(self, params, x):
        y_preds = self.forward(params, x)
        y_preds = y_preds / 2 + 0.5 + params[-1]
        y_preds = [1 if i > 0.5 else 0 for i in y_preds]
        return y_preds
    # 损失函数
    def cost(self, params, x, y):
        y_ = self.forward(params, x) / 2 + 0.5 + params[-1]
        loss = (y_ - y) ** 2
        return np.sum(loss)
    # 计算准确率
    def score(self, params, x, y):
        y_preds = self.predict(params, x)
        return np.mean(y_preds == y)

# 初始化参数
params = np.random.rand(17)
params[-1] = 0.0

# 实例化量子神经网络
myqnn = MyQNN()

# 初始化优化器
adam = Adam(params, 0.01)

# 训练
train_losses=[]
test_losses=[]
for epochs in range(epochs):
    for itr in range(train_num//batch_size):
        gradient = myqnn.backward(params, train_x[itr*batch_size:(itr+1)*batch_size], \
                                  train_y[itr*batch_size:(itr+1)*batch_size])
        params = adam.update(gradient)
    loss_show = myqnn.cost(params, train_x, train_y)
    train_losses.append(loss_show)
    loss_sh = myqnn.cost(params, test_x, test_y)
    test_losses.append(loss_sh)
    train_acc = myqnn.score(params, train_x, train_y)
    test_acc = myqnn.score(params, test_x, test_y)
    print(f"epochs:{epochs},loss:{loss_show},train_acc:{train_acc},test_acc:{test_acc}")

print(train_losses)
print(test_losses)
plt.plot(train_losses,range(epochs+1),c='blue')
plt.plot(test_losses,range(epochs+1),c='red')
plt.show()

