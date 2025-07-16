from matplotlib import pyplot as plt

from wuyue.register.quantumregister import QuantumRegister
from wuyue.register.classicalregister import ClassicalRegister
from wuyue.circuit.circuit import QuantumCircuit
from wuyue.element.gate import RY, RZ, CNOT,RX
from wuyue_machine_learning.utils import numpy as np
from wuyue_machine_learning.optimizer import Adam
from wuyue_machine_learning.qnn import QNN
from wuyue_machine_learning.backend.observable import *

# 构建量子神经网络
class MyQNN(QNN):
    def __init__(self,qubits_num):
        super().__init__()
        # self.f_cluster=f_cluster
        self.qubits_num=qubits_num  # 量子比特数量
    def forward(self, params, x):
        qubit = QuantumRegister(self.qubits_num)
        cbit = ClassicalRegister(self.qubits_num)
        circuit = QuantumCircuit(qubit, cbit)

        # 角度编码
        for i in range(self.qubits_num):
            # for value in self.f_cluster.values():
            #     for feature in value:
            #         temp=x[feature].to_numpy()
            #         circuit.add(RY, qubit[i], paras=temp * np.pi)
            # temp = x.to_numpy()
            # print(x)
            circuit.add(RY, qubit[i], paras=x[:, i] * np.pi)
            # circuit.add(RY, qubit[i], paras=np.arcsin(x[:,i])).add(RZ, qubit[i],paras=np.arccos(x[:,i] ** 2))

        # 参数化量子线路
        index=0
        #Z-Y-Z
        for i in range(self.qubits_num):
            circuit.add(RZ,qubit[i],paras=params[index])
            index += 1
            circuit.add(RY, qubit[i], paras=params[index])
            index += 1
            circuit.add(RZ, qubit[i], paras=params[index])
            index += 1
        # Z-Y-Z-CNOT
        # for i in range(self.qubits_num-1):
        #     circuit.add(CNOT, qubit[i], qubit[i+1])
        # circuit.add(CNOT,qubit[self.qubits_num-1], qubit[0])
        # Z-Y-Z-CNOT-Y
        for i in range(self.qubits_num):
            circuit.add(RY,qubit[i],paras=params[index])
            index += 1
        return self.backend.expectation(circuit, PauliZ(0))

        # 结构化参数化量子线路（多层）
        # param_idx = 0
        # for layer in range(3):
        #     # 参数化旋转层
        #     for i in range(self.qubits_num):
        #         circuit.add(RZ, qubit[i], paras=params[param_idx])
        #         param_idx += 1
        #         circuit.add(RY, qubit[i], paras=params[param_idx])
        #         param_idx += 1
        #         circuit.add(RZ, qubit[i], paras=params[param_idx])
        #         param_idx += 1
        #
        #     # 改进的纠缠层（全连接模式）
        #     for i in range(self.qubits_num):
        #         for j in range(i + 1, self.qubits_num):
        #             circuit.add(CNOT, qubit[j], qubit[i])
        #
        # # 输出层（每个qubit增加一个旋转门）
        # for i in range(self.qubits_num):
        #     circuit.add(RY, qubit[i], paras=params[param_idx])
        #     param_idx += 1
        #
        # # 使用所有qubit的测量结果（而非仅PauliZ(0)）
        # observables = [PauliZ(i) for i in range(self.qubits_num)]
        # return self.backend.expectation(circuit, observables)

    # 预测
    def predict(self, params, x):
        y_preds = self.forward(params, x)/ 2 + 0.5 + params[-1]
        return y_preds
    # 损失函数
    def cost(self, params, x, y):
        y_ = self.forward(params, x) / 2 + 0.5 + params[-1]
        # temp=y.to_numpy()
        loss = (y_ - y) ** 2
        return np.sum(loss)
    # 计算准确率
    def r2_Score(self, params, x, y):
        # temp = y.to_numpy()
        y_preds = self.predict(params, x)
        sse=np.sum((y_preds-y)**2)
        sst=np.sum((np.mean(y_preds)-y)**2)
        return 1-sse/sst

