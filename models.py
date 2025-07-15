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
        self.qubits_num=qubits_num  # 量子比特数量
    def forward(self, params, x):
        qubit = QuantumRegister(self.qubits_num)
        cbit = ClassicalRegister(self.qubits_num)
        circuit = QuantumCircuit(qubit, cbit)

        # 角度编码
        for i in range(self.qubits_num):
            circuit.add(RX, qubit[i], paras=np.arcsin(x[:,i])).add(RY, qubit[i],paras=np.arccos(x[:,i] ** 2))

        # # 参数化量子线路
        # index=0
        # #Z-Y-Z
        # for i in range(self.qubits_num):
        #     circuit.add(RZ,qubit[i],paras=params[index])
        #     index += 1
        #     circuit.add(RY, qubit[i], paras=params[index])
        #     index += 1
        #     circuit.add(RZ, qubit[i], paras=params[index])
        #     index += 1
        # #Z-Y-Z-CNOT
        # for i in range(self.qubits_num-1):
        #     circuit.add(CNOT, qubit[i+1], qubit[i])
        # circuit.add(CNOT,qubit[0], qubit[self.qubits_num-1])
        # # Z-Y-Z-CNOT-Y
        # for i in range(self.qubits_num):
        #     circuit.add(RY,qubit[i],paras=params[index])
        #     index += 1

        return self.backend.expectation(circuit, PauliZ(0))
    # 预测
    def predict(self, params, x):
        y_preds = self.forward(params, x)
        y_preds = y_preds / 2 + 0.5 + params[-1]
        # y_preds = [1 if i > 0.5 else 0 for i in y_preds]
        return y_preds
    # 损失函数
    def cost(self, params, x, y):
        y_ = self.forward(params, x) / 2 + 0.5 + params[-1]
        # print(y_)
        # print(y)
        loss = (y_ - y) ** 2
        return np.sum(loss)
    # 计算准确率
    def MSE_Score(self, params, x, y):
        y_preds = self.predict(params, x)/ 2 + 0.5 + params[-1]
        score=np.mean((y_preds-y)**2)
        return score

