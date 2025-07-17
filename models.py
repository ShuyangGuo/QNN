from matplotlib import pyplot as plt
import numpy as np
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
        # for i in range(self.qubits_num):
        #     circuit.add(RY, qubit[i], paras=x[:, i] * np.pi)
            # circuit.add(RY, qubit[i], paras=np.arcsin(x[:,i])).add(RZ, qubit[i],paras=np.arccos(x[:,i] ** 2))
        # for i in range(self.qubits_num - 6,self.qubits_num):
        #     circuit.add(RY, qubit[self.qubits_num - 6], paras=np.arcsin(x[:, i]) * np.pi)
        circuit.add(RY, qubit[0], paras=x[:, 0] * np.pi)
        circuit.add(RY, qubit[1], paras=x[:, 1] * np.pi)
        circuit.add(RY, qubit[2], paras=2 * np.arctan2(x[:,2],x[:,3]))
        circuit.add(RY, qubit[3], paras=2 * np.arctan2(x[:,4],x[:,5]))
        circuit.add(RY, qubit[4], paras=2 * np.arctan2(x[:, 6], x[:, 7]))
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

        # #Z-Y-Z-CNOT
        # circuit.add(CNOT, qubit[0], qubit[1])
        # circuit.add(CNOT, qubit[1], qubit[2])
        # circuit.add(CNOT, qubit[2], qubit[3])
        # circuit.add(CNOT, qubit[3], qubit[4])
        # circuit.add(CNOT, qubit[4], qubit[0])

        # for i in range(self.qubits_num-1):
        #     circuit.add(CNOT, qubit[i+1], qubit[i])
        # circuit.add(CNOT,qubit[0], qubit[self.qubits_num-1])
        # Z-Y-Z-CNOT-Y
        for i in range(self.qubits_num):
            circuit.add(RY,qubit[i],paras=params[index])
            index += 1
        return self.backend.expectation(circuit, PauliZ(0))
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

