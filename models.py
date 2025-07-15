from matplotlib import pyplot as plt

from wuyue.register.quantumregister import QuantumRegister
from wuyue.register.classicalregister import ClassicalRegister
from wuyue.circuit.circuit import QuantumCircuit
from wuyue.element.gate import RY, RZ, CNOT
from wuyue_machine_learning.utils import numpy as np
from wuyue_machine_learning.optimizer import Adam
from wuyue_machine_learning.qnn import QNN
from wuyue_machine_learning.backend.observable import *

# 构建量子神经网络
class MyQNN(QNN):
    def __init__(self):
        super().__init__()
        self.qubits_num = 4  # 量子比特数量
    def forward(self, params, x):
        qubit = QuantumRegister(self.qubits_num)
        cbit = ClassicalRegister(self.qubits_num)
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