from matplotlib import pyplot as plt
from data_process import get_data
from models import MyQNN
from wuyue_machine_learning.optimizer import Adam
from wuyue_machine_learning.utils import numpy as np
import openpyxl

train_x, test_x, train_y, test_y=get_data()
# train_x.to_excel('o.xlsx')
train_x,test_x, train_y, test_y=list(map(lambda df: df.to_numpy(), [train_x,test_x, train_y, test_y]))
# 参数设置
qubits_num = train_x.shape[1]  # 量子比特数量
batch_size = 1000  # 训练batch大小
epochs = 20 # 训练轮数
train_num=train_x.shape[0]

# 初始化参数
params = np.random.rand(qubits_num*4+5)
params[-1] = 0.0

# 实例化量子神经网络
myqnn = MyQNN(qubits_num)

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
    train_acc = myqnn.MSE_Score(params, train_x, train_y)
    test_acc = myqnn.MSE_Score(params, test_x, test_y)
    print(f"epochs:{epochs},loss:{loss_show},train_acc:{train_acc},test_acc:{test_acc}")

# print(train_losses)
# print(test_losses)
# plt.plot(range(epochs+1),train_losses,c='blue')
# plt.plot(range(epochs+1),test_losses,c='red')
# plt.show()

# 在测试集上进行预测
predictions = myqnn.predict(params,test_x)
true_values = test_y
print(predictions)
print(true_values)
# 6. 可视化结果（可选）
# ================================================================
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.scatter(true_values, predictions, alpha=0.6)
plt.plot([min(true_values), max(true_values)],
         [min(true_values), max(true_values)],
         'r--', lw=2)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title('True vs Predicted Values')
plt.grid(True)
plt.savefig('true_vs_predicted.png')
plt.show()