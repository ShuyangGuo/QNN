from matplotlib import pyplot as plt
from data_process import get_data
from models import MyQNN
from wuyue_machine_learning.optimizer import Adam
from wuyue_machine_learning.utils import numpy as np

train_x, test_x, train_y, test_y=get_data()
train_x,test_x, train_y, test_y=list(map(lambda df: df.to_numpy(), [train_x,test_x, train_y, test_y]))
# 参数设置
qubits_num = 4  # 量子比特数量
batch_size = 1000  # 训练batch大小
epochs = 20 # 训练轮数
train_num=train_x.shape[0]

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
plt.plot(range(epochs+1),train_losses,c='blue')
plt.plot(range(epochs+1),test_losses,c='red')
plt.show()