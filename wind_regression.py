from matplotlib import pyplot as plt
from data_process import get_data,get_folds
from cross_val import cross_val
from models import MyQNN
from wuyue_machine_learning.optimizer import Adam
from wuyue_machine_learning.utils import numpy as np
import openpyxl

def visualization(true_values,predictions):
    # 6. 可视化结果（可选）
    # ================================================================
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

def train(train_x, test_x, train_y, test_y,epochs=100,batch_size = 1000):
    train_x,test_x, train_y, test_y=list(map(lambda df: df.to_numpy(), [train_x,test_x, train_y, test_y]))
    # 初始化参数
    params = np.random.rand(qubits_num * 4 + 1)
    params[-1] = 0.0
    train_num = train_x.shape[0]

    # 初始化优化器
    adam = Adam(params, 0.01)
    train_losses = []
    test_losses = []
    for epochs in range(epochs):
        for itr in range(train_num // batch_size):
            gradient = myqnn.backward(params, train_x[itr * batch_size:(itr + 1) * batch_size], \
                                      train_y[itr * batch_size:(itr + 1) * batch_size])
            params = adam.update(gradient)
        loss_show = myqnn.cost(params, train_x, train_y)
        train_losses.append(loss_show)
        loss_sh = myqnn.cost(params, test_x, test_y)
        test_losses.append(loss_sh)
        train_r2 = myqnn.r2_Score(params, train_x, train_y)
        test_r2 = myqnn.r2_Score(params, test_x, test_y)
        print(f"epochs:{epochs},train_loss:{loss_show},test_loss:{loss_sh},train_r2:{train_r2},R² Score:{test_r2}")
    # print(train_losses)
    # print(test_losses)
    # plt.plot(range(epochs+1),train_losses,c='blue')
    # plt.plot(range(epochs+1),test_losses,c='red')
    # plt.show()
    return loss_show,loss_sh,params,test_r2





# 数据导入
# train_x, test_x, train_y, test_y=get_data()
# train_x.to_excel('o.xlsx')
# train_x,test_x, train_y, test_y=list(map(lambda df: df.to_numpy(), [train_x,test_x, train_y, test_y]))
folds=get_folds()
# 参数设置
qubits_num = folds[0][0].shape[1]-1  # 量子比特数量
# 实例化量子神经网络
myqnn = MyQNN(qubits_num)
# 训练(cross validation)
aver_val,avl_r2score,k_params=cross_val(folds,train)
print(f"average validation loss:{aver_val},average R² Score:{avl_r2score}")

# 采用bagging在完整数据集上进行预测
#合并权重
X,y=get_data()
X,y=list(map(lambda df: df.to_numpy(), [X,y]))
aver_params=np.stack(k_params).mean(axis=0)
predictions = myqnn.predict(aver_params, X)
#不合并权重，求预测的平均（未完成）
# predictions_bag=[]
# for param in k_params:
#     predictions_bag.append(myqnn.predict(param, X))
# print(predictions_bag)
# predictions = myqnn.predict(aver_params, X)

# 采用boosting在完整数据集上进行预测(未完成)
true_values = y
visualization(true_values,predictions)
