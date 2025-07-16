import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# 1. 数据准备
# ================================================================
# 从给定的表格数据创建DataFrame

df = pd.read_csv('row_data.csv')
# df = pd.read_excel('o.xlsx')
# 提取特征和目标
X = df[['Windspeed', 'Winddir', 'Grid_V']].values
y = df['Grid_W'].values.reshape(-1, 1)
# X = df[['Group_1_PC1', 'Group_2_PC1', 'Group_3_PC1']].values
# y = df['label'].values.reshape(-1, 1)

# 数据标准化
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_scaled, test_size=0.2, random_state=42
)

# 转换为PyTorch张量
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# 创建数据加载器
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)


# 2. 定义MLP模型
# ================================================================
class RegressionMLP(nn.Module):
    def __init__(self, input_size, hidden_size1=64, hidden_size2=32, output_size=1):
        super(RegressionMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


# 初始化模型
input_size = X_train.shape[1]
model = RegressionMLP(input_size)

# 3. 训练配置
# ================================================================
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)

# 4. 训练模型
# ================================================================
num_epochs = 200
best_loss = float('inf')
patience = 15
patience_counter = 0

for epoch in range(num_epochs):
    # 训练阶段
    model.train()
    train_loss = 0.0
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * inputs.size(0)

    train_loss /= len(train_loader.dataset)

    # 验证阶段
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item() * inputs.size(0)

    test_loss /= len(test_loader.dataset)

    # 更新学习率
    scheduler.step(test_loss)

    # 早停机制
    if test_loss < best_loss:
        best_loss = test_loss
        patience_counter = 0
        torch.save(model.state_dict(), 'best_model.pth')
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    # 打印进度
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}] | '
              f'Train Loss: {train_loss:.6f} | Test Loss: {test_loss:.6f}')

# 5. 加载最佳模型并评估
# ================================================================
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

# 在测试集上进行预测
predictions = []
true_values = []
with torch.no_grad():
    for inputs, targets in test_loader:
        outputs = model(inputs)
        predictions.append(outputs.numpy())
        true_values.append(targets.numpy())

predictions = np.vstack(predictions)
true_values = np.vstack(true_values)

# 反标准化
predictions_orig = scaler_y.inverse_transform(predictions)
true_values_orig = scaler_y.inverse_transform(true_values)
# predictions_orig=predictions
# true_values_orig=true_values
# 计算评估指标
from sklearn.metrics import mean_absolute_error, r2_score

mae = mean_absolute_error(true_values_orig, predictions_orig)
r2 = r2_score(true_values_orig, predictions_orig)

print(f"\nFinal Evaluation Results:")
print(f"MAE: {mae:.4f}")
print(f"R² Score: {r2:.4f}")

# 6. 可视化结果（可选）
# ================================================================
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.scatter(true_values_orig, predictions_orig, alpha=0.6)
plt.plot([min(true_values_orig), max(true_values_orig)],
         [min(true_values_orig), max(true_values_orig)],
         'r--', lw=2)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title('True vs Predicted Values')
plt.grid(True)
plt.savefig('true_vs_predicted.png')
plt.show()