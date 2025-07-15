import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from matplotlib import rcParams

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体作为默认字体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题




def generate_correlated_data(n_samples, correlation):
    """
    生成具有指定相关性的两个特征的数据集

    参数:
    n_samples (int): 样本数量
    correlation (float): 特征间的相关系数 (-1 到 1)

    返回:
    np.array: 生成的数据集 (n_samples × 2)
    """
    # 创建协方差矩阵
    cov_matrix = np.array([[1, correlation],
                           [correlation, 1]])

    # 生成多元正态分布数据
    np.random.seed(42)  # 确保可重复性
    data = np.random.multivariate_normal(mean=[0, 0], cov=cov_matrix, size=n_samples)

    return data


def plot_data_with_feature_vectors(data, correlation):
    """
    绘制数据和特征向量

    参数:
    data (np.array): 输入数据 (n_samples × 2)
    correlation (float): 特征间的相关系数
    """
    # 创建图形和坐标轴
    plt.figure(figsize=(10, 8))

    # 提取特征
    feature1 = data[:, 0]
    feature2 = data[:, 1]

    # 绘制散点图
    plt.scatter(feature1, feature2, alpha=0.7, edgecolor='k', label='样本点')

    # 计算特征向量的方向（协方差矩阵的特征向量）
    cov_matrix = np.cov(data.T)
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    # 确定主方向（最大特征值对应的特征向量）
    main_vector_idx = np.argmax(eigenvalues)
    main_vector = eigenvectors[:, main_vector_idx]

    # 绘制特征向量（从原点出发）
    origin = np.array([0, 0])

    # 特征1向量（X轴方向）
    plt.quiver(*origin, 1, 0, color='r', scale=5, width=0.01,
               label='特征1向量 (X轴)', angles='xy', scale_units='xy')

    # 特征2向量（Y轴方向）
    plt.quiver(*origin, 0, 1, color='b', scale=5, width=0.01,
               label='特征2向量 (Y轴)', angles='xy', scale_units='xy')

    # 主成分方向（数据的主要变化方向）
    plt.quiver(*origin, *main_vector, color='g', scale=5, width=0.012,
               label='主方向 (PC1)', angles='xy', scale_units='xy')

    # 添加相关系数信息
    plt.text(0.05, 0.95, f'相关系数 r = {correlation:.2f}',
             transform=plt.gca().transAxes, fontsize=14,
             bbox=dict(facecolor='white', alpha=0.8))

    # 设置图形属性
    plt.axhline(0, color='gray', linestyle='--', alpha=0.5)
    plt.axvline(0, color='gray', linestyle='--', alpha=0.5)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.title(f'特征空间可视化 (n={len(data)})', fontsize=16)
    plt.xlabel('特征 1', fontsize=14)
    plt.ylabel('特征 2', fontsize=14)
    plt.axis('equal')
    plt.legend(loc='best', fontsize=12)

    # 根据数据范围设置坐标轴
    max_val = np.max(np.abs(data)) * 1.2
    plt.xlim(-max_val, max_val)
    plt.ylim(-max_val, max_val)

    plt.tight_layout()
    plt.show()


# 获取用户输入
n_samples = 20
correlation = float(input("请输入两个特征间的相关系数 (-1 到 1): "))

# 验证输入
if not -1 <= correlation <= 1:
    print("错误：相关系数必须在 -1 和 1 之间")
    exit()

# 生成数据并绘图
data = generate_correlated_data(n_samples, correlation)
plot_data_with_feature_vectors(data, correlation)