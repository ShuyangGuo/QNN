import pandas as pd
from wuyue_machine_learning.utils import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster import hierarchy

def select_high_correlation_features(df, target_col, method='pearson', top_k=None, threshold=None):
    """
    选择与目标变量高度相关的特征

    参数:
    df: 包含特征和目标的数据框
    target_col: 目标列名
    method: 相关性计算方法 ('pearson', 'spearman', 'mutual_info')
    top_k: 选择前k个最相关的特征
    threshold: 选择相关性绝对值大于阈值的特征

    返回:
    选择的特征列表
    """
    # 计算相关性
    if method in ['pearson', 'spearman']:
        corr = df.corr(method=method)[target_col].abs()
        corr_df = corr.drop(target_col).sort_values(ascending=False)
    elif method == 'mutual_info':
        X = df.drop(columns=[target_col])
        y = df[target_col]
        mi = mutual_info_regression(X, y)
        corr_df = pd.Series(mi, index=X.columns).sort_values(ascending=False)

    # 可视化相关性
    plt.figure(figsize=(12, 6))
    corr_df.head(20).plot(kind='bar')
    plt.title(f'Top Features Correlation with {target_col} ({method})')
    plt.ylabel('Correlation Score')
    plt.show()

    # 选择特征
    if top_k:
        selected_features = corr_df.head(top_k).index.tolist()
    elif threshold:
        selected_features = corr_df[corr_df > threshold].index.tolist()
    else:
        selected_features = corr_df.index.tolist()

    print(f"Selected {len(selected_features)} features with highest correlation to {target_col}")
    return selected_features





def cluster_correlated_features(df, features, method='agglomerative', n_clusters=None, threshold=0.7):
    """
    将高度相关的特征分组

    参数:
    df: 数据框
    features: 要分组的特征列表
    method: 聚类方法 ('agglomerative', 'hierarchical')
    n_clusters: 聚类数量 (可选)
    threshold: 相关性阈值 (用于层次聚类)

    返回:
    特征分组字典 {组名: [特征列表]}
    """
    # 计算特征间相关性
    corr_matrix = df[features].corr().abs()

    # 可视化相关性热力图
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm')
    plt.title('Feature Correlation Matrix')
    plt.show()

    # 执行聚类
    if method == 'agglomerative':
        # 凝聚聚类
        if not n_clusters:
            n_clusters = int(len(features) / 5)  # 默认每组约5个特征

        cluster = AgglomerativeClustering(n_clusters=n_clusters,
                                          affinity='precomputed',
                                          linkage='complete')

        # 使用(1 - 相关性)作为距离
        distance_matrix = 1 - corr_matrix.values
        clusters = cluster.fit_predict(distance_matrix)

        # 创建分组
        feature_groups = {}
        for i, feat in enumerate(features):
            group_id = f"Group_{clusters[i] + 1}"
            if group_id not in feature_groups:
                feature_groups[group_id] = []
            feature_groups[group_id].append(feat)

    elif method == 'hierarchical':
        # 层次聚类
        linkage = hierarchy.linkage(1 - corr_matrix.values, method='ward')
        plt.figure(figsize=(12, 8))
        dendro = hierarchy.dendrogram(linkage, labels=features, orientation='left')
        plt.title('Feature Hierarchical Clustering Dendrogram')
        plt.show()

        # 基于阈值创建分组
        clusters = hierarchy.fcluster(linkage, t=threshold, criterion='distance')
        feature_groups = {}
        for i, cluster_id in enumerate(clusters):
            group_id = f"Group_{cluster_id}"
            if group_id not in feature_groups:
                feature_groups[group_id] = []
            feature_groups[group_id].append(features[i])

    # 输出分组信息
    print(f"Created {len(feature_groups)} feature groups:")
    for group, feats in feature_groups.items():
        print(f"{group}: {len(feats)} features")

    return feature_groups