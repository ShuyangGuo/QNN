import pandas as pd
from wuyue_machine_learning.utils import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster import hierarchy
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

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
    # plt.figure(figsize=(12, 6))
    # corr_df.head(20).plot(kind='bar')
    # plt.title(f'Top Features Correlation with {target_col} ({method})')
    # plt.ylabel('Correlation Score')
    # plt.show()

    # 选择特征
    if top_k:
        selected_features = corr_df.head(top_k).index.tolist()
    elif threshold:
        selected_features = corr_df[corr_df > threshold].index.tolist()
    else:
        selected_features = corr_df.index.tolist()

    print(f"Selected {len(selected_features)} features with highest correlation to {target_col}")
    return selected_features


def cluster_correlated_features(df, features, method='non_exclusive', n_clusters=None, threshold=0.7):
    """
    将高度相关的特征分组（非互斥分组）

    参数:
    df: 数据框
    features: 要分组的特征列表
    method: 聚类方法 ('agglomerative', 'hierarchical', 'non_exclusive')
    n_clusters: 聚类数量 (仅用于凝聚聚类)
    threshold: 相关性阈值 (用于层次聚类和非互斥分组)

    返回:
    特征分组字典 {组名: [特征列表]}
    """
    # 计算特征间相关性
    corr_matrix = df[features].corr().abs()

    # 可视化相关性热力图
    # plt.figure(figsize=(12, 10))
    # sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0.5)
    # plt.title('Feature Correlation Matrix')
    # plt.show()

    feature_groups = {}

    if method == 'non_exclusive':
        # ========== 新增的非互斥分组方法 ==========
        # 存储已处理的特征对
        processed_pairs = set()
        group_counter = 1

        # 遍历所有特征对
        for i, feat1 in enumerate(features):
            for j, feat2 in enumerate(features):
                if i >= j:  # 避免重复处理和对角线
                    continue

                # 检查相关性是否高于阈值且未处理过
                if corr_matrix.iloc[i, j] > threshold and (feat1, feat2) not in processed_pairs:
                    # 创建新组
                    group_name = f"Group_{group_counter}"
                    group_counter += 1
                    feature_groups[group_name] = [feat1, feat2]
                    processed_pairs.add((feat1, feat2))
                    processed_pairs.add((feat2, feat1))  # 添加反向对

                    # 查找与当前组相关的其他特征
                    for k, other_feat in enumerate(features):
                        if other_feat in [feat1, feat2]:
                            continue

                        # 检查与组内所有特征的相关性
                        if all(corr_matrix.loc[other_feat, feat] > threshold for feat in feature_groups[group_name]):
                            feature_groups[group_name].append(other_feat)
                            # 添加新特征与组内所有特征的配对
                            for feat in feature_groups[group_name]:
                                if feat != other_feat:
                                    processed_pairs.add((feat, other_feat))
                                    processed_pairs.add((other_feat, feat))

        # 处理未分组的高相关特征对
        for i, feat1 in enumerate(features):
            for j, feat2 in enumerate(features):
                if i >= j:
                    continue

                if corr_matrix.iloc[i, j] > threshold and (feat1, feat2) not in processed_pairs:
                    group_name = f"Group_{group_counter}"
                    group_counter += 1
                    feature_groups[group_name] = [feat1, feat2]
                    processed_pairs.add((feat1, feat2))
                    processed_pairs.add((feat2, feat1))
        for i, feat1 in enumerate(features):
            for value in feature_groups.values():
                if feat1 in value:
                    break
            else:
                group_name = f"Group_{group_counter}"
                group_counter += 1
                feature_groups[group_name] = [feat1]
    elif method == 'agglomerative':
        # ========== 凝聚聚类 (互斥分组) ==========
        from sklearn.cluster import AgglomerativeClustering

        if not n_clusters:
            n_clusters = max(1, int(len(features) / 5))  # 默认每组约5个特征

        cluster = AgglomerativeClustering(n_clusters=n_clusters,
                                          affinity='precomputed',
                                          linkage='complete')

        # 使用(1 - 相关性)作为距离
        distance_matrix = 1 - corr_matrix.values
        clusters = cluster.fit_predict(distance_matrix)

        # 创建分组
        for i, feat in enumerate(features):
            group_id = f"Group_{clusters[i] + 1}"
            if group_id not in feature_groups:
                feature_groups[group_id] = []
            feature_groups[group_id].append(feat)

    elif method == 'hierarchical':
        # ========== 层次聚类 (互斥分组) ==========
        linkage = hierarchy.linkage(1 - corr_matrix.values, method='ward')
        plt.figure(figsize=(12, 8))
        dendro = hierarchy.dendrogram(linkage, labels=features, orientation='left')
        plt.title('Feature Hierarchical Clustering Dendrogram')
        plt.show()

        # 基于阈值创建分组
        clusters = hierarchy.fcluster(linkage, t=threshold, criterion='distance')
        for i, cluster_id in enumerate(clusters):
            group_id = f"Group_{cluster_id}"
            if group_id not in feature_groups:
                feature_groups[group_id] = []
            feature_groups[group_id].append(features[i])

    # 输出分组信息
    print(f"Created {len(feature_groups)} feature groups:")
    for group, feats in feature_groups.items():
        print(f"{group}: {len(feats)} features - {feats}")

    return feature_groups



def apply_group_pca(df, feature_groups, n_components=0.95, standardize=True):
    """
    对每个特征组应用PCA降维

    参数:
    df: 原始数据框
    feature_groups: 特征分组字典 {组名: [特征列表]}
    n_components: 保留的主成分数量或方差解释比例
                 整数: 保留的主成分数量
                 浮点数(0-1): 保留的方差解释比例
    standardize: 是否在PCA前标准化数据

    返回:
    df_pca: 包含主成分的新数据框
    pca_info: 包含每个组PCA模型信息的字典
    """
    df_pca = pd.DataFrame()
    pca_info = {}

    # 遍历每个特征组
    for group_name, features in feature_groups.items():
        group_data = df[features].copy()

        # 处理只有一个特征的情况
        if len(features) == 1:
            # 单个特征无法进行PCA，直接使用原始特征
            principal_components = group_data.values
            comp_names = [f"{group_name}_PC1"]

            # 存储PCA信息
            pca_info[group_name] = {
                'n_components': 1,
                'explained_variance_ratio': [1.0],
                'components': np.array([[1.0]]),
                'features': features,
                'scaler': None,
                'pca_model': None
            }
        else:
            # 标准化数据
            if standardize:
                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(group_data)
            else:
                scaler = None
                scaled_data = group_data.values

            # 创建并拟合PCA模型
            pca = PCA(n_components=n_components)
            principal_components = pca.fit_transform(scaled_data)

            # 确定实际保留的主成分数量
            n_comp = principal_components.shape[1]
            comp_names = [f"{group_name}_PC{i + 1}" for i in range(1)]

            # 存储PCA信息
            pca_info[group_name] = {
                'n_components': n_comp,
                'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
                'components': pca.components_,
                'features': features,
                'scaler': scaler,
                'pca_model': pca
            }

        # 添加主成分到数据框
        for i, col_name in enumerate(comp_names):
            df_pca[col_name] = principal_components[:, i]

    return df_pca, pca_info
# def cluster_correlated_features(df, features, method='agglomerative', n_clusters=None, threshold=0.7):
#     """
#     将高度相关的特征分组
#
#     参数:
#     df: 数据框
#     features: 要分组的特征列表
#     method: 聚类方法 ('agglomerative', 'hierarchical')
#     n_clusters: 聚类数量 (可选)
#     threshold: 相关性阈值 (用于层次聚类)
#
#     返回:
#     特征分组字典 {组名: [特征列表]}
#     """
#     # 计算特征间相关性
#     corr_matrix = df[features].corr().abs()
#
#     # 可视化相关性热力图
#     plt.figure(figsize=(12, 10))
#     sns.heatmap(corr_matrix, annot=False, cmap='coolwarm')
#     plt.title('Feature Correlation Matrix')
#     plt.show()
#
#     # 执行聚类
#     if method == 'agglomerative':
#         # 凝聚聚类
#         if not n_clusters:
#             n_clusters = int(len(features) / 5)  # 默认每组约5个特征
#
#         cluster = AgglomerativeClustering(n_clusters=n_clusters,
#                                           affinity='precomputed',
#                                           linkage='complete')
#
#         # 使用(1 - 相关性)作为距离
#         distance_matrix = 1 - corr_matrix.values
#         clusters = cluster.fit_predict(distance_matrix)
#
#         # 创建分组
#         feature_groups = {}
#         for i, feat in enumerate(features):
#             group_id = f"Group_{clusters[i] + 1}"
#             if group_id not in feature_groups:
#                 feature_groups[group_id] = []
#             feature_groups[group_id].append(feat)
#
#     elif method == 'hierarchical':
#         # 层次聚类
#         linkage = hierarchy.linkage(1 - corr_matrix.values, method='ward')
#         plt.figure(figsize=(12, 8))
#         dendro = hierarchy.dendrogram(linkage, labels=features, orientation='left')
#         plt.title('Feature Hierarchical Clustering Dendrogram')
#         plt.show()
#
#         # 基于阈值创建分组
#         clusters = hierarchy.fcluster(linkage, t=threshold, criterion='distance')
#         feature_groups = {}
#         for i, cluster_id in enumerate(clusters):
#             group_id = f"Group_{cluster_id}"
#             if group_id not in feature_groups:
#                 feature_groups[group_id] = []
#             feature_groups[group_id].append(features[i])
#
#     # 输出分组信息
#     print(f"Created {len(feature_groups)} feature groups:")
#     for group, feats in feature_groups.items():
#         print(f"{group}: {len(feats)} features")
#
#     return feature_groups