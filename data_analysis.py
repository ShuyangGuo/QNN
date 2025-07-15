import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


def analyze_data_anomalies(df, num_threshold=3, cat_rare_threshold=0.01, nan_threshold=0.3):
    """
    统计分析数据中的异常值（NaN、离群值、稀有类别等）

    参数:
    df: 输入数据框
    num_threshold: 数值特征离群值的Z分数阈值（默认3）
    cat_rare_threshold: 分类特征稀有类别的频率阈值（默认0.01）
    nan_threshold: 高缺失率特征的阈值（默认0.3）

    返回:
    anomaly_report: 包含异常统计的字典
    """
    anomaly_report = {
        "missing_values": {},
        "outliers": {},
        "rare_categories": {},
        "high_missing_features": [],
        "constant_features": [],
        "high_cardinality_features": []
    }

    # 1. 分析缺失值
    total_rows = len(df)
    missing_data = df.isnull().sum()
    missing_percent = (missing_data / total_rows) * 100

    for col in df.columns:
        if missing_data[col] > 0:
            anomaly_report["missing_values"][col] = {
                "count": missing_data[col],
                "percent": missing_percent[col],
                "samples": df[df[col].isnull()].sample(min(5, missing_data[col])).index.tolist()
            }

            # 识别高缺失率特征
            if missing_percent[col] > nan_threshold * 100:
                anomaly_report["high_missing_features"].append(col)

    # 2. 分析数值特征
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns

    for col in numerical_cols:
        col_data = df[col].dropna()

        # 检查是否为常数特征
        if col_data.nunique() == 1:
            anomaly_report["constant_features"].append(col)
            continue

        # 计算Z分数检测离群值
        z_scores = np.abs(stats.zscore(col_data))
        outliers = col_data[z_scores > num_threshold]

        if not outliers.empty:
            anomaly_report["outliers"][col] = {
                "count": len(outliers),
                "percent": (len(outliers) / len(col_data)) * 100,
                "min": outliers.min(),
                "max": outliers.max(),
                "mean": outliers.mean(),
                "samples": outliers.sample(min(5, len(outliers))).index.tolist()
            }

    # 3. 分析分类特征
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns

    for col in categorical_cols:
        col_data = df[col]

        # 检查高基数特征
        unique_count = col_data.nunique()
        if unique_count > 50 and unique_count > len(col_data) * 0.5:
            anomaly_report["high_cardinality_features"].append({
                "feature": col,
                "unique_count": unique_count,
                "samples": col_data.sample(min(5, len(col_data))).unique().tolist()
            })

        # 检查常数特征
        if unique_count == 1:
            anomaly_report["constant_features"].append(col)
            continue

        # 识别稀有类别
        value_counts = col_data.value_counts(normalize=True)
        rare_categories = value_counts[value_counts < cat_rare_threshold].index.tolist()

        if rare_categories:
            rare_samples = []
            for category in rare_categories:
                samples = df[df[col] == category].sample(min(3, len(df[df[col] == category]))).index.tolist()
                rare_samples.append({
                    "category": category,
                    "percent": value_counts[category] * 100,
                    "samples": samples
                })

            anomaly_report["rare_categories"][col] = {
                "count": len(rare_categories),
                "total_categories": unique_count,
                "categories": rare_samples
            }

    # 4. 可视化分析
    # 缺失值热力图
    if not df.isnull().sum().sum() == 0:
        plt.figure(figsize=(12, 8))
        sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
        plt.title('Missing Values Heatmap')
        plt.show()

    # 数值特征分布箱线图
    if numerical_cols.any():
        plt.figure(figsize=(12, 8))
        df[numerical_cols].boxplot()
        plt.title('Numerical Features Distribution (Boxplot)')
        plt.xticks(rotation=45)
        plt.show()

    # 5. 生成报告摘要
    print("\n" + "=" * 50)
    print("数据异常分析报告")
    print("=" * 50)

    # 缺失值报告
    if anomaly_report["missing_values"]:
        print(f"\n[缺失值] 共发现 {len(anomaly_report['missing_values'])} 个特征有缺失值:")
        for col, data in anomaly_report["missing_values"].items():
            print(f"  - {col}: {data['count']} 个缺失 ({data['percent']:.2f}%)")

        if anomaly_report["high_missing_features"]:
            print(f"\n[高缺失率特征] 以下特征缺失率 > {nan_threshold * 100}%:")
            for col in anomaly_report["high_missing_features"]:
                percent = anomaly_report["missing_values"][col]["percent"]
                print(f"  - {col}: {percent:.2f}%")

    # 离群值报告
    if anomaly_report["outliers"]:
        print(f"\n[离群值] 共发现 {len(anomaly_report['outliers'])} 个特征有离群值 (Z分数 > {num_threshold}):")
        for col, data in anomaly_report["outliers"].items():
            print(f"  - {col}: {data['count']} 个离群值 ({data['percent']:.2f}%)")

    # 稀有类别报告
    if anomaly_report["rare_categories"]:
        print(
            f"\n[稀有类别] 共发现 {len(anomaly_report['rare_categories'])} 个特征有稀有类别 (频率 < {cat_rare_threshold * 100}%):")
        for col, data in anomaly_report["rare_categories"].items():
            print(f"  - {col}: {data['count']}/{data['total_categories']} 个稀有类别")

    # 常数特征报告
    if anomaly_report["constant_features"]:
        print(f"\n[常数特征] 共发现 {len(anomaly_report['constant_features'])} 个常数特征:")
        for col in anomaly_report["constant_features"]:
            print(f"  - {col}")

    # 高基数特征报告
    if anomaly_report["high_cardinality_features"]:
        print(f"\n[高基数特征] 共发现 {len(anomaly_report['high_cardinality_features'])} 个高基数特征:")
        for data in anomaly_report["high_cardinality_features"]:
            print(f"  - {data['feature']}: {data['unique_count']} 个唯一值")

    # 总体情况
    print("\n" + "-" * 50)
    print("数据异常统计摘要:")
    print(f"总特征数: {len(df.columns)}")
    print(f"有缺失值的特征数: {len(anomaly_report['missing_values'])}")
    print(f"有离群值的特征数: {len(anomaly_report['outliers'])}")
    print(f"有稀有类别的特征数: {len(anomaly_report['rare_categories'])}")
    print(f"常数特征数: {len(anomaly_report['constant_features'])}")
    print(f"高基数特征数: {len(anomaly_report['high_cardinality_features'])}")
    print(f"高缺失率特征数: {len(anomaly_report['high_missing_features'])}")

    return anomaly_report