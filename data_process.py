import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import feature_process
from data_analysis import analyze_data_anomalies
from cross_val import generate_k_folds
import openpyxl

def TimeToStamp(time_str):
    # 解析时间格式（%Y：年，%m：月，%d：日，%H：时，%M：分）
    dt = datetime.strptime(time_str, "%Y/%m/%d %H:%M")
    # 转换为时间戳（秒级）
    timestamp = dt.timestamp()
    return timestamp

def min_max(data):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    return scaled_data


def get_data():
    # 从CSV文件导入数据，需替换为实际文件路径
    try:
        data = pd.read_csv('row_data.csv')
        print(f"数据成功导入，共{len(data)}行，{len(data.columns)}列")
        # print("数据前几行信息：")
        # print(data.head().to_string())
        data['Time']=data['Time'].map(TimeToStamp)
        #特征筛选与分类
        select_f = feature_process.select_high_correlation_features(data,'Grid_W',threshold=0.4)
        f_cluster=feature_process.cluster_correlated_features(data,select_f)
        #计算主成分
        # data_pca, pca_info=feature_process.apply_group_pca(data,f_cluster)
        # data_pca['label'] = data.iloc[:, 2]

        data = pd.DataFrame(
            min_max(data),
            columns=data.columns,  # 保留列名
            index=data.index  # 保留索引
        )
        # data_pca.to_excel('o.xlsx')
        # 提取特征列和目标列（需根据实际数据调整）
        # 假设最后一列是目标变量，其余为特征
        X = data[select_f]
        y = data['Grid_W']
        return X,y
        # X = data[['Windspeed', 'Winddir', 'Grid_V']]
        # y = data['Grid_W']

        # analyze_data_anomalies(X)
        # analyze_data_anomalies(y)
        # 划分训练集和验证集（80%训练，20%验证）
        # X_train, X_val, y_train, y_val = train_test_split(
        #     X, y, test_size=0.2, random_state=42, shuffle=True
        # )
        #
        # print(f"\n训练集：{X_train.shape[0]}条数据，特征维度{X_train.shape[1]}")
        # print(f"验证集：{X_val.shape[0]}条数据，特征维度{X_val.shape[1]}")
        #
        # return X_train, X_val, y_train, y_val
        # 保存划分后的数据集（可选）
        # X_train.to_csv('X_train.csv', index=False)
        # y_train.to_csv('y_train.csv', index=False)
        # X_val.to_csv('X_val.csv', index=False)
        # y_val.to_csv('y_val.csv', index=False)
        # print("\n划分结果已保存至CSV文件")

    except FileNotFoundError:
        print("错误：文件未找到，请检查文件路径是否正确")
    except Exception as e:
        print(f"错误：发生未知异常 - {e}")

def get_folds():
    # 从CSV文件导入数据，需替换为实际文件路径
    try:
        data = pd.read_csv('row_data.csv')
        print(f"数据成功导入，共{len(data)}行，{len(data.columns)}列")
        # print("数据前几行信息：")
        # print(data.head().to_string())
        data['Time']=data['Time'].map(TimeToStamp)
        #特征筛选与分类
        select_f = feature_process.select_high_correlation_features(data,'Grid_W',threshold=0.4)
        f_cluster=feature_process.cluster_correlated_features(data,select_f)
        #计算主成分
        # data_pca, pca_info=feature_process.apply_group_pca(data,f_cluster)
        # data_pca['label'] = data.iloc[:, 2]

        data = pd.DataFrame(
            min_max(data),
            columns=data.columns,  # 保留列名
            index=data.index  # 保留索引
        )
        select_f.append('Grid_W')
        data=data[select_f]
        # 生成k-fold（k默认5）
        return generate_k_folds(data)


    except FileNotFoundError:
        print("错误：文件未找到，请检查文件路径是否正确")
    except Exception as e:
        print(f"错误：发生未知异常 - {e}")