from wuyue_machine_learning.utils import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold

def generate_k_folds(df, k=5, stratify_col=None, random_state=42):
    """
    输入DataFrame，生成k个fold（训练集+验证集）

    参数：
    df: pd.DataFrame - 输入的数据集
    k: int - 折数，默认5
    stratify_col: str - 分层依据的列名（用于分类任务，保持类别比例），默认None（普通k折）
    random_state: int - 随机种子，确保结果可复现

    返回：
    folds: list - 每个元素为元组(train_df, val_df)，即第i折的训练集和验证集
    """
    # 检查输入合法性
    if k < 2:
        raise ValueError("k必须大于等于2")
    if stratify_col is not None and stratify_col not in df.columns:
        raise ValueError(f"分层列'{stratify_col}'不在DataFrame中")

    # 生成索引划分器
    if stratify_col is not None:
        # 分层k折（适用于分类任务，保持类别比例）
        y = df[stratify_col]  # 分层依据的目标列
        kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state)
        splits = kf.split(df, y)  # 基于目标列分层划分
    else:
        # 普通k折（适用于回归或类别平衡的分类任务）
        kf = KFold(n_splits=k, shuffle=True, random_state=random_state)
        splits = kf.split(df)  # 随机划分

    # 生成每个fold的训练集和验证集
    folds = []
    for train_idx, val_idx in splits:
        train_df = df.iloc[train_idx].copy()  # 训练集（根据索引提取）
        val_df = df.iloc[val_idx].copy()  # 验证集（根据索引提取）
        folds.append((train_df, val_df))

    return folds

def cross_val(folds,train_fuc):
    sum_val=[]
    k_params=[]
    R2Scores=[]
    for fold in folds:
        # print(qubits_num)
        # batch_size = 1000  # 训练batch大小
        # epochs = 100  # 训练轮数
        train_y,train_x = fold[0]['Grid_W'],fold[0].drop(['Grid_W'],axis=1)
        test_y,test_x = fold[1]['Grid_W'],fold[1].drop(['Grid_W'],axis=1)
        loss_train,loss_val,params,R2Score=train_fuc(train_x, test_x, train_y, test_y)
        k_params.append(params)
        sum_val.append(loss_val)
        R2Scores.append(R2Score)
    aver_val=np.mean(sum_val)
    avl_r2score=np.mean(R2Scores)
    return aver_val,avl_r2score,k_params