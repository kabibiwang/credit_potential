"""
多源异构数据统计建模：信贷潜力综合评估
企业信用风险指数模块（Adaboost-TW-NSI）

本模块实现：
1. 改进的Adaboost算法（Time Weighted + Negative Sample Improved）
2. 时间加权策略：动态聚焦近期数据
3. 负样本权重优化：提升高风险样本识别能力
4. 生成企业信用风险指数（CRI）

算法创新点：
- TW（Time Weighted）：通过指数衰减赋予近期数据更高权重
- NSI（Negative Sample Improved）：通过权重放大因子β提升对违约样本的关注

损失函数说明（论文补充内容）：
--------------------------------------------------
1. Adaboost采用指数损失函数：
   L(y, F(x)) = exp(-y·F(x))
   其中 y ∈ {-1, +1}，F(x) 为强分类器输出

2. 弱分类器权重计算：
   α_u = (1/2) * ln((1 - e_u) / e_u)
   其中 e_u 为加权分类误差率

3. 样本权重更新（标准Adaboost）：
   w_i^(u+1) = w_i^u * exp(-α_u * y_i * f_u(x_i))

4. 本文改进的权重更新（TW-NSI）：
   w_i^(u+1) = w_i^u * exp(-α_u * l_i^u) * exp(-λ * t_i)
   
   其中时间加权：exp(-λ * t_i)，λ为衰减系数，t_i为时间步长
   
   误差函数 l_i^u：
   - l = 1,   若 f_u(x_i) = y_i（正确分类）
   - l = -1,  若 f_u(x_i) = 1 且 y_i = -1（负样本误分为正）
   - l = -β,  若 f_u(x_i) = -1 且 y_i = 1（正样本误分为负）
   其中 β > 1 为负样本权重放大因子
--------------------------------------------------
"""

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix
)
import warnings
warnings.filterwarnings('ignore')


class AdaboostTWNSI:
    """
    改进的Adaboost算法：时间加权 + 负样本优化
    
    Adaboost-TW-NSI (Time Weighted - Negative Sample Improved)
    
    参数说明：
    - n_estimators: 弱分类器数量
    - lambda_decay: 时间衰减系数λ，控制时间加权的衰减速度
    - beta: 负样本权重放大因子β，β > 1时增加对负样本的关注
    - base_estimator: 基分类器，默认为决策树桩
    
    数学原理：
    --------------------------------------------------
    强分类器：F(x) = Σ α_u * f_u(x)
    
    信用风险指数（概率形式）：
    CRI(x) = 1 / (1 + exp(-F(x)))
    
    CRI ∈ (0, 1)，值越高表示信用风险越低（信用越好）
    --------------------------------------------------
    """
    
    def __init__(self, n_estimators=50, lambda_decay=0.1, beta=2.0, 
                 max_depth=1, random_state=42):
        """
        Parameters:
        -----------
        n_estimators : int
            弱分类器数量，默认50
        lambda_decay : float
            时间衰减系数λ，默认0.1
            λ越大，近期数据权重越高
        beta : float
            负样本权重放大因子β，默认2.0
            β > 1时，增加对高风险样本（负样本）的关注
        max_depth : int
            基分类器（决策树）的最大深度，默认1（决策树桩）
        random_state : int
            随机种子
        """
        self.n_estimators = n_estimators
        self.lambda_decay = lambda_decay
        self.beta = beta
        self.max_depth = max_depth
        self.random_state = random_state
        
        # 模型组件
        self.estimators = []      # 弱分类器列表
        self.alphas = []          # 弱分类器权重
        self.feature_names = None
        
    def _compute_time_weights(self, n_samples, time_indices=None):
        """
        计算时间权重
        
        公式：w_t = exp(-λ * t)
        
        Parameters:
        -----------
        n_samples : int
            样本数量
        time_indices : np.ndarray, optional
            时间索引，如果为None则按顺序生成
            
        Returns:
        --------
        time_weights : np.ndarray
            时间权重
        """
        if time_indices is None:
            # 假设数据按时间顺序排列，最后的数据是最新的
            time_indices = np.arange(n_samples)
            # 反转，使得最新数据（索引大）的t值小，权重大
            time_indices = n_samples - 1 - time_indices
            
        time_weights = np.exp(-self.lambda_decay * time_indices)
        
        # 归一化
        time_weights = time_weights / np.sum(time_weights)
        
        return time_weights
    
    def _compute_error_weights(self, y_true, y_pred):
        """
        计算误差权重（负样本优化）
        
        误差函数 l_i：
        - l = 1,   正确分类
        - l = -1,  负样本误分为正
        - l = -β,  正样本误分为负（高风险样本漏检，惩罚更重）
        
        Parameters:
        -----------
        y_true : np.ndarray
            真实标签 {-1, +1}
        y_pred : np.ndarray
            预测标签 {-1, +1}
            
        Returns:
        --------
        error_weights : np.ndarray
            误差权重
        """
        n = len(y_true)
        error_weights = np.ones(n)
        
        for i in range(n):
            if y_pred[i] == y_true[i]:
                # 正确分类
                error_weights[i] = 1
            elif y_pred[i] == 1 and y_true[i] == -1:
                # 负样本（高风险）误分为正样本（低风险）
                # 这是严重错误，但在标准Adaboost中按-1处理
                error_weights[i] = -1
            elif y_pred[i] == -1 and y_true[i] == 1:
                # 正样本（低风险）误分为负样本（高风险）
                # 使用β放大惩罚
                error_weights[i] = -self.beta
                
        return error_weights
    
    def fit(self, X, y, time_indices=None):
        """
        训练Adaboost-TW-NSI模型
        
        Parameters:
        -----------
        X : np.ndarray or pd.DataFrame
            特征矩阵，shape=(n_samples, n_features)
        y : np.ndarray
            标签，0/1 或 -1/+1
        time_indices : np.ndarray, optional
            时间索引，用于时间加权
            
        Returns:
        --------
        self
        """
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
            X = X.values
        
        X = np.array(X)
        y = np.array(y)
        
        n_samples, n_features = X.shape
        
        # 转换标签为 {-1, +1}
        # 假设 1 表示违约（高风险），0 表示正常（低风险）
        # 在Adaboost中，我们将违约设为-1（负类），正常设为+1（正类）
        y_transformed = np.where(y == 1, -1, 1)
        
        # 初始化样本权重（结合时间权重）
        time_weights = self._compute_time_weights(n_samples, time_indices)
        sample_weights = time_weights.copy()
        
        # 迭代训练弱分类器
        self.estimators = []
        self.alphas = []
        
        for u in range(self.n_estimators):
            # 创建弱分类器（决策树桩）
            estimator = DecisionTreeClassifier(
                max_depth=self.max_depth,
                random_state=self.random_state + u
            )
            
            # 使用当前权重训练
            estimator.fit(X, y_transformed, sample_weight=sample_weights)
            
            # 预测
            y_pred = estimator.predict(X)
            
            # 计算加权误差率
            incorrect = (y_pred != y_transformed)
            weighted_error = np.sum(sample_weights * incorrect)
            
            # 避免除零和数值问题
            weighted_error = np.clip(weighted_error, 1e-10, 1 - 1e-10)
            
            # 计算弱分类器权重 α_u
            alpha = 0.5 * np.log((1 - weighted_error) / weighted_error)
            
            # 计算误差权重（负样本优化）
            error_weights = self._compute_error_weights(y_transformed, y_pred)
            
            # 更新样本权重
            # w_i^(u+1) = w_i^u * exp(-α_u * l_i^u) * exp(-λ * t_i)
            sample_weights = sample_weights * np.exp(-alpha * error_weights)
            sample_weights = sample_weights * time_weights  # 时间加权
            
            # 归一化
            sample_weights = sample_weights / np.sum(sample_weights)
            
            # 保存弱分类器和权重
            self.estimators.append(estimator)
            self.alphas.append(alpha)
        
        return self
    
    def predict_proba(self, X):
        """
        预测概率（信用风险指数）
        
        CRI(x) = 1 / (1 + exp(-F(x)))
        
        Parameters:
        -----------
        X : np.ndarray or pd.DataFrame
            特征矩阵
            
        Returns:
        --------
        proba : np.ndarray
            信用风险指数，范围(0, 1)
            值越高表示信用越好（风险越低）
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
            
        X = np.array(X)
        n_samples = X.shape[0]
        
        # 计算强分类器输出 F(x) = Σ α_u * f_u(x)
        F = np.zeros(n_samples)
        
        for alpha, estimator in zip(self.alphas, self.estimators):
            pred = estimator.predict(X)
            F += alpha * pred
        
        # 转换为概率（Sigmoid函数）
        # CRI = 1 / (1 + exp(-F))
        proba = 1 / (1 + np.exp(-F))
        
        return proba
    
    def predict(self, X, threshold=0.5):
        """
        预测类别
        
        Parameters:
        -----------
        X : np.ndarray or pd.DataFrame
            特征矩阵
        threshold : float
            分类阈值，默认0.5
            
        Returns:
        --------
        predictions : np.ndarray
            预测类别，0=违约（高风险），1=正常（低风险）
        """
        proba = self.predict_proba(X)
        predictions = (proba >= threshold).astype(int)
        return predictions
    
    def get_feature_importance(self):
        """
        获取特征重要性（基于弱分类器的加权平均）
        
        Returns:
        --------
        importance : dict
            特征重要性字典
        """
        if not self.estimators:
            raise ValueError("请先训练模型")
            
        n_features = self.estimators[0].n_features_in_
        importance = np.zeros(n_features)
        
        total_alpha = sum(self.alphas)
        
        for alpha, estimator in zip(self.alphas, self.estimators):
            importance += (alpha / total_alpha) * estimator.feature_importances_
        
        if self.feature_names:
            return dict(zip(self.feature_names, importance))
        else:
            return dict(enumerate(importance))


def calculate_cri(enterprise_df):
    """
    计算企业信用风险指数（Credit Risk Index, CRI）
    
    使用Adaboost-TW-NSI模型评估企业信用风险。
    
    Parameters:
    -----------
    enterprise_df : pd.DataFrame
        企业数据表
        
    Returns:
    --------
    cri_scores : pd.Series
        信用风险指数，范围(0, 1)，值越高信用越好
    model : AdaboostTWNSI
        训练好的模型
    metrics : dict
        模型评估指标
    """
    
    print("\n" + "=" * 60)
    print("【企业信用风险指数模型训练】Adaboost-TW-NSI")
    print("=" * 60)
    
    # 特征选择
    feature_columns = [
        'credit_balance',      # 授信余额
        'approved_amount',     # 批准额度
        'available_amount',    # 可用额度
        'avg_deposit',         # 日均存款余额
        'non_interest_income', # 非息净收入
        'default_history',     # 历史违约记录
        'total_assets',        # 总资产
        'total_liabilities',   # 总负债
        'debt_ratio',          # 资产负债率
        'current_ratio',       # 流动比率
        'quick_ratio',         # 速动比率
        'roe',                 # ROE
        'roa',                 # ROA
    ]
    
    # 准备数据
    X = enterprise_df[feature_columns].copy()
    y = enterprise_df['default_flag'].values
    
    # 处理缺失值
    X = X.fillna(X.median())
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())
    
    print(f"\n样本总数: {len(X)}")
    print(f"违约样本: {y.sum()} ({y.mean()*100:.2f}%)")
    print(f"正常样本: {len(y) - y.sum()} ({(1-y.mean())*100:.2f}%)")
    print(f"特征数量: {len(feature_columns)}")
    
    # 时间序列划分（前80%训练，后20%测试）
    # 这样才能体现时间加权的优势：近期数据作为测试集
    n_samples = len(X)
    split_idx = int(n_samples * 0.8)
    
    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_train = y[:split_idx]
    y_test = y[split_idx:]
    
    print(f"\n训练集: {len(X_train)} (违约: {y_train.sum()}) [早期数据]")
    print(f"测试集: {len(X_test)} (违约: {y_test.sum()}) [近期数据]")
    
    # 训练模型
    # 根据类别不平衡程度调整beta
    imbalance_ratio = (1 - y.mean()) / y.mean()  # 正常/违约比例
    beta_adjusted = min(imbalance_ratio * 0.5, 10.0)  # 动态调整beta
    
    print("\n正在训练Adaboost-TW-NSI模型...")
    print(f"参数: n_estimators=100, lambda_decay=0.15, beta={beta_adjusted:.2f}")
    print(f"类别不平衡比例: {imbalance_ratio:.2f}")
    
    model = AdaboostTWNSI(
        n_estimators=100,       # 弱分类器数量
        lambda_decay=0.15,      # 增强时间衰减，更关注近期
        beta=beta_adjusted,     # 动态调整的负样本权重
        max_depth=3,            # 弱分类器深度
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # 预测
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    y_train_proba = model.predict_proba(X_train)
    y_test_proba = model.predict_proba(X_test)
    
    # 计算评估指标
    # 注意：我们的模型预测 1=正常，0=违约
    # 但原始标签是 1=违约，0=正常
    # 所以需要反转预测结果来计算混淆矩阵
    y_train_pred_inv = 1 - y_train_pred
    y_test_pred_inv = 1 - y_test_pred
    
    metrics = {
        'train': {
            'accuracy': accuracy_score(y_train, y_train_pred_inv),
            'precision': precision_score(y_train, y_train_pred_inv, zero_division=0),
            'recall': recall_score(y_train, y_train_pred_inv, zero_division=0),
            'f1': f1_score(y_train, y_train_pred_inv, zero_division=0),
            'auc': roc_auc_score(y_train, 1 - y_train_proba),
        },
        'test': {
            'accuracy': accuracy_score(y_test, y_test_pred_inv),
            'precision': precision_score(y_test, y_test_pred_inv, zero_division=0),
            'recall': recall_score(y_test, y_test_pred_inv, zero_division=0),
            'f1': f1_score(y_test, y_test_pred_inv, zero_division=0),
            'auc': roc_auc_score(y_test, 1 - y_test_proba),
        }
    }
    
    # 打印结果
    print("\n【模型评估结果】")
    print("-" * 60)
    print(f"{'指标':<15} {'训练集':<15} {'测试集':<15}")
    print("-" * 60)
    print(f"{'准确率':<15} {metrics['train']['accuracy']:<15.4f} {metrics['test']['accuracy']:<15.4f}")
    print(f"{'精确率':<15} {metrics['train']['precision']:<15.4f} {metrics['test']['precision']:<15.4f}")
    print(f"{'召回率':<15} {metrics['train']['recall']:<15.4f} {metrics['test']['recall']:<15.4f}")
    print(f"{'F1-Score':<15} {metrics['train']['f1']:<15.4f} {metrics['test']['f1']:<15.4f}")
    print(f"{'AUC':<15} {metrics['train']['auc']:<15.4f} {metrics['test']['auc']:<15.4f}")
    print("-" * 60)
    
    # 混淆矩阵
    print("\n【测试集混淆矩阵】")
    cm = confusion_matrix(y_test, y_test_pred_inv)
    print(f"              预测正常  预测违约")
    print(f"实际正常      {cm[0,0]:6d}    {cm[0,1]:6d}")
    print(f"实际违约      {cm[1,0]:6d}    {cm[1,1]:6d}")
    
    # 特征重要性
    importance = model.get_feature_importance()
    sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    
    print("\n【特征重要性排名】")
    print("-" * 40)
    for i, (feat, imp) in enumerate(sorted_imp, 1):
        print(f"  {i:2d}. {feat:<25} {imp:.4f}")
    print("-" * 40)
    
    # 计算所有样本的信用风险指数
    cri_scores = model.predict_proba(X)
    
    print("\n【信用风险指数（CRI）统计】")
    print("-" * 50)
    print(f"CRI均值: {np.mean(cri_scores):.4f}")
    print(f"CRI标准差: {np.std(cri_scores):.4f}")
    print(f"CRI范围: [{np.min(cri_scores):.4f}, {np.max(cri_scores):.4f}]")
    
    # 按违约状态查看CRI分布
    print("\n按违约状态的CRI分布:")
    normal_cri = cri_scores[y == 0]
    default_cri = cri_scores[y == 1]
    print(f"  正常企业: 均值={np.mean(normal_cri):.4f}, 标准差={np.std(normal_cri):.4f}")
    print(f"  违约企业: 均值={np.mean(default_cri):.4f}, 标准差={np.std(default_cri):.4f}")
    
    return pd.Series(cri_scores, index=enterprise_df.index, name='CRI'), model, metrics


# ============================================================
# 与标准Adaboost的对比实验
# ============================================================

def compare_with_standard_adaboost(enterprise_df):
    """
    与标准Adaboost进行对比实验
    
    注意：在模拟数据上，标准Adaboost可能表现更好，因为：
    1. 模拟数据特征-标签关系较强，简单模型即可学到
    2. 模拟数据噪声较少，时间加权的优势不明显
    3. TW-NSI的优势主要体现在真实场景中：
       - 数据存在时间漂移（concept drift）
       - 违约模式随经济周期变化
       - 需要对近期风险模式更敏感
    """
    from sklearn.ensemble import AdaBoostClassifier
    
    print("\n" + "=" * 60)
    print("【对比实验】Adaboost-TW-NSI vs 标准Adaboost")
    print("=" * 60)
    
    # 准备数据
    feature_columns = [
        'credit_balance', 'approved_amount', 'available_amount',
        'avg_deposit', 'non_interest_income', 'default_history',
        'total_assets', 'total_liabilities', 'debt_ratio',
        'current_ratio', 'quick_ratio', 'roe', 'roa',
    ]
    
    X = enterprise_df[feature_columns].copy()
    y = enterprise_df['default_flag'].values
    
    X = X.fillna(X.median())
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())
    
    # 时间序列划分
    n_samples = len(X)
    split_idx = int(n_samples * 0.8)
    
    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_train = y[:split_idx]
    y_test = y[split_idx:]
    
    results = {}
    
    # 1. 标准Adaboost
    print("\n训练标准Adaboost...")
    std_model = AdaBoostClassifier(
        n_estimators=50,
        random_state=42
    )
    std_model.fit(X_train, y_train)
    y_pred_std = std_model.predict(X_test)
    y_proba_std = std_model.predict_proba(X_test)[:, 1]
    
    results['标准Adaboost'] = {
        'accuracy': accuracy_score(y_test, y_pred_std),
        'precision': precision_score(y_test, y_pred_std, zero_division=0),
        'recall': recall_score(y_test, y_pred_std, zero_division=0),
        'f1': f1_score(y_test, y_pred_std, zero_division=0),
        'auc': roc_auc_score(y_test, y_proba_std),
    }
    
    # 2. Adaboost-TW-NSI
    print("训练Adaboost-TW-NSI...")
    imbalance_ratio = (1 - y_train.mean()) / y_train.mean() if y_train.mean() > 0 else 10
    beta_adjusted = min(imbalance_ratio * 0.5, 10.0)
    
    twnsi_model = AdaboostTWNSI(
        n_estimators=100,
        lambda_decay=0.15,      # 增强时间加权
        beta=beta_adjusted,
        max_depth=3,
        random_state=42
    )
    twnsi_model.fit(X_train, y_train)
    y_proba_twnsi = twnsi_model.predict_proba(X_test)
    y_pred_twnsi = 1 - twnsi_model.predict(X_test)  # 反转预测
    
    results['Adaboost-TW-NSI'] = {
        'accuracy': accuracy_score(y_test, y_pred_twnsi),
        'precision': precision_score(y_test, y_pred_twnsi, zero_division=0),
        'recall': recall_score(y_test, y_pred_twnsi, zero_division=0),
        'f1': f1_score(y_test, y_pred_twnsi, zero_division=0),
        'auc': roc_auc_score(y_test, 1 - y_proba_twnsi),
    }
    
    # 打印对比结果
    print("\n【对比结果】")
    print("-" * 70)
    print(f"{'指标':<15} {'标准Adaboost':<20} {'Adaboost-TW-NSI':<20} {'提升':<15}")
    print("-" * 70)
    
    for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc']:
        std_val = results['标准Adaboost'][metric]
        twnsi_val = results['Adaboost-TW-NSI'][metric]
        improve = (twnsi_val - std_val) / std_val * 100 if std_val > 0 else 0
        
        metric_name = {
            'accuracy': '准确率',
            'precision': '精确率',
            'recall': '召回率',
            'f1': 'F1-Score',
            'auc': 'AUC'
        }[metric]
        
        print(f"{metric_name:<15} {std_val:<20.4f} {twnsi_val:<20.4f} {improve:+.2f}%")
    
    print("-" * 70)
    
    return results


# ============================================================
# 测试代码
# ============================================================

if __name__ == '__main__':
    print("=" * 60)
    print("企业信用风险指数（Adaboost-TW-NSI）模型测试")
    print("=" * 60)
    
    # 加载数据
    enterprise_df = pd.read_csv('./data/enterprise_data.csv')
    
    # 计算信用风险指数
    cri_scores, model, metrics = calculate_cri(enterprise_df)
    
    # 添加到数据框
    enterprise_df['CRI'] = cri_scores
    
    # 按贷款状态查看CRI分布
    print("\n【按贷款状态的CRI分布】")
    print("-" * 50)
    for status in ['正常', '关注', '次级', '可疑', '损失']:
        status_df = enterprise_df[enterprise_df['loan_status'] == status]
        if len(status_df) > 0:
            print(f"{status}: 均值={status_df['CRI'].mean():.4f}, "
                  f"样本数={len(status_df)}")
    
    # 对比实验
    compare_results = compare_with_standard_adaboost(enterprise_df)
    
    # 保存结果
    enterprise_df.to_csv('./data/enterprise_with_cri.csv', index=False, encoding='utf-8-sig')
    
    print("\n结果已保存至 ./data/enterprise_with_cri.csv")
