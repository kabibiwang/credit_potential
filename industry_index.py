"""
多源异构数据统计建模：信贷潜力综合评估
行业指标模块

本模块实现：
1. 行业重要性指数（Industry Importance Index, III）
   - 基于复杂网络模型
   - 格兰杰因果检验确定行业间因果关系
   - 交叉相关函数量化关联强度
   - 网络出度作为重要性指标

2. 行业用电景气指数（Electricity Prosperity Index, EPI）
   - Farm Predict方法进行因子分解
   - SVM进行非线性分类
   - 预测行业是否处于景气状态（用电量前30%）

理论基础与损失函数说明：
--------------------------------------------------
1. 格兰杰因果检验：
   Y_t = Σα_i·Y_{t-i} + Σβ_j·X_{t-j} + ε_t
   检验H0: β_1 = β_2 = ... = β_m = 0
   使用F统计量判断因果关系

2. Farm Predict因子模型：
   X = F·B' + U
   其中 F 为公共因子，U 为特异成分
   目标函数：min_{β,γ} L_n(y, Fγ + Uβ) + λR_n(β)
   - L_n: 损失函数（分类用交叉熵，回归用MSE）
   - R_n: 正则化项（Lasso或Ridge）

3. SVM分类器：
   目标函数：min_{w,b,ξ} (1/2)||w||² + C·Σξ_i
   约束：y_i(w'φ(x_i) + b) ≥ 1 - ξ_i, ξ_i ≥ 0
   - 采用RBF核：K(x_i, x_j) = exp(-||x_i-x_j||²/(2σ²))
   - Hinge损失：max(0, 1 - y·f(x))
--------------------------------------------------
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# 1. 行业重要性指数（III）- 复杂网络模型
# ============================================================

class IndustryNetwork:
    """
    行业复杂网络模型
    
    基于格兰杰因果检验和交叉相关函数构建行业关联网络，
    计算网络出度作为行业重要性指数。
    
    方法：
    1. 格兰杰因果检验：确定行业间是否存在因果关系
    2. 交叉相关函数：量化行业间关联强度
    3. 复杂网络分析：计算节点出度作为重要性指标
    """
    
    def __init__(self, max_lag=4, significance_level=0.05):
        """
        Parameters:
        -----------
        max_lag : int
            格兰杰因果检验的最大滞后阶数
        significance_level : float
            显著性水平
        """
        self.max_lag = max_lag
        self.significance_level = significance_level
        self.adjacency_matrix = None
        self.weight_matrix = None
        self.industries = None
        
    def granger_causality_test(self, x, y, max_lag):
        """
        简化版格兰杰因果检验
        
        检验 x 是否是 y 的格兰杰原因
        
        模型：Y_t = Σα_i·Y_{t-i} + Σβ_j·X_{t-j} + ε_t
        H0: β_1 = β_2 = ... = β_m = 0（x不是y的格兰杰原因）
        
        Parameters:
        -----------
        x : np.ndarray
            潜在原因序列
        y : np.ndarray
            结果序列
        max_lag : int
            最大滞后阶数
            
        Returns:
        --------
        is_causal : bool
            是否存在格兰杰因果关系
        p_value : float
            检验的p值
        """
        n = len(y)
        
        if n <= 2 * max_lag + 1:
            return False, 1.0
        
        # 构建滞后矩阵
        # 受限模型：只用y的滞后
        # 非受限模型：用y和x的滞后
        
        Y = y[max_lag:]
        n_obs = len(Y)
        
        # 构建设计矩阵
        X_restricted = np.column_stack([
            y[max_lag-i-1:n-i-1] for i in range(max_lag)
        ])
        
        X_unrestricted = np.column_stack([
            X_restricted,
            *[x[max_lag-i-1:n-i-1] for i in range(max_lag)]
        ])
        
        # 添加常数项
        X_restricted = np.column_stack([np.ones(n_obs), X_restricted])
        X_unrestricted = np.column_stack([np.ones(n_obs), X_unrestricted])
        
        try:
            # OLS估计
            beta_r = np.linalg.lstsq(X_restricted, Y, rcond=None)[0]
            beta_u = np.linalg.lstsq(X_unrestricted, Y, rcond=None)[0]
            
            # 残差平方和
            RSS_r = np.sum((Y - X_restricted @ beta_r) ** 2)
            RSS_u = np.sum((Y - X_unrestricted @ beta_u) ** 2)
            
            # F统计量
            df1 = max_lag  # 约束数量
            df2 = n_obs - 2 * max_lag - 1  # 非受限模型自由度
            
            if df2 <= 0 or RSS_u <= 0:
                return False, 1.0
                
            F_stat = ((RSS_r - RSS_u) / df1) / (RSS_u / df2)
            
            # p值
            p_value = 1 - stats.f.cdf(F_stat, df1, df2)
            
            is_causal = p_value < self.significance_level
            
            return is_causal, p_value
            
        except:
            return False, 1.0
    
    def cross_correlation(self, x, y, max_lag=10):
        """
        计算交叉相关函数
        
        g_k^{xy} = (1/(n-k)) * Σ(y_t - ȳ)(x_{t+k} - x̄)
        
        Parameters:
        -----------
        x : np.ndarray
            序列x
        y : np.ndarray
            序列y
        max_lag : int
            最大滞后期
            
        Returns:
        --------
        max_corr : float
            最大绝对交叉相关值
        """
        n = len(x)
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        
        max_corr = 0
        
        for k in range(-max_lag, max_lag + 1):
            if k >= 0:
                x_lagged = x[k:]
                y_aligned = y[:n-k]
            else:
                x_lagged = x[:n+k]
                y_aligned = y[-k:]
            
            if len(x_lagged) < 2:
                continue
                
            # 计算交叉相关
            cov = np.mean((y_aligned - y_mean) * (x_lagged - x_mean))
            std_x = np.std(x_lagged)
            std_y = np.std(y_aligned)
            
            if std_x > 0 and std_y > 0:
                corr = cov / (std_x * std_y)
                if abs(corr) > abs(max_corr):
                    max_corr = corr
        
        return abs(max_corr)
    
    def build_network(self, electricity_df):
        """
        构建行业关联网络
        
        Parameters:
        -----------
        electricity_df : pd.DataFrame
            行业用电量时间序列数据
            
        Returns:
        --------
        self
        """
        self.industries = electricity_df.columns.tolist()
        n_industries = len(self.industries)
        
        print(f"\n构建行业网络...")
        print(f"行业数量: {n_industries}")
        print(f"时间序列长度: {len(electricity_df)}")
        
        # 初始化邻接矩阵和权重矩阵
        self.adjacency_matrix = np.zeros((n_industries, n_industries))
        self.weight_matrix = np.zeros((n_industries, n_industries))
        
        # 检验所有行业对
        total_pairs = n_industries * (n_industries - 1)
        significant_edges = 0
        
        for i, ind_i in enumerate(self.industries):
            for j, ind_j in enumerate(self.industries):
                if i == j:
                    continue
                
                x = electricity_df[ind_i].values
                y = electricity_df[ind_j].values
                
                # 格兰杰因果检验
                is_causal, p_value = self.granger_causality_test(x, y, self.max_lag)
                
                if is_causal:
                    self.adjacency_matrix[i, j] = 1
                    significant_edges += 1
                    
                    # 计算交叉相关作为边权重
                    weight = self.cross_correlation(x, y)
                    self.weight_matrix[i, j] = weight
        
        print(f"显著因果关系数: {significant_edges} / {total_pairs}")
        print(f"网络密度: {significant_edges / total_pairs:.4f}")
        
        return self
    
    def calculate_importance(self):
        """
        计算行业重要性指数（基于出度）
        
        OutStrength_i = Σ W̃_{ij}
        
        Returns:
        --------
        importance : pd.Series
            行业重要性指数
        """
        if self.weight_matrix is None:
            raise ValueError("请先调用build_network方法")
        
        # 归一化权重矩阵（按行）
        row_sums = np.sum(self.weight_matrix, axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)  # 避免除零
        normalized_weights = self.weight_matrix / row_sums
        
        # 计算出度（OutStrength）
        out_strength = np.sum(normalized_weights, axis=1)
        
        # 标准化到[0, 1]
        if out_strength.max() > out_strength.min():
            importance = (out_strength - out_strength.min()) / (out_strength.max() - out_strength.min())
        else:
            importance = np.ones(len(out_strength)) * 0.5
        
        return pd.Series(importance, index=self.industries, name='III')


def calculate_iii(electricity_df, aggregate_regions=True):
    """
    计算行业重要性指数（Industry Importance Index, III）
    
    Parameters:
    -----------
    electricity_df : pd.DataFrame
        行业用电量时间序列
    aggregate_regions : bool
        是否聚合区域数据
        
    Returns:
    --------
    iii_scores : pd.Series or pd.DataFrame
        行业重要性指数
    network : IndustryNetwork
        网络模型
    """
    
    print("\n" + "=" * 60)
    print("【行业重要性指数（III）计算】")
    print("=" * 60)
    
    if aggregate_regions:
        # 按行业聚合（去掉区域前缀）
        industry_data = {}
        for col in electricity_df.columns:
            # 列名格式：区域_行业
            parts = col.split('_', 1)
            if len(parts) == 2:
                industry = parts[1]
                if industry not in industry_data:
                    industry_data[industry] = []
                industry_data[industry].append(electricity_df[col])
        
        # 计算各行业的平均用电量
        aggregated_df = pd.DataFrame({
            ind: np.mean(series, axis=0) 
            for ind, series in industry_data.items()
        }, index=electricity_df.index)
        
        print(f"聚合后行业数量: {len(aggregated_df.columns)}")
        df_to_use = aggregated_df
    else:
        df_to_use = electricity_df
    
    # 构建网络
    network = IndustryNetwork(max_lag=4, significance_level=0.05)
    network.build_network(df_to_use)
    
    # 计算重要性指数
    iii_scores = network.calculate_importance()
    
    # 打印结果
    print("\n【行业重要性指数排名 Top 10】")
    print("-" * 50)
    top10 = iii_scores.nlargest(10)
    for i, (ind, score) in enumerate(top10.items(), 1):
        print(f"  {i:2d}. {ind}: {score:.4f}")
    
    print(f"\nIII均值: {iii_scores.mean():.4f}")
    print(f"III标准差: {iii_scores.std():.4f}")
    
    return iii_scores, network


# ============================================================
# 2. 行业用电景气指数（EPI）- Farm Predict + SVM
# ============================================================

class FarmPredict:
    """
    Farm Predict因子调整正则化模型
    
    将高维协变量分解为公共因子和特异成分：
    X = F·B' + U
    
    然后基于因子和特异成分进行预测：
    y = F·γ + U·β + ε
    
    目标函数：
    min_{β,γ} L_n(y, Fγ + Uβ) + λR_n(β)
    
    其中：
    - L_n: 损失函数
    - R_n: 正则化项（默认L1-Lasso）
    - λ: 正则化参数
    """
    
    def __init__(self, n_factors=3):
        """
        Parameters:
        -----------
        n_factors : int
            公共因子数量
        """
        self.n_factors = n_factors
        self.pca = None
        self.scaler = None
        
    def fit_transform(self, X):
        """
        拟合并转换数据
        
        Parameters:
        -----------
        X : np.ndarray
            原始特征矩阵
            
        Returns:
        --------
        F : np.ndarray
            公共因子矩阵
        U : np.ndarray
            特异成分矩阵
        """
        # 标准化
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # PCA提取公共因子
        n_components = min(self.n_factors, X.shape[1], X.shape[0])
        self.pca = PCA(n_components=n_components)
        F = self.pca.fit_transform(X_scaled)
        
        # 特异成分 = 原始数据 - 因子重构
        X_reconstructed = self.pca.inverse_transform(F)
        U = X_scaled - X_reconstructed
        
        return F, U
    
    def transform(self, X):
        """
        转换新数据
        """
        X_scaled = self.scaler.transform(X)
        F = self.pca.transform(X_scaled)
        X_reconstructed = self.pca.inverse_transform(F)
        U = X_scaled - X_reconstructed
        
        return F, U


class ElectricityProsperityModel:
    """
    行业用电景气指数模型
    
    结合Farm Predict和SVM预测行业是否处于景气状态
    （用电量是否在行业前30%）
    
    SVM目标函数：
    min_{w,b,ξ} (1/2)||w||² + C·Σξ_i
    s.t. y_i(w'φ(x_i) + b) ≥ 1 - ξ_i, ξ_i ≥ 0
    
    采用RBF核函数：
    K(x_i, x_j) = exp(-γ||x_i - x_j||²)
    其中 γ = 1/(2σ²)
    """
    
    def __init__(self, n_factors=3, top_percentile=0.3):
        """
        Parameters:
        -----------
        n_factors : int
            Farm Predict的因子数量
        top_percentile : float
            景气阈值（前30%）
        """
        self.n_factors = n_factors
        self.top_percentile = top_percentile
        self.farm = None
        self.svm = None
        self.feature_scaler = None
        
    def _create_features(self, electricity_series, window_size=4):
        """
        创建特征（滞后值 + 工作天数等）
        
        Parameters:
        -----------
        electricity_series : pd.Series
            单个行业的用电量时间序列
        window_size : int
            滞后窗口大小
            
        Returns:
        --------
        X : np.ndarray
            特征矩阵
        y : np.ndarray
            标签（是否在前30%）
        valid_idx : np.ndarray
            有效样本的索引
        """
        values = electricity_series.values
        n = len(values)
        
        # 计算景气阈值（前30%）
        threshold = np.percentile(values, 100 * (1 - self.top_percentile))
        
        features = []
        labels = []
        valid_indices = []
        
        for t in range(window_size, n):
            # 滞后特征
            lag_features = values[t-window_size:t]
            
            # 移动平均
            ma = np.mean(lag_features)
            
            # 移动标准差
            ms = np.std(lag_features)
            
            # 变化率
            if values[t-1] > 0:
                change_rate = (values[t-1] - values[t-2]) / values[t-1]
            else:
                change_rate = 0
            
            # 组合特征
            feat = np.concatenate([
                lag_features,
                [ma, ms, change_rate]
            ])
            
            features.append(feat)
            labels.append(1 if values[t] >= threshold else 0)
            valid_indices.append(t)
        
        return np.array(features), np.array(labels), np.array(valid_indices)
    
    def fit(self, electricity_df, industry_col):
        """
        训练模型
        
        Parameters:
        -----------
        electricity_df : pd.DataFrame
            用电量数据
        industry_col : str
            目标行业列名
        """
        # 创建特征
        X, y, _ = self._create_features(electricity_df[industry_col])
        
        if len(X) < 20:
            raise ValueError("样本数量不足")
        
        # Farm Predict分解
        self.farm = FarmPredict(n_factors=self.n_factors)
        F, U = self.farm.fit_transform(X)
        
        # 合并因子和特异成分作为SVM输入
        X_combined = np.hstack([F, U])
        
        # 标准化
        self.feature_scaler = StandardScaler()
        X_scaled = self.feature_scaler.fit_transform(X_combined)
        
        # 训练SVM
        self.svm = SVC(
            kernel='rbf',      # RBF核
            C=1.0,             # 正则化参数
            gamma='scale',     # 核函数参数
            probability=True,  # 输出概率
            random_state=42
        )
        self.svm.fit(X_scaled, y)
        
        return self
    
    def predict_proba(self, electricity_df, industry_col):
        """
        预测景气概率
        
        Parameters:
        -----------
        electricity_df : pd.DataFrame
            用电量数据
        industry_col : str
            目标行业列名
            
        Returns:
        --------
        proba : np.ndarray
            景气概率
        valid_idx : np.ndarray
            有效索引
        """
        X, _, valid_idx = self._create_features(electricity_df[industry_col])
        
        F, U = self.farm.transform(X)
        X_combined = np.hstack([F, U])
        X_scaled = self.feature_scaler.transform(X_combined)
        
        proba = self.svm.predict_proba(X_scaled)[:, 1]
        
        return proba, valid_idx


def calculate_epi(electricity_df, aggregate_regions=True):
    """
    计算行业用电景气指数（Electricity Prosperity Index, EPI）
    
    Parameters:
    -----------
    electricity_df : pd.DataFrame
        行业用电量时间序列
    aggregate_regions : bool
        是否聚合区域数据
        
    Returns:
    --------
    epi_scores : pd.DataFrame
        各行业最新的景气指数
    metrics : dict
        模型评估指标
    """
    
    print("\n" + "=" * 60)
    print("【行业用电景气指数（EPI）计算】")
    print("=" * 60)
    
    if aggregate_regions:
        # 按行业聚合
        industry_data = {}
        for col in electricity_df.columns:
            parts = col.split('_', 1)
            if len(parts) == 2:
                industry = parts[1]
                if industry not in industry_data:
                    industry_data[industry] = []
                industry_data[industry].append(electricity_df[col])
        
        aggregated_df = pd.DataFrame({
            ind: np.mean(series, axis=0) 
            for ind, series in industry_data.items()
        }, index=electricity_df.index)
        
        df_to_use = aggregated_df
    else:
        df_to_use = electricity_df
    
    industries = df_to_use.columns.tolist()
    print(f"行业数量: {len(industries)}")
    
    # 为每个行业训练模型并预测
    epi_results = {}
    all_metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
    
    print("\n训练行业景气指数模型...")
    
    for i, industry in enumerate(industries):
        try:
            # 创建特征和标签
            model = ElectricityProsperityModel(n_factors=3, top_percentile=0.3)
            X, y, valid_idx = model._create_features(df_to_use[industry])
            
            if len(X) < 30:
                continue
            
            # 划分训练测试集
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            # Farm Predict分解
            farm = FarmPredict(n_factors=3)
            F_train, U_train = farm.fit_transform(X_train)
            F_test, U_test = farm.transform(X_test)
            
            X_train_combined = np.hstack([F_train, U_train])
            X_test_combined = np.hstack([F_test, U_test])
            
            # 标准化
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_combined)
            X_test_scaled = scaler.transform(X_test_combined)
            
            # 训练SVM
            svm = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42)
            svm.fit(X_train_scaled, y_train)
            
            # 预测
            y_pred = svm.predict(X_test_scaled)
            y_proba = svm.predict_proba(X_test_scaled)[:, 1]
            
            # 评估
            if len(np.unique(y_test)) > 1:
                all_metrics['accuracy'].append(accuracy_score(y_test, y_pred))
                all_metrics['precision'].append(precision_score(y_test, y_pred, zero_division=0))
                all_metrics['recall'].append(recall_score(y_test, y_pred, zero_division=0))
                all_metrics['f1'].append(f1_score(y_test, y_pred, zero_division=0))
            
            # 保存最新的景气概率
            epi_results[industry] = y_proba[-1] if len(y_proba) > 0 else 0.5
            
        except Exception as e:
            epi_results[industry] = 0.5
    
    # 汇总结果
    epi_scores = pd.Series(epi_results, name='EPI')
    
    # 计算平均指标
    avg_metrics = {k: np.mean(v) if v else 0 for k, v in all_metrics.items()}
    
    print("\n【模型平均评估指标】")
    print("-" * 40)
    print(f"准确率: {avg_metrics['accuracy']:.4f}")
    print(f"精确率: {avg_metrics['precision']:.4f}")
    print(f"召回率: {avg_metrics['recall']:.4f}")
    print(f"F1-Score: {avg_metrics['f1']:.4f}")
    
    # 打印景气行业排名
    print("\n【行业景气指数排名 Top 10】")
    print("-" * 50)
    top10 = epi_scores.nlargest(10)
    for i, (ind, score) in enumerate(top10.items(), 1):
        print(f"  {i:2d}. {ind}: {score:.4f}")
    
    print(f"\nEPI均值: {epi_scores.mean():.4f}")
    print(f"EPI标准差: {epi_scores.std():.4f}")
    
    return epi_scores, avg_metrics


# ============================================================
# 测试代码
# ============================================================

if __name__ == '__main__':
    print("=" * 60)
    print("行业指标模块测试")
    print("=" * 60)
    
    # 加载数据
    electricity_df = pd.read_csv('./data/industry_electricity.csv', index_col=0, parse_dates=True)
    
    print(f"\n数据维度: {electricity_df.shape}")
    print(f"时间范围: {electricity_df.index[0]} 至 {electricity_df.index[-1]}")
    
    # 计算行业重要性指数
    iii_scores, network = calculate_iii(electricity_df)
    
    # 计算行业用电景气指数
    epi_scores, epi_metrics = calculate_epi(electricity_df)
    
    # 合并结果
    industry_indicators = pd.DataFrame({
        'III': iii_scores,
        'EPI': epi_scores
    })
    
    # 处理缺失值
    industry_indicators = industry_indicators.fillna(industry_indicators.mean())
    
    # 保存结果
    industry_indicators.to_csv('./data/industry_indicators.csv', encoding='utf-8-sig')
    
    print("\n" + "=" * 60)
    print("【行业综合指标】")
    print("=" * 60)
    print(industry_indicators.head(10))
    
    print("\n结果已保存至 ./data/industry_indicators.csv")
