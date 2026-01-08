"""
多源异构数据统计建模：信贷潜力综合评估
熵权法与TOPSIS方法模块

本模块实现：
1. 熵权法（Entropy Weight Method）- 客观赋权
2. TOPSIS方法（Technique for Order Preference by Similarity to Ideal Solution）
3. 企业发展潜力指数（EDP）计算
4. 区域发展潜力指数（RDP）计算
"""

import numpy as np
import pandas as pd


class EntropyWeight:
    """
    熵权法类
    
    通过信息熵客观确定各指标权重，避免主观赋权偏差。
    
    数学原理：
    - 信息熵越大，指标差异性越小，权重越低
    - 信息熵越小，指标差异性越大，权重越高
    
    损失函数说明（论文补充）：
    - 熵权法本身不涉及损失函数优化
    - 其核心是通过信息论中的熵值度量指标的信息量
    - 权重计算公式：w_j = (1 - H_j) / Σ(1 - H_j)
    """
    
    def __init__(self, method='minmax'):
        """
        Parameters:
        -----------
        method : str
            标准化方法，'minmax' 或 'zscore'
        """
        self.method = method
        self.weights = None
        self.entropy = None
        
    def _normalize(self, X, indicator_types=None):
        """
        数据标准化处理
        
        Parameters:
        -----------
        X : np.ndarray
            原始数据矩阵，shape=(n_samples, n_features)
        indicator_types : list, optional
            指标类型列表，'positive'（正向）或'negative'（负向）
            正向指标：值越大越好
            负向指标：值越小越好
            
        Returns:
        --------
        R : np.ndarray
            标准化后的矩阵
        """
        X = np.array(X, dtype=float)
        n, m = X.shape
        R = np.zeros_like(X)
        
        if indicator_types is None:
            indicator_types = ['positive'] * m
            
        for j in range(m):
            col = X[:, j]
            min_val = np.min(col)
            max_val = np.max(col)
            
            # 避免除零
            if max_val - min_val == 0:
                R[:, j] = 1.0 / n
            else:
                if indicator_types[j] == 'positive':
                    # 正向指标：越大越好
                    R[:, j] = (col - min_val) / (max_val - min_val)
                else:
                    # 负向指标：越小越好
                    R[:, j] = (max_val - col) / (max_val - min_val)
        
        # 避免出现0值（log(0)问题）
        R = np.clip(R, 1e-10, 1.0)
        
        return R
    
    def _calculate_entropy(self, R):
        """
        计算各指标的信息熵
        
        公式：H_j = -k * Σ(p_ij * ln(p_ij))
        其中 k = 1/ln(n)，确保 H_j ∈ [0, 1]
        
        Parameters:
        -----------
        R : np.ndarray
            标准化后的矩阵
            
        Returns:
        --------
        H : np.ndarray
            各指标的信息熵
        """
        n, m = R.shape
        k = 1.0 / np.log(n)
        
        # 计算比重 p_ij
        col_sums = np.sum(R, axis=0)
        P = R / col_sums
        
        # 避免log(0)
        P = np.clip(P, 1e-10, 1.0)
        
        # 计算信息熵
        H = -k * np.sum(P * np.log(P), axis=0)
        
        return H
    
    def fit(self, X, indicator_types=None):
        """
        拟合熵权法模型，计算权重
        
        Parameters:
        -----------
        X : np.ndarray or pd.DataFrame
            原始数据矩阵
        indicator_types : list, optional
            指标类型列表
            
        Returns:
        --------
        self
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
            
        # 标准化
        R = self._normalize(X, indicator_types)
        
        # 计算信息熵
        self.entropy = self._calculate_entropy(R)
        
        # 计算权重
        d = 1 - self.entropy  # 差异系数
        self.weights = d / np.sum(d)
        
        return self
    
    def get_weights(self):
        """返回计算得到的权重"""
        return self.weights
    
    def get_entropy(self):
        """返回各指标的信息熵"""
        return self.entropy
    
    def transform(self, X, indicator_types=None):
        """
        计算综合得分
        
        Parameters:
        -----------
        X : np.ndarray or pd.DataFrame
            原始数据矩阵
        indicator_types : list, optional
            指标类型列表
            
        Returns:
        --------
        scores : np.ndarray
            综合得分
        """
        if self.weights is None:
            raise ValueError("请先调用fit方法计算权重")
            
        if isinstance(X, pd.DataFrame):
            X = X.values
            
        R = self._normalize(X, indicator_types)
        scores = np.dot(R, self.weights)
        
        return scores
    
    def fit_transform(self, X, indicator_types=None):
        """
        拟合并计算综合得分
        """
        self.fit(X, indicator_types)
        return self.transform(X, indicator_types)


class TOPSIS:
    """
    TOPSIS方法类
    
    基于与理想解的相似性排序技术，用于多属性决策问题。
    
    数学原理：
    - 正理想解：各指标最优值的组合
    - 负理想解：各指标最劣值的组合
    - 综合得分：样本与负理想解距离 / (与正理想解距离 + 与负理想解距离)
    
    距离度量（论文补充）：
    - 采用欧氏距离：d = sqrt(Σ(v_ij - A_j)^2)
    """
    
    def __init__(self, weight_method='entropy'):
        """
        Parameters:
        -----------
        weight_method : str
            权重计算方法，'entropy'（熵权法）或 'equal'（等权重）
        """
        self.weight_method = weight_method
        self.weights = None
        self.positive_ideal = None
        self.negative_ideal = None
        
    def _normalize(self, X, indicator_types=None):
        """向量归一化"""
        X = np.array(X, dtype=float)
        n, m = X.shape
        R = np.zeros_like(X)
        
        if indicator_types is None:
            indicator_types = ['positive'] * m
            
        for j in range(m):
            col = X[:, j]
            min_val = np.min(col)
            max_val = np.max(col)
            
            if max_val - min_val == 0:
                R[:, j] = 0.5
            else:
                if indicator_types[j] == 'positive':
                    R[:, j] = (col - min_val) / (max_val - min_val)
                else:
                    R[:, j] = (max_val - col) / (max_val - min_val)
                    
        return R
    
    def fit(self, X, indicator_types=None, weights=None):
        """
        拟合TOPSIS模型
        
        Parameters:
        -----------
        X : np.ndarray or pd.DataFrame
            原始数据矩阵
        indicator_types : list, optional
            指标类型列表
        weights : np.ndarray, optional
            预设权重，如果为None则根据weight_method计算
            
        Returns:
        --------
        self
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
            
        n, m = X.shape
        
        # 标准化
        R = self._normalize(X, indicator_types)
        
        # 计算权重
        if weights is not None:
            self.weights = np.array(weights)
        elif self.weight_method == 'entropy':
            ew = EntropyWeight()
            ew.fit(X, indicator_types)
            self.weights = ew.get_weights()
        else:
            self.weights = np.ones(m) / m
            
        # 加权标准化矩阵
        V = R * self.weights
        
        # 确定正负理想解
        self.positive_ideal = np.max(V, axis=0)
        self.negative_ideal = np.min(V, axis=0)
        
        return self
    
    def transform(self, X, indicator_types=None):
        """
        计算TOPSIS得分
        
        Parameters:
        -----------
        X : np.ndarray or pd.DataFrame
            原始数据矩阵
        indicator_types : list, optional
            指标类型列表
            
        Returns:
        --------
        scores : np.ndarray
            TOPSIS得分，范围[0, 1]
        """
        if self.weights is None:
            raise ValueError("请先调用fit方法")
            
        if isinstance(X, pd.DataFrame):
            X = X.values
            
        R = self._normalize(X, indicator_types)
        V = R * self.weights
        
        # 计算与正负理想解的距离
        d_positive = np.sqrt(np.sum((V - self.positive_ideal) ** 2, axis=1))
        d_negative = np.sqrt(np.sum((V - self.negative_ideal) ** 2, axis=1))
        
        # 计算综合得分
        # CPCI = d_negative / (d_positive + d_negative)
        scores = d_negative / (d_positive + d_negative + 1e-10)
        
        return scores
    
    def fit_transform(self, X, indicator_types=None, weights=None):
        """拟合并计算得分"""
        self.fit(X, indicator_types, weights)
        return self.transform(X, indicator_types)


# ============================================================
# 企业发展潜力指数（EDP）计算
# ============================================================

def calculate_edp(enterprise_df):
    """
    计算企业发展潜力指数（Enterprise Development Potential, EDP）
    
    使用熵权法对多维指标进行综合评价，量化企业未来发展潜力。
    
    指标体系：
    - 资源潜力：研发投入占比、专利数量
    - 经验潜力：企业年龄、市场份额
    - 成长潜力：营收增长率、员工增长率
    
    Parameters:
    -----------
    enterprise_df : pd.DataFrame
        企业数据表
        
    Returns:
    --------
    edp_scores : pd.Series
        企业发展潜力指数
    weights : dict
        各指标权重
    """
    
    # 选择用于EDP计算的指标
    edp_columns = [
        'rd_ratio',          # 研发投入占比（正向）
        'patent_count',      # 专利数量（正向）
        'company_age',       # 企业年龄（正向，经验积累）
        'market_share',      # 市场份额（正向）
        'revenue_growth',    # 营收增长率（正向）
        'employee_growth',   # 员工增长率（正向）
    ]
    
    indicator_types = ['positive'] * len(edp_columns)
    
    # 提取数据
    X = enterprise_df[edp_columns].copy()
    
    # 处理缺失值
    X = X.fillna(X.median())
    
    # 使用熵权法计算
    ew = EntropyWeight()
    edp_scores = ew.fit_transform(X.values, indicator_types)
    
    # 整理权重
    weights = dict(zip(edp_columns, ew.get_weights()))
    
    print("\n【企业发展潜力指数（EDP）】")
    print("-" * 50)
    print("指标权重：")
    for col, w in weights.items():
        print(f"  {col}: {w:.4f}")
    print("-" * 50)
    print(f"EDP均值: {np.mean(edp_scores):.4f}")
    print(f"EDP标准差: {np.std(edp_scores):.4f}")
    print(f"EDP范围: [{np.min(edp_scores):.4f}, {np.max(edp_scores):.4f}]")
    
    return pd.Series(edp_scores, index=enterprise_df.index, name='EDP'), weights


# ============================================================
# 区域发展潜力指数（RDP）计算
# ============================================================

def calculate_rdp(regional_df):
    """
    计算区域发展潜力指数（Regional Development Potential, RDP）
    
    基于波特钻石理论，从多维度量化区域经济发展潜力。
    
    指标体系（基于波特钻石模型）：
    - 生产要素：人口密度、道路密度、夜间灯光指数
    - 市场需求：GDP、人均GDP
    - 相关产业：产业结构比例
    - 企业竞争：高新技术企业数量
    - 创新能力：R&D投入、专利授权数
    
    Parameters:
    -----------
    regional_df : pd.DataFrame
        区域数据表
        
    Returns:
    --------
    rdp_scores : pd.Series
        区域发展潜力指数
    weights : dict
        各指标权重
    """
    
    # 选择用于RDP计算的指标
    rdp_columns = [
        'population_density',      # 人口密度（正向）
        'road_density',            # 道路密度（正向）
        'night_light_index',       # 夜间灯光指数（正向）
        'gdp',                     # GDP（正向）
        'gdp_per_capita',          # 人均GDP（正向）
        'tertiary_industry_ratio', # 第三产业占比（正向）
        'high_tech_ratio',         # 高新技术产业占比（正向）
        'high_tech_companies',     # 高新技术企业数量（正向）
        'rd_investment',           # R&D投入（正向）
        'patent_grants',           # 专利授权数（正向）
    ]
    
    indicator_types = ['positive'] * len(rdp_columns)
    
    # 提取数据
    X = regional_df[rdp_columns].copy()
    
    # 处理缺失值
    X = X.fillna(X.median())
    
    # 使用熵权法计算
    ew = EntropyWeight()
    rdp_scores = ew.fit_transform(X.values, indicator_types)
    
    # 整理权重
    weights = dict(zip(rdp_columns, ew.get_weights()))
    
    print("\n【区域发展潜力指数（RDP）】")
    print("-" * 50)
    print("指标权重：")
    for col, w in weights.items():
        print(f"  {col}: {w:.4f}")
    print("-" * 50)
    print(f"RDP均值: {np.mean(rdp_scores):.4f}")
    print(f"RDP标准差: {np.std(rdp_scores):.4f}")
    print(f"RDP范围: [{np.min(rdp_scores):.4f}, {np.max(rdp_scores):.4f}]")
    
    return pd.Series(rdp_scores, index=regional_df.index, name='RDP'), weights


# ============================================================
# 测试代码
# ============================================================

if __name__ == '__main__':
    # 加载数据
    print("=" * 60)
    print("熵权法与TOPSIS方法测试")
    print("=" * 60)
    
    enterprise_df = pd.read_csv('./data/enterprise_data.csv')
    regional_df = pd.read_csv('./data/regional_data.csv')
    
    # 计算企业发展潜力指数
    edp_scores, edp_weights = calculate_edp(enterprise_df)
    
    # 计算区域发展潜力指数
    rdp_scores, rdp_weights = calculate_rdp(regional_df)
    
    # 展示区域排名
    print("\n【区域发展潜力排名 Top 10】")
    regional_df['RDP'] = rdp_scores
    top10 = regional_df.nlargest(10, 'RDP')[['region', 'RDP']]
    for i, (_, row) in enumerate(top10.iterrows(), 1):
        print(f"  {i}. {row['region']}: {row['RDP']:.4f}")
    
    # 展示企业EDP分布
    print("\n【企业发展潜力分布】")
    enterprise_df['EDP'] = edp_scores
    bins = [0, 0.3, 0.4, 0.5, 0.6, 1.0]
    labels = ['0.30以下', '0.30-0.40', '0.40-0.50', '0.50-0.60', '0.60以上']
    enterprise_df['EDP_bin'] = pd.cut(edp_scores, bins=bins, labels=labels)
    dist = enterprise_df['EDP_bin'].value_counts().sort_index()
    for label, count in dist.items():
        print(f"  {label}: {count} ({count/len(enterprise_df)*100:.1f}%)")
    
    # 保存结果
    enterprise_df.to_csv('./data/enterprise_with_edp.csv', index=False, encoding='utf-8-sig')
    regional_df.to_csv('./data/regional_with_rdp.csv', index=False, encoding='utf-8-sig')
    
    print("\n结果已保存！")
