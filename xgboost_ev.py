"""
多源异构数据统计建模：信贷潜力综合评估
企业价值指数模块（XGBoost）

本模块实现：
1. 基于XGBoost预测托宾Q值
2. 利用上市企业数据训练，推广到非上市企业
3. 生成企业价值指数（EV）

理论基础：
- 托宾Q值 = 企业市场价值 / 资产重置成本
- Q > 1：市场价值高于重置成本，企业具有增长潜力
- Q < 1：市场对企业未来预期保守

损失函数说明（论文补充内容）：
- XGBoost目标函数：L = Σl(y_i, ŷ_i) + ΣΩ(f_k)
- 损失函数 l：采用平方损失（回归任务）
  l(y, ŷ) = (y - ŷ)²
- 正则化项 Ω：控制模型复杂度
  Ω(f) = γT + (1/2)λ||w||²
  其中 T 为叶子节点数，w 为叶子权重，γ和λ为正则化参数
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# 尝试导入xgboost，如果没有则使用sklearn的GradientBoosting作为替代
try:
    import xgboost as xgb
    USE_XGBOOST = True
    print("使用 XGBoost 库")
except ImportError:
    from sklearn.ensemble import GradientBoostingRegressor
    USE_XGBOOST = False
    print("XGBoost未安装，使用sklearn.GradientBoostingRegressor替代")
    print("建议安装: pip3 install xgboost")


class EnterpriseValueModel:
    """
    企业价值评估模型
    
    使用XGBoost（或GradientBoosting）基于财务特征预测托宾Q值。
    
    模型特点：
    1. 基于上市企业数据训练
    2. 迁移到非上市企业进行预测
    3. 非线性建模能力强，适合高维特征
    
    XGBoost损失函数详解：
    --------------------------------------------------
    目标函数：L(φ) = Σl(y_i, ŷ_i) + ΣΩ(f_k)
    
    1. 损失函数部分 l(y_i, ŷ_i)：
       - 回归任务采用平方损失：l = (y - ŷ)²
       - 一阶导数（梯度）：g_i = ∂l/∂ŷ = 2(ŷ - y)
       - 二阶导数（Hessian）：h_i = ∂²l/∂ŷ² = 2
    
    2. 正则化部分 Ω(f)：
       - Ω(f) = γT + (1/2)λΣw_j²
       - T：叶子节点数量（控制树的复杂度）
       - w_j：第j个叶子节点的权重
       - γ：叶子节点数惩罚系数
       - λ：L2正则化系数
    
    3. 二阶泰勒展开优化：
       - L^(t) ≈ Σ[g_i·f_t(x_i) + (1/2)h_i·f_t(x_i)²] + Ω(f_t)
       - 最优叶子权重：w_j* = -G_j/(H_j + λ)
       - 最优目标值：L* = -(1/2)Σ[G_j²/(H_j + λ)] + γT
       其中 G_j = Σg_i, H_j = Σh_i（属于叶子j的样本）
    --------------------------------------------------
    """
    
    def __init__(self, use_grid_search=True):
        """
        Parameters:
        -----------
        use_grid_search : bool
            是否使用网格搜索优化超参数
        """
        self.use_grid_search = use_grid_search
        self.model = None
        self.feature_columns = None
        self.feature_importance = None
        self.best_params = None
        
    def _get_feature_columns(self):
        """定义用于预测的财务特征"""
        return [
            'total_assets',       # 总资产
            'total_liabilities',  # 总负债
            'net_assets',         # 净资产
            'revenue',            # 营业收入
            'net_profit',         # 净利润
            'roe',                # 净资产收益率
            'roa',                # 总资产收益率
            'debt_ratio',         # 资产负债率
            'current_ratio',      # 流动比率
            'quick_ratio',        # 速动比率
            'rd_ratio',           # 研发投入占比
            'revenue_growth',     # 营收增长率
            'company_age',        # 企业年龄
        ]
    
    def _prepare_data(self, df, is_training=True):
        """
        准备训练/预测数据
        
        Parameters:
        -----------
        df : pd.DataFrame
            企业数据
        is_training : bool
            是否为训练模式（训练模式需要y标签）
            
        Returns:
        --------
        X : np.ndarray
            特征矩阵
        y : np.ndarray or None
            目标变量（仅训练模式）
        """
        self.feature_columns = self._get_feature_columns()
        
        X = df[self.feature_columns].copy()
        
        # 处理缺失值和异常值
        X = X.fillna(X.median())
        
        # 处理无穷值
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median())
        
        if is_training:
            y = df['tobin_q'].values
            return X.values, y
        else:
            return X.values, None
    
    def fit(self, df_listed):
        """
        使用上市企业数据训练模型
        
        Parameters:
        -----------
        df_listed : pd.DataFrame
            上市企业数据（包含tobin_q列）
            
        Returns:
        --------
        self
        metrics : dict
            训练评估指标
        """
        print("\n" + "=" * 60)
        print("【企业价值指数模型训练】")
        print("=" * 60)
        
        # 准备数据
        X, y = self._prepare_data(df_listed, is_training=True)
        
        print(f"\n训练样本数: {len(X)}")
        print(f"特征数量: {len(self.feature_columns)}")
        print(f"托宾Q值范围: [{y.min():.4f}, {y.max():.4f}]")
        
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print(f"训练集: {len(X_train)}, 测试集: {len(X_test)}")
        
        # 模型训练
        if USE_XGBOOST:
            self._fit_xgboost(X_train, y_train, X_test, y_test)
        else:
            self._fit_sklearn(X_train, y_train, X_test, y_test)
        
        # 评估模型
        metrics = self._evaluate(X_train, y_train, X_test, y_test)
        
        # 特征重要性
        self._get_feature_importance()
        
        return self, metrics
    
    def _fit_xgboost(self, X_train, y_train, X_test, y_test):
        """使用XGBoost训练"""
        
        if self.use_grid_search:
            print("\n正在进行网格搜索优化超参数...")
            
            param_grid = {
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.05, 0.1],
                'n_estimators': [100, 200],
                'min_child_weight': [1, 3],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0],
            }
            
            base_model = xgb.XGBRegressor(
                objective='reg:squarederror',  # 平方损失
                reg_alpha=0,      # L1正则化
                reg_lambda=1,     # L2正则化
                random_state=42
            )
            
            # 使用较小的参数网格进行快速搜索
            param_grid_small = {
                'max_depth': [3, 5],
                'learning_rate': [0.05, 0.1],
                'n_estimators': [100, 200],
            }
            
            grid_search = GridSearchCV(
                base_model, 
                param_grid_small,
                cv=3,
                scoring='r2',
                n_jobs=-1
            )
            
            grid_search.fit(X_train, y_train)
            self.model = grid_search.best_estimator_
            self.best_params = grid_search.best_params_
            
            print(f"最优参数: {self.best_params}")
            
        else:
            # 使用默认参数
            self.model = xgb.XGBRegressor(
                objective='reg:squarederror',
                max_depth=5,
                learning_rate=0.1,
                n_estimators=100,
                min_child_weight=1,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0,
                reg_lambda=1,
                random_state=42
            )
            self.model.fit(X_train, y_train)
    
    def _fit_sklearn(self, X_train, y_train, X_test, y_test):
        """使用sklearn的GradientBoosting训练（备选方案）"""
        
        if self.use_grid_search:
            print("\n正在进行网格搜索优化超参数...")
            
            param_grid = {
                'max_depth': [3, 5],
                'learning_rate': [0.05, 0.1],
                'n_estimators': [100, 200],
            }
            
            base_model = GradientBoostingRegressor(
                loss='squared_error',
                random_state=42
            )
            
            grid_search = GridSearchCV(
                base_model,
                param_grid,
                cv=3,
                scoring='r2',
                n_jobs=-1
            )
            
            grid_search.fit(X_train, y_train)
            self.model = grid_search.best_estimator_
            self.best_params = grid_search.best_params_
            
            print(f"最优参数: {self.best_params}")
            
        else:
            self.model = GradientBoostingRegressor(
                loss='squared_error',
                max_depth=5,
                learning_rate=0.1,
                n_estimators=100,
                random_state=42
            )
            self.model.fit(X_train, y_train)
    
    def _evaluate(self, X_train, y_train, X_test, y_test):
        """评估模型性能"""
        
        # 预测
        y_train_pred = self.model.predict(X_train)
        y_test_pred = self.model.predict(X_test)
        
        # 计算指标
        metrics = {
            'train': {
                'R2': r2_score(y_train, y_train_pred),
                'MSE': mean_squared_error(y_train, y_train_pred),
                'MAE': mean_absolute_error(y_train, y_train_pred),
            },
            'test': {
                'R2': r2_score(y_test, y_test_pred),
                'MSE': mean_squared_error(y_test, y_test_pred),
                'MAE': mean_absolute_error(y_test, y_test_pred),
            }
        }
        
        print("\n【模型评估结果】")
        print("-" * 50)
        print(f"{'指标':<15} {'训练集':<15} {'测试集':<15}")
        print("-" * 50)
        print(f"{'R² 值':<15} {metrics['train']['R2']:<15.4f} {metrics['test']['R2']:<15.4f}")
        print(f"{'MSE':<15} {metrics['train']['MSE']:<15.4f} {metrics['test']['MSE']:<15.4f}")
        print(f"{'MAE':<15} {metrics['train']['MAE']:<15.4f} {metrics['test']['MAE']:<15.4f}")
        print("-" * 50)
        
        return metrics
    
    def _get_feature_importance(self):
        """获取特征重要性"""
        
        if USE_XGBOOST:
            importance = self.model.feature_importances_
        else:
            importance = self.model.feature_importances_
            
        self.feature_importance = dict(zip(self.feature_columns, importance))
        
        # 排序
        sorted_importance = sorted(
            self.feature_importance.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        print("\n【特征重要性排名】")
        print("-" * 40)
        for i, (feat, imp) in enumerate(sorted_importance, 1):
            print(f"  {i:2d}. {feat:<20} {imp:.4f}")
        print("-" * 40)
        
        return self.feature_importance
    
    def predict(self, df):
        """
        预测托宾Q值
        
        Parameters:
        -----------
        df : pd.DataFrame
            企业数据
            
        Returns:
        --------
        predictions : np.ndarray
            预测的托宾Q值
        """
        if self.model is None:
            raise ValueError("请先调用fit方法训练模型")
            
        X, _ = self._prepare_data(df, is_training=False)
        predictions = self.model.predict(X)
        
        # 限制预测值范围
        predictions = np.clip(predictions, 0.1, 10.0)
        
        return predictions


def calculate_ev(enterprise_df):
    """
    计算企业价值指数（Enterprise Value, EV）
    
    使用XGBoost模型预测所有企业的托宾Q值作为企业价值指数。
    
    流程：
    1. 使用上市企业数据训练模型
    2. 预测非上市企业的托宾Q值
    3. 生成统一的企业价值指数
    
    Parameters:
    -----------
    enterprise_df : pd.DataFrame
        企业数据表
        
    Returns:
    --------
    ev_scores : pd.Series
        企业价值指数
    model : EnterpriseValueModel
        训练好的模型
    metrics : dict
        模型评估指标
    """
    
    # 分离上市和非上市企业
    listed_df = enterprise_df[enterprise_df['is_listed'] == 1].copy()
    unlisted_df = enterprise_df[enterprise_df['is_listed'] == 0].copy()
    
    print(f"\n上市企业数量: {len(listed_df)}")
    print(f"非上市企业数量: {len(unlisted_df)}")
    
    # 训练模型
    model = EnterpriseValueModel(use_grid_search=True)
    model, metrics = model.fit(listed_df)
    
    # 预测所有企业的托宾Q值
    all_predictions = model.predict(enterprise_df)
    
    # 对于上市企业，使用真实值与预测值的加权平均
    ev_scores = np.zeros(len(enterprise_df))
    
    for i, row in enterprise_df.iterrows():
        if row['is_listed'] == 1 and not np.isnan(row['tobin_q']):
            # 上市企业：80%真实值 + 20%预测值（平滑处理）
            ev_scores[i] = 0.8 * row['tobin_q'] + 0.2 * all_predictions[i]
        else:
            # 非上市企业：使用预测值
            ev_scores[i] = all_predictions[i]
    
    # 标准化到[0, 1]区间
    ev_min, ev_max = ev_scores.min(), ev_scores.max()
    ev_normalized = (ev_scores - ev_min) / (ev_max - ev_min)
    
    print("\n【企业价值指数（EV）统计】")
    print("-" * 50)
    print(f"原始托宾Q - 均值: {np.mean(ev_scores):.4f}, 标准差: {np.std(ev_scores):.4f}")
    print(f"标准化EV - 均值: {np.mean(ev_normalized):.4f}, 标准差: {np.std(ev_normalized):.4f}")
    print(f"标准化EV - 范围: [{np.min(ev_normalized):.4f}, {np.max(ev_normalized):.4f}]")
    
    return pd.Series(ev_normalized, index=enterprise_df.index, name='EV'), model, metrics


# ============================================================
# 测试代码
# ============================================================

if __name__ == '__main__':
    print("=" * 60)
    print("企业价值指数（XGBoost）模型测试")
    print("=" * 60)
    
    # 加载数据
    enterprise_df = pd.read_csv('./data/enterprise_data.csv')
    
    # 计算企业价值指数
    ev_scores, model, metrics = calculate_ev(enterprise_df)
    
    # 添加到数据框
    enterprise_df['EV'] = ev_scores
    
    # 按企业规模查看EV分布
    print("\n【按企业规模的EV分布】")
    print("-" * 40)
    for scale in ['大型', '中型', '小型', '微型']:
        scale_ev = enterprise_df[enterprise_df['scale'] == scale]['EV']
        print(f"{scale}企业: 均值={scale_ev.mean():.4f}, 标准差={scale_ev.std():.4f}")
    
    # 按上市状态查看EV分布
    print("\n【按上市状态的EV分布】")
    print("-" * 40)
    for status, label in [(1, '上市企业'), (0, '非上市企业')]:
        status_ev = enterprise_df[enterprise_df['is_listed'] == status]['EV']
        print(f"{label}: 均值={status_ev.mean():.4f}, 标准差={status_ev.std():.4f}")
    
    # 保存结果
    enterprise_df.to_csv('./data/enterprise_with_ev.csv', index=False, encoding='utf-8-sig')
    
    print("\n结果已保存至 ./data/enterprise_with_ev.csv")
