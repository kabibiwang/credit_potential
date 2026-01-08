"""
多源异构数据统计建模：信贷潜力综合评估
综合信贷潜力指数模块（CPCI）

本模块实现：
整合企业、行业、区域三个层次的指标，生成综合信贷潜力指数

输入指标：
- 企业层：CRI（信用风险指数）、EV（企业价值指数）、EDP（发展潜力指数）
- 行业层：III（行业重要性指数）、EPI（用电景气指数）
- 区域层：RDP（区域发展潜力指数）

方法：
1. 熵权法：客观确定各指标权重
2. TOPSIS：计算与理想解的相对距离

CPCI计算公式：
CPCI_i = d_i^- / (d_i^+ + d_i^-)
其中 d_i^+ 为与正理想解的距离，d_i^- 为与负理想解的距离
"""

import numpy as np
import pandas as pd
import os
import sys

# 导入各模块
from entropy_weight import EntropyWeight, TOPSIS, calculate_edp, calculate_rdp
from xgboost_ev import calculate_ev
from adaboost_cri import calculate_cri
from industry_index import calculate_iii, calculate_epi


class CPCICalculator:
    """
    综合信贷潜力指数计算器
    
    整合企业、行业、区域三维数据，生成CPCI
    """
    
    def __init__(self):
        self.enterprise_indicators = None
        self.industry_indicators = None
        self.regional_indicators = None
        self.weights = None
        self.cpci_scores = None
        
    def load_and_calculate_all(self, enterprise_df, electricity_df, regional_df):
        """
        加载数据并计算所有指标
        
        Parameters:
        -----------
        enterprise_df : pd.DataFrame
            企业数据
        electricity_df : pd.DataFrame
            行业用电数据
        regional_df : pd.DataFrame
            区域数据
        """
        print("\n" + "=" * 70)
        print("【综合信贷潜力指数（CPCI）计算】")
        print("=" * 70)
        
        # ===== 1. 企业层指标 =====
        print("\n" + "-" * 50)
        print("第一部分：企业层指标计算")
        print("-" * 50)
        
        # 信用风险指数 CRI
        cri_scores, _, _ = calculate_cri(enterprise_df)
        
        # 企业价值指数 EV
        ev_scores, _, _ = calculate_ev(enterprise_df)
        
        # 企业发展潜力指数 EDP
        edp_scores, _ = calculate_edp(enterprise_df)
        
        # 整合企业层指标
        self.enterprise_indicators = pd.DataFrame({
            'enterprise_id': enterprise_df['enterprise_id'],
            'enterprise_name': enterprise_df['enterprise_name'],
            'industry': enterprise_df['industry'],
            'region': enterprise_df['region'],
            'CRI': cri_scores.values,
            'EV': ev_scores.values,
            'EDP': edp_scores.values
        })
        
        # ===== 2. 行业层指标 =====
        print("\n" + "-" * 50)
        print("第二部分：行业层指标计算")
        print("-" * 50)
        
        # 行业重要性指数 III
        iii_scores, _ = calculate_iii(electricity_df)
        
        # 行业用电景气指数 EPI
        epi_scores, _ = calculate_epi(electricity_df)
        
        # 整合行业层指标
        self.industry_indicators = pd.DataFrame({
            'III': iii_scores,
            'EPI': epi_scores
        })
        self.industry_indicators = self.industry_indicators.fillna(self.industry_indicators.mean())
        
        # ===== 3. 区域层指标 =====
        print("\n" + "-" * 50)
        print("第三部分：区域层指标计算")
        print("-" * 50)
        
        # 区域发展潜力指数 RDP
        rdp_scores, _ = calculate_rdp(regional_df)
        
        # 整合区域层指标
        self.regional_indicators = pd.DataFrame({
            'region': regional_df['region'],
            'RDP': rdp_scores.values
        })
        
        print("\n" + "-" * 50)
        print("所有基础指标计算完成")
        print("-" * 50)
        
        return self
    
    def merge_indicators(self):
        """
        将行业和区域指标匹配到企业层
        """
        print("\n正在整合多层指标...")
        
        # 为企业匹配行业指标
        enterprise_with_industry = self.enterprise_indicators.copy()
        
        # 创建行业指标映射
        industry_map_iii = dict(zip(self.industry_indicators.index, self.industry_indicators['III']))
        industry_map_epi = dict(zip(self.industry_indicators.index, self.industry_indicators['EPI']))
        
        # 匹配行业指标（使用默认值处理未匹配的行业）
        default_iii = self.industry_indicators['III'].mean()
        default_epi = self.industry_indicators['EPI'].mean()
        
        enterprise_with_industry['III'] = enterprise_with_industry['industry'].map(
            lambda x: industry_map_iii.get(x, default_iii)
        )
        enterprise_with_industry['EPI'] = enterprise_with_industry['industry'].map(
            lambda x: industry_map_epi.get(x, default_epi)
        )
        
        # 匹配区域指标
        region_map = dict(zip(self.regional_indicators['region'], self.regional_indicators['RDP']))
        default_rdp = self.regional_indicators['RDP'].mean()
        
        enterprise_with_industry['RDP'] = enterprise_with_industry['region'].map(
            lambda x: region_map.get(x, default_rdp)
        )
        
        self.merged_data = enterprise_with_industry
        
        print(f"整合完成，共 {len(self.merged_data)} 家企业")
        
        return self
    
    def calculate_cpci(self):
        """
        使用熵权法+TOPSIS计算综合信贷潜力指数
        """
        print("\n" + "=" * 70)
        print("【计算综合信贷潜力指数（CPCI）】")
        print("=" * 70)
        
        # 指标列
        indicator_columns = ['CRI', 'EV', 'EDP', 'III', 'EPI', 'RDP']
        
        # 所有指标都是正向指标（值越大越好）
        indicator_types = ['positive'] * len(indicator_columns)
        
        # 提取指标矩阵
        X = self.merged_data[indicator_columns].values
        
        # 处理缺失值
        X = np.nan_to_num(X, nan=np.nanmean(X, axis=0))
        
        print(f"\n指标矩阵维度: {X.shape}")
        print(f"指标列: {indicator_columns}")
        
        # 使用TOPSIS方法计算CPCI
        topsis = TOPSIS(weight_method='entropy')
        cpci_scores = topsis.fit_transform(X, indicator_types)
        
        # 保存结果
        self.cpci_scores = cpci_scores
        self.weights = dict(zip(indicator_columns, topsis.weights))
        self.merged_data['CPCI'] = cpci_scores
        
        # 打印权重
        print("\n【熵权法计算的指标权重】")
        print("-" * 50)
        for ind, w in sorted(self.weights.items(), key=lambda x: x[1], reverse=True):
            bar = '█' * int(w * 50)
            print(f"  {ind:<6}: {w:.4f} {bar}")
        print("-" * 50)
        
        # 打印CPCI统计
        print("\n【CPCI统计摘要】")
        print("-" * 50)
        print(f"均值: {np.mean(cpci_scores):.4f}")
        print(f"标准差: {np.std(cpci_scores):.4f}")
        print(f"最小值: {np.min(cpci_scores):.4f}")
        print(f"最大值: {np.max(cpci_scores):.4f}")
        print(f"中位数: {np.median(cpci_scores):.4f}")
        
        return self
    
    def analyze_results(self):
        """
        分析CPCI结果
        """
        print("\n" + "=" * 70)
        print("【CPCI结果分析】")
        print("=" * 70)
        
        df = self.merged_data
        
        # 按企业规模分析
        if 'scale' in df.columns or True:
            # 从原始数据获取规模信息
            pass
        
        # 按行业分析
        print("\n【按行业的CPCI均值 Top 10】")
        print("-" * 50)
        industry_cpci = df.groupby('industry')['CPCI'].mean().sort_values(ascending=False)
        for i, (ind, cpci) in enumerate(industry_cpci.head(10).items(), 1):
            print(f"  {i:2d}. {ind}: {cpci:.4f}")
        
        # 按区域分析
        print("\n【按区域的CPCI均值 Top 10】")
        print("-" * 50)
        region_cpci = df.groupby('region')['CPCI'].mean().sort_values(ascending=False)
        for i, (reg, cpci) in enumerate(region_cpci.head(10).items(), 1):
            print(f"  {i:2d}. {reg}: {cpci:.4f}")
        
        # CPCI分布
        print("\n【CPCI分布】")
        print("-" * 50)
        bins = [0, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0]
        labels = ['0.30以下', '0.30-0.40', '0.40-0.50', '0.50-0.60', '0.60-0.70', '0.70以上']
        df['CPCI_bin'] = pd.cut(df['CPCI'], bins=bins, labels=labels)
        dist = df['CPCI_bin'].value_counts().sort_index()
        for label, count in dist.items():
            pct = count / len(df) * 100
            bar = '█' * int(pct / 2)
            print(f"  {label}: {count:4d} ({pct:5.1f}%) {bar}")
        
        # 高潜力企业（CPCI > 0.7）
        print("\n【高信贷潜力企业 Top 10】")
        print("-" * 70)
        top10 = df.nlargest(10, 'CPCI')[['enterprise_name', 'industry', 'region', 'CPCI', 'CRI', 'EV', 'EDP']]
        for i, (_, row) in enumerate(top10.iterrows(), 1):
            print(f"  {i:2d}. {row['enterprise_name']} | {row['industry'][:8]} | {row['region']} | CPCI={row['CPCI']:.4f}")
        
        return self
    
    def save_results(self, output_dir='./data'):
        """
        保存结果
        """
        # 保存完整数据
        output_path = os.path.join(output_dir, 'enterprise_cpci.csv')
        self.merged_data.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"\n结果已保存至: {output_path}")
        
        # 保存权重
        weights_df = pd.DataFrame([self.weights])
        weights_path = os.path.join(output_dir, 'cpci_weights.csv')
        weights_df.to_csv(weights_path, index=False, encoding='utf-8-sig')
        print(f"权重已保存至: {weights_path}")
        
        return self


def main():
    """
    主函数：运行完整的CPCI计算流程
    """
    print("=" * 70)
    print("多源异构数据统计建模：信贷潜力综合评估")
    print("=" * 70)
    
    # 加载数据
    print("\n正在加载数据...")
    
    enterprise_df = pd.read_csv('./data/enterprise_data.csv')
    electricity_df = pd.read_csv('./data/industry_electricity.csv', index_col=0, parse_dates=True)
    regional_df = pd.read_csv('./data/regional_data.csv')
    
    print(f"企业数据: {enterprise_df.shape}")
    print(f"行业用电数据: {electricity_df.shape}")
    print(f"区域数据: {regional_df.shape}")
    
    # 创建计算器并运行
    calculator = CPCICalculator()
    
    calculator.load_and_calculate_all(enterprise_df, electricity_df, regional_df)
    calculator.merge_indicators()
    calculator.calculate_cpci()
    calculator.analyze_results()
    calculator.save_results()
    
    print("\n" + "=" * 70)
    print("【计算完成】")
    print("=" * 70)
    
    return calculator


if __name__ == '__main__':
    calculator = main()
