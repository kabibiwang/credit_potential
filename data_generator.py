"""
多源异构数据统计建模：信贷潜力综合评估
模拟数据生成模块

生成三个层次的模拟数据：
1. 企业层数据：财务指标、信用记录、违约标志等
2. 行业层数据：40个行业的用电量时间序列
3. 区域层数据：区域经济指标
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os

# 设置随机种子，保证可复现
np.random.seed(42)

# ============================================================
# 1. 企业层数据生成
# ============================================================

def generate_enterprise_data(n_enterprises=1000):
    """
    生成企业层模拟数据
    
    Parameters:
    -----------
    n_enterprises : int
        企业数量
    
    Returns:
    --------
    df : pd.DataFrame
        企业数据表
    """
    
    # 行业列表（40个行业）
    industries = [
        '农副食品加工业', '食品制造业', '酒饮料制造业', '纺织业',
        '服装制造业', '皮革制造业', '木材加工业', '家具制造业',
        '造纸业', '印刷业', '石油加工业', '化学原料制造业',
        '化学纤维制造业', '橡胶制品制造业', '塑料制品业', '非金属矿物制品业',
        '黑色金属冶炼业', '有色金属冶炼业', '金属制品业', '通用设备制造业',
        '专用设备制造业', '汽车制造业', '铁路航空制造业', '电气机械制造业',
        '计算机通信制造业', '仪器仪表制造业', '电力热力生产业', '燃气生产供应业',
        '水的生产供应业', '建筑业', '批发零售业', '交通运输业',
        '住宿餐饮业', '信息技术服务业', '金融业', '房地产业',
        '租赁商务服务业', '科学研究服务业', '教育', '卫生社会工作'
    ]
    
    # 区域列表（31个省级行政单位）
    regions = [
        '北京', '天津', '河北', '山西', '内蒙古',
        '辽宁', '吉林', '黑龙江', '上海', '江苏',
        '浙江', '安徽', '福建', '江西', '山东',
        '河南', '湖北', '湖南', '广东', '广西',
        '海南', '重庆', '四川', '贵州', '云南',
        '西藏', '陕西', '甘肃', '青海', '宁夏', '新疆'
    ]
    
    # 企业规模
    scales = ['大型', '中型', '小型', '微型']
    scale_weights = [0.1, 0.25, 0.4, 0.25]
    
    # 生成基础数据
    data = {
        'enterprise_id': [f'ENT_{i:06d}' for i in range(n_enterprises)],
        'enterprise_name': [f'企业_{i}' for i in range(n_enterprises)],
        'industry': np.random.choice(industries, n_enterprises),
        'region': np.random.choice(regions, n_enterprises),
        'scale': np.random.choice(scales, n_enterprises, p=scale_weights),
        'is_listed': np.random.choice([0, 1], n_enterprises, p=[0.85, 0.15]),
    }
    
    # 根据企业规模生成财务指标
    scale_multiplier = {'大型': 10, '中型': 5, '小型': 2, '微型': 1}
    
    # 授信余额（万元）
    data['credit_balance'] = [
        np.random.lognormal(mean=8, sigma=1) * scale_multiplier[s] 
        for s in data['scale']
    ]
    
    # 批准额度（万元）
    data['approved_amount'] = [
        cb * np.random.uniform(1.1, 1.5) for cb in data['credit_balance']
    ]
    
    # 可用额度
    data['available_amount'] = [
        aa - cb for aa, cb in zip(data['approved_amount'], data['credit_balance'])
    ]
    
    # 日均存款余额（万元）
    data['avg_deposit'] = [
        np.random.lognormal(mean=7, sigma=1.2) * scale_multiplier[s]
        for s in data['scale']
    ]
    
    # 非息净收入（万元）
    data['non_interest_income'] = [
        np.random.lognormal(mean=5, sigma=1.5) * scale_multiplier[s]
        for s in data['scale']
    ]
    
    # 历史违约记录（0-5次）
    data['default_history'] = np.random.choice(
        [0, 1, 2, 3, 4, 5], n_enterprises, 
        p=[0.7, 0.15, 0.08, 0.04, 0.02, 0.01]
    )
    
    # 贷款状态
    loan_status = ['正常', '关注', '次级', '可疑', '损失']
    status_weights = [0.85, 0.08, 0.04, 0.02, 0.01]
    data['loan_status'] = np.random.choice(loan_status, n_enterprises, p=status_weights)
    
    # 违约标志（与贷款状态、历史违约记录、财务指标相关）
    default_prob = []
    for i in range(n_enterprises):
        base_prob = 0.02
        # 贷款状态影响
        if data['loan_status'][i] in ['次级', '可疑', '损失']:
            base_prob += 0.4
        elif data['loan_status'][i] == '关注':
            base_prob += 0.15
        # 历史违约记录影响
        base_prob += data['default_history'][i] * 0.08
        # 资产负债率影响（后面会计算）
        # 先用规模作为代理
        if data['scale'][i] == '微型':
            base_prob += 0.05
        elif data['scale'][i] == '小型':
            base_prob += 0.03
        default_prob.append(min(base_prob, 0.85))
    
    data['default_flag'] = [
        np.random.choice([0, 1], p=[1-p, p]) for p in default_prob
    ]
    
    # ========== 用于企业价值评估的财务指标 ==========
    
    # 总资产（万元）
    data['total_assets'] = [
        np.random.lognormal(mean=10, sigma=1) * scale_multiplier[s]
        for s in data['scale']
    ]
    
    # 总负债（万元）
    data['total_liabilities'] = [
        ta * np.random.uniform(0.3, 0.7) for ta in data['total_assets']
    ]
    
    # 净资产
    data['net_assets'] = [
        ta - tl for ta, tl in zip(data['total_assets'], data['total_liabilities'])
    ]
    
    # 营业收入（万元）
    data['revenue'] = [
        np.random.lognormal(mean=9, sigma=1.2) * scale_multiplier[s]
        for s in data['scale']
    ]
    
    # 净利润（万元）
    data['net_profit'] = [
        rev * np.random.uniform(-0.1, 0.25) for rev in data['revenue']
    ]
    
    # ROE
    data['roe'] = [
        np if na > 0 else 0 for np, na in zip(data['net_profit'], data['net_assets'])
    ]
    data['roe'] = [np / na if na > 0 else 0 for np, na in zip(data['net_profit'], data['net_assets'])]
    
    # ROA
    data['roa'] = [np / ta if ta > 0 else 0 for np, ta in zip(data['net_profit'], data['total_assets'])]
    
    # 资产负债率
    data['debt_ratio'] = [tl / ta if ta > 0 else 0 for tl, ta in zip(data['total_liabilities'], data['total_assets'])]
    
    # 流动比率
    data['current_ratio'] = [np.random.uniform(0.5, 3.0) for _ in range(n_enterprises)]
    
    # 速动比率
    data['quick_ratio'] = [cr * np.random.uniform(0.6, 0.9) for cr in data['current_ratio']]
    
    # 【关键改进】根据违约状态调整财务指标，让违约企业特征更明显
    # 同时添加时间漂移（Concept Drift），模拟真实场景中违约模式随时间变化
    for i in range(n_enterprises):
        # 模拟时间因素：企业按时间顺序排列，后面的企业是"更近期"的
        time_factor = i / n_enterprises  # 0到1，表示时间进度
        
        if data['default_flag'][i] == 1:
            # 违约企业：财务状况更差
            # 【时间漂移】近期违约企业的特征模式略有变化（不同特征的重要性变化）
            # 早期：debt_ratio是主要特征
            # 近期：roa和current_ratio更重要
            
            # 基础恶化程度
            base_deterioration = 0.7  # 基础恶化系数
            
            # 早期违约：debt_ratio特征明显
            if time_factor < 0.5:
                data['debt_ratio'][i] = min(data['debt_ratio'][i] * 1.5 + 0.2, 0.95)
                data['current_ratio'][i] = max(data['current_ratio'][i] * 0.7, 0.4)
                data['quick_ratio'][i] = max(data['quick_ratio'][i] * 0.6, 0.25)
                data['roe'][i] = data['roe'][i] * 0.3 - 0.1
                data['roa'][i] = data['roa'][i] * 0.4 - 0.05
            # 近期违约：roa和流动性特征更重要
            else:
                data['debt_ratio'][i] = min(data['debt_ratio'][i] * 1.2 + 0.1, 0.85)  # 变化较小
                data['current_ratio'][i] = max(data['current_ratio'][i] * 0.4, 0.25)  # 变化更大
                data['quick_ratio'][i] = max(data['quick_ratio'][i] * 0.3, 0.15)
                data['roe'][i] = data['roe'][i] * 0.15 - 0.2  # 变化更大
                data['roa'][i] = data['roa'][i] * 0.1 - 0.1   # 变化更大
            
            # 共同特征
            data['avg_deposit'][i] = data['avg_deposit'][i] * (0.3 + 0.2 * time_factor)
            data['net_profit'][i] = data['net_profit'][i] * 0.2
    
    # 托宾Q值（仅上市企业有真实值）
    tobin_q = []
    for i in range(n_enterprises):
        if data['is_listed'][i] == 1:
            # 上市企业：基于财务指标生成托宾Q
            base_q = 1.0
            base_q += data['roe'][i] * 2  # ROE影响
            base_q += np.random.uniform(-0.3, 0.5)  # 随机波动
            base_q = max(0.3, min(base_q, 5.0))  # 限制范围
        else:
            base_q = np.nan  # 非上市企业无托宾Q
        tobin_q.append(base_q)
    data['tobin_q'] = tobin_q
    
    # ========== 用于企业发展潜力评估的指标 ==========
    
    # 研发投入占比
    data['rd_ratio'] = np.random.beta(2, 10, n_enterprises) * 0.3
    
    # 员工增长率
    data['employee_growth'] = np.random.normal(0.05, 0.15, n_enterprises)
    
    # 营收增长率
    data['revenue_growth'] = np.random.normal(0.1, 0.2, n_enterprises)
    
    # 市场份额（行业内）
    data['market_share'] = np.random.beta(2, 20, n_enterprises)
    
    # 专利数量
    data['patent_count'] = np.random.poisson(5, n_enterprises)
    
    # 企业年龄
    data['company_age'] = np.random.randint(1, 50, n_enterprises)
    
    df = pd.DataFrame(data)
    
    return df


# ============================================================
# 2. 行业层数据生成（用电量时间序列）
# ============================================================

def generate_industry_electricity_data(n_weeks=130):
    """
    生成行业用电量时间序列数据
    
    Parameters:
    -----------
    n_weeks : int
        时间序列长度（周数），默认130周（约2.5年）
    
    Returns:
    --------
    df : pd.DataFrame
        行业用电量数据，行为时间，列为行业
    """
    
    industries = [
        '农副食品加工业', '食品制造业', '酒饮料制造业', '纺织业',
        '服装制造业', '皮革制造业', '木材加工业', '家具制造业',
        '造纸业', '印刷业', '石油加工业', '化学原料制造业',
        '化学纤维制造业', '橡胶制品制造业', '塑料制品业', '非金属矿物制品业',
        '黑色金属冶炼业', '有色金属冶炼业', '金属制品业', '通用设备制造业',
        '专用设备制造业', '汽车制造业', '铁路航空制造业', '电气机械制造业',
        '计算机通信制造业', '仪器仪表制造业', '电力热力生产业', '燃气生产供应业',
        '水的生产供应业', '建筑业', '批发零售业', '交通运输业',
        '住宿餐饮业', '信息技术服务业', '金融业', '房地产业',
        '租赁商务服务业', '科学研究服务业', '教育', '卫生社会工作'
    ]
    
    regions = ['东部', '中部', '西部', '东北', '华南', '华北', '西南']
    
    # 生成日期索引
    start_date = datetime(2020, 6, 1)
    dates = [start_date + timedelta(weeks=i) for i in range(n_weeks)]
    
    data = {'date': dates}
    
    # 为每个行业-区域组合生成用电量时间序列
    for region in regions:
        for industry in industries:
            col_name = f'{region}_{industry}'
            
            # 基础用电量（考虑行业差异）
            if '制造' in industry or '冶炼' in industry:
                base_level = np.random.uniform(800, 1500)
            elif '服务' in industry or '金融' in industry:
                base_level = np.random.uniform(200, 500)
            else:
                base_level = np.random.uniform(400, 800)
            
            # 生成时间序列
            # 趋势项
            trend = np.linspace(0, np.random.uniform(-0.1, 0.2) * base_level, n_weeks)
            
            # 季节性（年度周期）
            seasonal = base_level * 0.1 * np.sin(np.linspace(0, 4*np.pi, n_weeks))
            
            # 随机波动
            noise = np.random.normal(0, base_level * 0.05, n_weeks)
            
            # 组合
            electricity = base_level + trend + seasonal + noise
            electricity = np.maximum(electricity, 0)  # 确保非负
            
            data[col_name] = electricity
    
    df = pd.DataFrame(data)
    df.set_index('date', inplace=True)
    
    return df


# ============================================================
# 3. 区域层数据生成
# ============================================================

def generate_regional_data():
    """
    生成区域经济指标数据
    
    Returns:
    --------
    df : pd.DataFrame
        区域经济数据
    """
    
    regions = [
        '北京', '天津', '河北', '山西', '内蒙古',
        '辽宁', '吉林', '黑龙江', '上海', '江苏',
        '浙江', '安徽', '福建', '江西', '山东',
        '河南', '湖北', '湖南', '广东', '广西',
        '海南', '重庆', '四川', '贵州', '云南',
        '西藏', '陕西', '甘肃', '青海', '宁夏', '新疆'
    ]
    
    # 区域分类（用于生成差异化数据）
    east_regions = ['北京', '天津', '上海', '江苏', '浙江', '福建', '山东', '广东', '海南']
    central_regions = ['河北', '山西', '安徽', '江西', '河南', '湖北', '湖南']
    west_regions = ['内蒙古', '广西', '重庆', '四川', '贵州', '云南', '西藏', '陕西', '甘肃', '青海', '宁夏', '新疆']
    northeast_regions = ['辽宁', '吉林', '黑龙江']
    
    data = {'region': regions}
    
    # 区域禀赋指标
    # 人口密度（人/平方公里）
    pop_density = []
    for r in regions:
        if r in east_regions:
            pop_density.append(np.random.uniform(400, 1500))
        elif r in central_regions:
            pop_density.append(np.random.uniform(200, 600))
        elif r in northeast_regions:
            pop_density.append(np.random.uniform(100, 300))
        else:
            pop_density.append(np.random.uniform(10, 200))
    data['population_density'] = pop_density
    
    # 道路密度（公里/平方公里）
    data['road_density'] = [pd * np.random.uniform(0.001, 0.003) for pd in pop_density]
    
    # 夜间灯光指数（卫星数据代理）
    light_index = []
    for r in regions:
        if r in east_regions:
            light_index.append(np.random.uniform(60, 100))
        elif r in central_regions:
            light_index.append(np.random.uniform(30, 60))
        elif r in northeast_regions:
            light_index.append(np.random.uniform(25, 50))
        else:
            light_index.append(np.random.uniform(5, 35))
    data['night_light_index'] = light_index
    
    # 区域产出能力
    # GDP（亿元）
    gdp = []
    for r in regions:
        if r in ['广东', '江苏', '山东', '浙江']:
            gdp.append(np.random.uniform(80000, 130000))
        elif r in east_regions:
            gdp.append(np.random.uniform(30000, 80000))
        elif r in central_regions:
            gdp.append(np.random.uniform(20000, 50000))
        else:
            gdp.append(np.random.uniform(2000, 25000))
    data['gdp'] = gdp
    
    # 人均GDP
    data['gdp_per_capita'] = [g / (pd * 100) * 10000 for g, pd in zip(gdp, pop_density)]
    
    # 区域产业结构
    # 第一产业占比
    data['primary_industry_ratio'] = np.random.uniform(0.03, 0.20, len(regions))
    
    # 第二产业占比
    data['secondary_industry_ratio'] = np.random.uniform(0.30, 0.50, len(regions))
    
    # 第三产业占比
    data['tertiary_industry_ratio'] = [
        1 - p - s for p, s in zip(data['primary_industry_ratio'], data['secondary_industry_ratio'])
    ]
    
    # 高新技术产业占比
    high_tech_ratio = []
    for r in regions:
        if r in ['北京', '上海', '广东', '江苏', '浙江']:
            high_tech_ratio.append(np.random.uniform(0.15, 0.30))
        elif r in east_regions:
            high_tech_ratio.append(np.random.uniform(0.08, 0.18))
        else:
            high_tech_ratio.append(np.random.uniform(0.03, 0.12))
    data['high_tech_ratio'] = high_tech_ratio
    
    # 区域创新水平
    # 高新技术企业数量
    high_tech_companies = []
    for r in regions:
        if r in ['广东', '北京', '江苏', '浙江', '上海']:
            high_tech_companies.append(np.random.randint(20000, 60000))
        elif r in east_regions:
            high_tech_companies.append(np.random.randint(5000, 20000))
        elif r in central_regions:
            high_tech_companies.append(np.random.randint(3000, 12000))
        else:
            high_tech_companies.append(np.random.randint(500, 5000))
    data['high_tech_companies'] = high_tech_companies
    
    # R&D投入（亿元）
    data['rd_investment'] = [g * np.random.uniform(0.015, 0.04) for g in gdp]
    
    # R&D人员数量（万人）
    data['rd_personnel'] = [htc * np.random.uniform(0.5, 2) / 1000 for htc in high_tech_companies]
    
    # 专利授权数
    data['patent_grants'] = [htc * np.random.uniform(3, 10) for htc in high_tech_companies]
    
    df = pd.DataFrame(data)
    
    return df


# ============================================================
# 主函数：生成所有数据并保存
# ============================================================

def generate_all_data(output_dir='./data'):
    """
    生成所有模拟数据并保存到指定目录
    """
    
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print("=" * 60)
    print("开始生成模拟数据...")
    print("=" * 60)
    
    # 1. 生成企业数据
    print("\n[1/3] 生成企业层数据...")
    enterprise_df = generate_enterprise_data(n_enterprises=1000)
    enterprise_df.to_csv(f'{output_dir}/enterprise_data.csv', index=False, encoding='utf-8-sig')
    print(f"      企业数量: {len(enterprise_df)}")
    print(f"      违约企业: {enterprise_df['default_flag'].sum()} ({enterprise_df['default_flag'].mean()*100:.2f}%)")
    print(f"      上市企业: {enterprise_df['is_listed'].sum()}")
    
    # 2. 生成行业用电数据
    print("\n[2/3] 生成行业用电量时间序列数据...")
    electricity_df = generate_industry_electricity_data(n_weeks=130)
    electricity_df.to_csv(f'{output_dir}/industry_electricity.csv', encoding='utf-8-sig')
    print(f"      时间跨度: {electricity_df.index[0].strftime('%Y-%m-%d')} 至 {electricity_df.index[-1].strftime('%Y-%m-%d')}")
    print(f"      行业-区域组合数: {len(electricity_df.columns)}")
    
    # 3. 生成区域数据
    print("\n[3/3] 生成区域经济指标数据...")
    regional_df = generate_regional_data()
    regional_df.to_csv(f'{output_dir}/regional_data.csv', index=False, encoding='utf-8-sig')
    print(f"      区域数量: {len(regional_df)}")
    
    print("\n" + "=" * 60)
    print("数据生成完成！")
    print(f"数据保存路径: {os.path.abspath(output_dir)}")
    print("=" * 60)
    
    # 返回数据供后续使用
    return enterprise_df, electricity_df, regional_df


if __name__ == '__main__':
    enterprise_df, electricity_df, regional_df = generate_all_data()
    
    # 打印数据概览
    print("\n\n【企业数据概览】")
    print(enterprise_df.head())
    print(f"\n数据维度: {enterprise_df.shape}")
    
    print("\n\n【行业用电数据概览】")
    print(electricity_df.iloc[:5, :5])
    print(f"\n数据维度: {electricity_df.shape}")
    
    print("\n\n【区域数据概览】")
    print(regional_df.head())
    print(f"\n数据维度: {regional_df.shape}")
