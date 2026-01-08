# 多源异构数据统计建模：信贷潜力综合评估

## 代码说明

本代码库实现了论文《多源异构数据统计建模：信贷潜力综合评估》中的所有模型和方法，数据为模拟数据。

## 文件结构

```
├── data_generator.py      # 模拟数据生成器
├── entropy_weight.py      # 熵权法与TOPSIS方法
├── xgboost_ev.py          # XGBoost企业价值指数（EV）
├── adaboost_cri.py        # Adaboost-TW-NSI信用风险指数（CRI）
├── industry_index.py      # 行业指标（III、EPI）
├── cpci_calculator.py     # 综合信贷潜力指数（CPCI）整合
└── data/                  # 数据目录
    ├── enterprise_data.csv
    ├── industry_electricity.csv
    ├── regional_data.csv
    └── enterprise_cpci.csv (输出)
```

## 快速开始

### 1. 安装依赖

```bash
pip install numpy pandas scikit-learn xgboost scipy
```

### 2. 生成模拟数据

```bash
python data_generator.py
```

### 3. 运行各模块（可选，用于单独测试）

```bash
# 熵权法（EDP、RDP）
python entropy_weight.py

# 企业价值指数（XGBoost）
python xgboost_ev.py

# 信用风险指数（Adaboost-TW-NSI）
python adaboost_cri.py

# 行业指标（复杂网络+SVM）
python industry_index.py
```

### 4. 运行完整CPCI计算

```bash
python cpci_calculator.py
```

## 模型说明

### 1. 企业信用风险指数（CRI）- Adaboost-TW-NSI

**损失函数**：指数损失 L(y, F(x)) = exp(-y·F(x))

**权重更新**（改进版）：
```
w_i^(u+1) = w_i^u * exp(-α_u * l_i^u) * exp(-λ * t_i)
```
- λ：时间衰减系数（Time Weighted）
- β：负样本权重放大因子（Negative Sample Improved）

### 2. 企业价值指数（EV）- XGBoost

**目标函数**：
```
L = Σl(y_i, ŷ_i) + ΣΩ(f_k)
```
- 损失函数 l：平方损失 (y - ŷ)²
- 正则化项 Ω(f) = γT + (1/2)λ||w||²

### 3. 行业用电景气指数（EPI）- Farm Predict + SVM

**Farm Predict**：
```
X = F·B' + U（因子分解）
目标：min_{β,γ} L_n(y, Fγ + Uβ) + λR_n(β)
```

**SVM**（RBF核）：
```
min (1/2)||w||² + C·Σξ_i
K(x_i, x_j) = exp(-γ||x_i - x_j||²)
```

### 4. 综合信贷潜力指数（CPCI）- 熵权法 + TOPSIS

**熵权法**计算权重：
```
H_j = -k * Σ(p_ij * ln(p_ij))  # 信息熵
w_j = (1 - H_j) / Σ(1 - H_j)   # 权重
```

**TOPSIS**计算得分：
```
CPCI_i = d_i^- / (d_i^+ + d_i^-)
```

## 输出说明

| 指标 | 范围 | 含义 |
|------|------|------|
| CRI | [0, 1] | 值越高，信用越好 |
| EV | [0, 1] | 值越高，企业价值越高 |
| EDP | [0, 1] | 值越高，发展潜力越大 |
| III | [0, 1] | 值越高，行业越重要 |
| EPI | [0, 1] | 值越高，行业越景气 |
| RDP | [0, 1] | 值越高，区域潜力越大 |
| CPCI | [0, 1] | 值越高，信贷潜力越大 |

## 引用

如果使用本代码，请引用：

```
王舜尧, 钱江, 葛鹏, 周阳, 陈钊. 多源异构数据统计建模：信贷潜力综合评估[J]. 数理统计与管理, 2026.
```

## 联系方式

如有问题，请联系作者：wangshunyao@fudan.edu.cn
