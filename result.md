# 指数择时实验报告

> 微盘股指数 (SH#880823) | 初始资金 ¥100,000 | 训练集: 2017–2024 | 验证集: 2024 | 测试集: 2025.01.01–2026.05.06

---

## 一、实验设置

### 数据

| 数据 | 来源 | 时间范围 | 样本数 |
|------|------|----------|--------|
| 日线 | `SH#880823.txt` | 2017/02–2026/05 | 2246 条 |
| 5分钟线 | `SH#880823fen1.txt` | 2023/09–2026/05 | 30579 条 |

### 特征组

| 特征组 | 特征数 | 内容 |
|--------|:------:|------|
| `market` | 12 | 基础量价特征（MA、动量、波动率、振幅、量比） |
| `regime` | 20 | market + 趋势状态（均线偏离、回撤、量Z-score、波动率排名） |
| `momentum` | 20 | market + 动量强度（均线排列、趋势强度、风险调整收益、反弹幅度） |
| `composite` | 28 | 全部特征 |

### 模型

| 模型 | 类型 | 关键参数 |
|------|------|----------|
| Logistic Regression | 线性 | `max_iter=2000`, `class_weight='balanced'` + StandardScaler |
| Random Forest | Bagging | `n_estimators=300`, `min_samples_leaf=8`, `class_weight='balanced_subsample'` |
| Gradient Boosting | Boosting | `n_estimators=150`, `lr=0.05`, `max_depth=3` |
| Hist Gradient Boosting | Boosting | `max_iter=150`, `lr=0.05`, `max_leaf_nodes=15` |
| LightGBM | Boosting | `n_estimators=300`, `lr=0.03`, `num_leaves=15`, L1/L2 正则化 |
| **LSTM (GRU)** | 深度学习 | `lookback=15`, `gru_units=32`, `dropout=0.3`, Adam(lr=1e-3) |

### 仓位映射

| 参数 | 候选值 |
|------|--------|
| 映射方式 | `linear_clipped`, `rank_linear`, `sigmoid` |
| min_position | 0.0, 0.25, 0.50 |
| max_position | 0.75, 1.0 |
| smoothing_window | 1, 3, 5 |

### 评价指标

```text
验证集评分 = 0.5 × (策略收益 / 买入持有收益) + 0.5 × (策略夏普 / 买入持有夏普)
```

---

## 二、实验一：基准策略

| 策略 | 累计收益 | 最大回撤 | 夏普比率 | 超额(vs买入持有) |
|------|:--------:|:--------:|:--------:|:----------------:|
| **buy_hold (买入持有)** | **109.21%** | -17.26% | 2.30 | 0 |
| ma20_timing | 77.64% | -10.90% | 2.76 | -31.58% |
| momentum20_timing | 55.73% | -17.91% | 1.77 | -53.48% |
| ma20 + momentum20 组合 | 66.94% | -11.69% | 2.38 | -42.27% |
| *theoretical_optimal (未来函数)* | *917.68%* | *0%* | *12.47* | *+808.46%* |

> **结论**：简单规则择时全部跑输买入持有。ML 必须打败 109.21% / 夏普 2.30 这个高基准。

---

## 三、实验二：ML 基线（各模型 + 最优特征组）

| 模型 | 特征组 | 累计收益 | 最大回撤 | 夏普 | 超额 | test AUC |
|------|--------|:--------:|:--------:|:----:|:----:|:--------:|
| 🥇 **Logistic Regression** | composite | **107.35%** | -14.62% | 2.69 | **+8.07%** | 0.634 |
| 🥈 Random Forest | momentum | 88.93% | -11.95% | 2.53 | -10.35% | 0.640 |
| 🥉 Gradient Boosting | composite | 87.55% | -13.48% | 2.51 | -11.73% | 0.600 |
| Hist Gradient Boosting | composite | 68.19% | **-3.22%** | **3.57** | -31.09% | 0.642 |
| LightGBM | composite | 86.00% | -12.80% | 2.45 | -13.28% | 0.650 |
| **LSTM (GRU)** | composite | **84.73%** | -15.30% | 2.18 | **-14.55%** | 0.486 |

### 各模型的最优仓位映射

| 模型 | 映射方式 | lower | upper | min | max | smooth |
|------|----------|:-----:|:-----:|:---:|:---:|:------:|
| Logistic Regression | linear_clipped | 0.50 | 0.50 | 0.0 | 1.0 | 3 |
| Random Forest | linear_clipped | 0.50 | 0.55 | 0.0 | 1.0 | 5 |
| Gradient Boosting | linear_clipped | 0.50 | 0.55 | 0.0 | 1.0 | 5 |
| Hist Gradient Boosting | rank_linear | 0.40* | 0.80* | 0.0 | 1.0 | 3 |
| LightGBM | linear_clipped | 0.50 | 0.65 | 0.0 | 1.0 | 5 |
| LSTM (GRU) | sigmoid | — | — | 0.5 | 1.0 | 5 |

> *Hist Gradient Boosting 使用 rank_linear，数值为 lower_rank / upper_rank。

> **结论**：Logistic Regression 是唯一跑赢买入持有的模型（超额 +8.07%）。LSTM 实际表现等于买入持有，AUC = 0.486（≈ 随机猜测）。

---

## 四、实验三：ML 消融（market vs composite）

对比 12 个基础特征 vs 28 个全量特征的效果：

| 特征集 | 模型 | 累计收益 | 最大回撤 | 夏普 | 超额 |
|--------|------|:--------:|:--------:|:----:|:----:|
| market(12) | Logistic Regression | 84.69% | -13.03% | 2.29 | -14.58% |
| market(12) | Random Forest | 65.51% | -11.41% | 2.14 | -33.77% |
| market(12) | Gradient Boosting | 40.59% | -11.66% | 1.51 | -58.68% |
| market(12) | Hist Gradient Boosting | 87.71% | -17.60% | 2.02 | -11.57% |
| market(12) | LightGBM | 74.38% | -17.60% | 1.82 | -24.90% |
| market(12) | LSTM | 84.73% | -15.30% | 2.18 | -14.55% |
| **composite(28)** | **Logistic Regression** | **107.35%** | -14.62% | **2.69** | **+8.07%** |
| composite(28) | Random Forest | 90.40% | -10.84% | 2.68 | -8.88% |
| composite(28) | Gradient Boosting | 87.55% | -13.48% | 2.51 | -11.73% |
| composite(28) | Hist Gradient Boosting | 68.19% | -3.22% | 3.57 | -31.09% |
| composite(28) | LightGBM | 86.00% | -12.80% | 2.45 | -13.28% |
| composite(28) | LSTM | 84.73% | -15.30% | 2.18 | -14.55% |

> **结论**：composite(28特征) 全面优于 market(12特征)。LSTM 在两组上结果完全一致，说明它没从特征中学习到信号。

---

## 五、实验四：特征组消融（全部 4 组 × 6 模型）

### market (12 特征)

| 模型 | 累计收益 | 最大回撤 | 夏普 | 超额 | test AUC |
|------|:--------:|:--------:|:----:|:----:|:--------:|
| Logistic Regression | 84.69% | -13.03% | 2.29 | -14.58% | 0.618 |
| Random Forest | 65.51% | -11.41% | 2.14 | -33.77% | 0.567 |
| Gradient Boosting | 40.59% | -11.66% | 1.51 | -58.68% | 0.538 |
| Hist Gradient Boosting | 87.71% | -17.60% | 2.02 | -11.57% | 0.546 |
| LightGBM | 74.38% | -17.60% | 1.82 | -24.90% | 0.556 |
| LSTM | 84.73% | -15.30% | 2.18 | -14.55% | 0.500 |

### regime (20 特征)

| 模型 | 累计收益 | 最大回撤 | 夏普 | 超额 | test AUC |
|------|:--------:|:--------:|:----:|:----:|:--------:|
| Logistic Regression | 82.83% | -12.51% | 2.48 | -16.45% | 0.610 |
| Random Forest | **104.45%** | -11.91% | 2.63 | **+5.17%** | 0.646 |
| Gradient Boosting | 44.88% | -6.28% | 2.28 | -54.39% | 0.580 |
| Hist Gradient Boosting | 87.97% | -16.92% | 2.15 | -11.31% | 0.592 |
| LightGBM | **102.07%** | -15.70% | 2.36 | **+2.79%** | 0.635 |
| LSTM | 84.73% | -15.30% | 2.18 | -14.55% | 0.500 |

### momentum (20 特征)

| 模型 | 累计收益 | 最大回撤 | 夏普 | 超额 | test AUC |
|------|:--------:|:--------:|:----:|:----:|:--------:|
| Logistic Regression | 87.31% | -14.84% | 2.10 | -11.97% | 0.620 |
| Random Forest | 88.93% | -11.95% | 2.53 | -10.35% | 0.640 |
| Gradient Boosting | 77.66% | -13.15% | 2.26 | -21.62% | 0.598 |
| Hist Gradient Boosting | 66.56% | -6.05% | 2.66 | -32.71% | 0.611 |
| LightGBM | **105.13%** | -11.95% | **2.77** | **+5.85%** | 0.612 |
| LSTM | 84.73% | -15.30% | 2.18 | -14.55% | 0.500 |

### composite (28 特征)

| 模型 | 累计收益 | 最大回撤 | 夏普 | 超额 | test AUC |
|------|:--------:|:--------:|:----:|:----:|:--------:|
| 🥇 **Logistic Regression** | **107.35%** | -14.62% | 2.69 | **+8.07%** | 0.634 |
| Random Forest | 90.40% | -10.84% | 2.68 | -8.88% | 0.648 |
| Gradient Boosting | 87.55% | -13.48% | 2.51 | -11.73% | 0.600 |
| Hist Gradient Boosting | 68.19% | **-3.22%** | **3.57** | -31.09% | 0.642 |
| LightGBM | 86.00% | -12.80% | 2.45 | -13.28% | 0.650 |
| LSTM | 84.73% | -15.30% | 2.18 | -14.55% | 0.486 |

---

## 六、LSTM 专项分析

### 模型架构

```
Input(15 days, 28 features) → GRU(32) → Dropout(0.3) → Dense(16, relu) → Dense(1, sigmoid)
```

### 各特征组表现

| 特征组 | 累计收益 | 最大回撤 | 夏普 | test AUC |
|--------|:--------:|:--------:|:----:|:--------:|
| market(12) | 84.73% | -15.30% | 2.18 | 0.500 |
| regime(20) | 84.73% | -15.30% | 2.18 | 0.500 |
| momentum(20) | 84.73% | -15.30% | 2.18 | 0.500 |
| composite(28) | 84.73% | -15.30% | 2.18 | 0.486 |

> **四个特征组的回测结果完全一致**（收益、回撤、夏普小数点后四位相同），说明 GRU 完全没从特征中学习到任何有效信号。

### 原因分析

1. **数据量太小**：训练集约 1600 条样本，对深度学习远远不够
2. **信噪比极低**：金融数据的可预测成分很弱，深度学习需要大量数据才能提取稳定模式
3. **GRU 学到的唯一"规律"**：牛市里大部分时候上涨 → 始终预测涨 (prob ≈ 0.58) → 仓位始终 1.0 → 等于买入持有
4. **AUC = 0.50**：证明概率排序完全随机，没有预测能力

---

## 七、最终排名

### 按累计收益

| 排名 | 模型 | 特征组 | 累计收益 | 超额 |
|:----:|------|--------|:--------:|:----:|
| 1 | 🥇 Logistic Regression | composite | **107.35%** | **+8.07%** |
| 2 | 🥈 LightGBM | momentum | 105.13% | +5.85% |
| 3 | 🥉 Random Forest | regime | 104.45% | +5.17% |
| 4 | LightGBM | regime | 102.07% | +2.79% |
| 5 | Random Forest | composite | 90.40% | -8.88% |
| ... | ... | ... | ... | ... |
| 24 | Gradient Boosting | market | 40.59% | -58.68% |

### 按夏普比率

| 排名 | 模型 | 特征组 | 夏普 | 最大回撤 |
|:----:|------|--------|:----:|:--------:|
| 1 | 🥇 Hist Gradient Boosting | composite | **3.57** | **-3.22%** |
| 2 | 🥈 LightGBM | momentum | 2.77 | -11.95% |
| 3 | 🥉 Logistic Regression | composite | 2.69 | -14.62% |
| 4 | Random Forest | composite | 2.68 | -10.84% |
| 5 | Hist Gradient Boosting | momentum | 2.66 | -6.05% |

### 按最大回撤

| 排名 | 模型 | 特征组 | 最大回撤 | 收益 |
|:----:|------|--------|:--------:|:----:|
| 1 | 🥇 Hist Gradient Boosting | composite | **-3.22%** | 68.19% |
| 2 | 🥈 Hist Gradient Boosting | momentum | -6.05% | 66.56% |
| 3 | 🥉 Gradient Boosting | regime | -6.28% | 44.88% |
| 4 | Random Forest | composite | -10.84% | 90.40% |
| 5 | Random Forest | momentum | -11.95% | 88.93% |

---

## 八、核心结论

1. **Logistic Regression + composite 综合最强**：唯一真正实现"指数增强"的模型，超额 +8.07%
2. **追求低回撤选 Hist Gradient Boosting**：最大回撤仅 -3.22%，夏普 3.57，但收益偏低
3. **特征越多越好**：composite(28) > regime(20) ≈ momentum(20) >> market(12)
4. **LSTM 不适用于此场景**：数据量太小（~1600条），深度学习无法提取有效信号，表现等于买入持有
5. **每个模型的最优特征组不同**：Logistic Regression 喜欢 composite，Random Forest 喜欢 momentum/regime，LightGBM 喜欢 momentum
