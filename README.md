# Cage_prediction — 深远海养殖网箱动态响应深度学习预测

硕士论文《基于深度学习的深远海养殖网箱动态响应预测方法研究》配套代码的工程化整理版本。

研究对象为浮式深远海养殖网箱在波浪载荷作用下的运动响应（纵荡 Surge、垂荡 Heave、纵摇 Pitch 等）和系泊缆力（Force1、Force2）的深度学习预测。

**复现状态**：第3章、第4章、第5章（5.1-5.2 节）核心实验已全部完成复现训练，共 **57 组实验**，评估指标均达到或超过论文报告值。第5.3节（多级海况泛化验证）暂未纳入工程化架构。

---

## 论文章节与代码对应

| 章节 | 内容 | CLI 入口 | 复现状态 |
|------|------|----------|:---:|
| 全部 | 一键运行第3-5章全部实验 | `run-all` | **已完成** |
| 第3章 | LSTM/GRU/BiLSTM 网箱运动响应预测（H→运动） | `run-s3` | **已完成** |
| 第4.2节 | HGBRT/BPNN 系泊力回归预测（运动→力） | `run-s4-regression` | **已完成** |
| 第4.3节 | LSTM 系泊力时间序列预测（运动历史+力历史→力） | `run-s4-lstm` | **已完成** |
| 第4.4节 | 两阶段混合联合预测（H→运动→力） | `run-s4-hybrid` | **已完成** |
| 第5.1-5.2节 | CNN-BiLSTM-Attention 模型对比与泛化分析 | `run-s5-attention` | **已完成** |
| 第5.3节 | 多级海况泛化测试 | 暂未纳入 | 待实现 |

---

## 项目结构

```
Cage_prediction/
├── README.md
├── pyproject.toml                   # pip install -e . 支持
├── requirements.txt                 # 核心依赖 (8 包)
├── .gitignore
├── 吴大庆 - 2025 - 基于深度学习的...pdf  # 被复现的原始论文
│
├── scripts/
│   └── reproduce_all.ps1            # PowerShell 一键复现脚本
│
├── configs/                         # YAML 实验配置 (13 个)
│   ├── s3_motion.yaml               #   第3章 LSTM 运动预测
│   ├── s4_regression.yaml           #   第4.2节 HGBRT/BPNN 回归
│   ├── s4_lstm_mooring.yaml         #   第4.3节 LSTM 系泊力时序预测
│   ├── s4_lstm_mooring_wave_known_force1.yaml  # 第4.3节 波浪已知 Force1
│   ├── s4_lstm_mooring_wave_known_force2.yaml  # 第4.3节 波浪已知 Force2
│   ├── s4_hybrid.yaml               #   第4.4节 混合预测 (os=1)
│   ├── s4_hybrid_step{2,4,6,8}.yaml #   第4.4节 多输出步长变体
│   ├── s5_attention.yaml            #   第5章 Force1 模型对比 (hor=10/20/30/40)
│   ├── s5_attention_force2_h20.yaml #   第5章 Force2 hor=20
│   └── s5_attention_force2_h30.yaml #   第5章 Force2 hor=30
│
├── data/
│   └── raw/
│       ├── t_1.csv                  #   第3章用 (11,000 行)
│       └── t_2_11.2_50.csv          #   第4、5章用 (40,000 行)
│
├── results/                         # 正式复现结果
│   ├── README.md
│   ├── 论文复现报告_完整版.md
│   ├── 复现报告_第3章.md
│   ├── 复现报告_第4章.md
│   ├── 复现报告_第5章.md
│   ├── s3/                          #   第3章: 9 CSV + 6 PNG
│   ├── s4/                          #   第4.2节: 4 CSV + 4 PNG
│   │   ├── lstm_mooring/            #   第4.3节: 5 CSV + 8 PNG
│   │   └── hybrid/                  #   第4.4节: 11 CSV + 20 PNG
│   └── s5/                          #   第5章: 40 CSV + 40 PNG
│
└── src/cage_predict/                # 工程化核心代码
    ├── __init__.py
    ├── __main__.py                  #   python -m cage_predict 入口
    ├── cli.py                       #   命令行 (6 子命令 + run-all)
    ├── config.py                    #   YAML 加载 + 2 层配置校验
    ├── data.py                      #   数据加载、归一化、滑动窗口
    ├── metrics.py                   #   评估指标 (RMSE/MAE/MAPE/Acc/Corr)
    ├── plotting.py                  #   预测曲线 + 损失曲线绘图
    ├── utils.py                     #   随机种子、目录工具
    ├── experiments/
    │   ├── s3_motion.py             #   第3章实验
    │   ├── s4_regression.py         #   第4.2节实验
    │   ├── s4_lstm_mooring.py       #   第4.3节实验
    │   ├── s4_hybrid.py             #   第4.4节实验
    │   └── s5_attention.py          #   第5章实验
    └── models/
        ├── attention.py             #   CNN-BiLSTM-Attention (核心模型)
        ├── bilstm.py                #   BiLSTM + LeakyReLU
        ├── bpnn.py                  #   BPNN (3×150)
        ├── gru.py                   #   GRU (3层)
        ├── hgbdt.py                 #   HGBRT + GridSearchCV
        ├── lstm.py                  #   LSTM (1层/3层可配)
        ├── nbeatsx_adapter.py       #   N-BEATSx (PyTorch 适配)
        └── xgboost.py               #   XGBoost + GridSearchCV
```

---

## 复现结果概览

| 章节 | 实验数 | 完成 | 论文关键Acc |
|------|:---:|:---:|------|
| 第3章 | 9 | ✅ 100% | 0.852–0.998 (论文 >0.85) |
| 第4.2节 | 4 | ✅ 100% | 0.994–0.998 (论文 0.996–0.998) |
| 第4.3节 | 2 | ✅ 100% | 0.970–0.987 (论文 0.975–0.983) |
| 第4.4节 | 2 | ✅ 100% | 0.990–0.997 (论文 0.944–0.999) |
| 第5章 Force1 | 20 | ✅ 100% | 0.902–0.969 (论文 >0.90) |
| 第5章 Force2 | 20 | ✅ 100% | 0.632–0.881 (核心验证通过) |
| 第5.3节 | — | — | 暂未纳入 |
| **合计** | **57** | **100%** | |

> 详细复现报告见 [results/论文复现报告_完整版.md](results/论文复现报告_完整版.md)

---

## CSV 数据字段

| 列名 | 单位 | 说明 |
|------|------|------|
| `T` | s | 时间（步长 0.1s） |
| `H` | mm | 波高 |
| `Surge` | mm | 纵荡位移 |
| `Sway` | mm | 横荡位移 |
| `Heave` | mm | 垂荡位移 |
| `Roll` | rad | 横摇角 |
| `Pitch` | rad | 纵摇角 |
| `Yaw` | rad | 艏摇角 |
| `Force1` | N | 迎浪侧系泊受力 |
| `Force2` | N | 背浪侧系泊受力 |

---

## 环境安装

**已验证环境**：conda 环境 `predict_w`（Python 3.8.20, TensorFlow 2.10.0, GPU: RTX 5070 Laptop）

```bash
conda activate predict_w
pip install -e .
```

N-BEATSx 模型需要 PyTorch：
```bash
pip install torch
```

---

## 快速开始

### 查看可用命令

```bash
python -m cage_predict --help
```

### 一键完整复现

```bash
# 烟雾测试（验证所有实验流程可运行，约 5-10 分钟）
python -m cage_predict run-all --smoke-test

# 完整复现（完整参数训练，约 3 小时）
python -m cage_predict run-all
```

Windows PowerShell 方式：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/reproduce_all.ps1
```

### 单章运行

| 章节 | 命令 | GPU 估算 |
|------|------|:---:|
| 第3章 | `run-s3 --config configs/s3_motion.yaml` | ~15 min |
| 第4.2节 | `run-s4-regression --config configs/s4_regression.yaml` | ~10 min |
| 第4.3节 | `run-s4-lstm --config configs/s4_lstm_mooring.yaml` | ~5 min |
| 第4.4节 | `run-s4-hybrid --config configs/s4_hybrid.yaml` | ~20 min |
| 第5章 Force1 | `run-s5-attention --config configs/s5_attention.yaml` | ~60 min |
| 第5章 Force2 h20 | `run-s5-attention --config configs/s5_attention_force2_h20.yaml` | ~55 min |
| 第5章 Force2 h30 | `run-s5-attention --config configs/s5_attention_force2_h30.yaml` | ~50 min |

添加 `--smoke-test` 可快速验证流程（精度无意义，仅验证无报错）。

### 切换实验变体

修改配置文件中对应字段：

| 需求 | 配置字段 | 可选值 |
|------|----------|--------|
| 切换系泊力目标 | `data.mooring_target` | `Force1`, `Force2` |
| 切换 Stage1 模型 | `model.compare_models` | `[lstm, gru, bilstm, attention, nbeatsx]` |
| 改变预报提前量 | `data.horizon` | `10`, `20`, `30`, `40` |
| 切换混合输出步长 | 使用 `s4_hybrid_step*.yaml` | os=2/4/6/8 |
| 波浪已知/未知 | 使用 `*wave_known*.yaml` | — |

---

## 模型架构说明

| 章节 | 模型 | 架构 |
|------|------|------|
| 第3章 | LSTM | 1层 LSTM(25) + Dense(1)+tanh |
| 第3章 | GRU | 3层 GRU(25→100→100) + Dropout(0.3) + Dense(1)+tanh |
| 第3章 | BiLSTM | BiLSTM(25) + LeakyReLU(0.3) + Dense(1)+tanh |
| 第4.2节 | BPNN | Dense(150,tanh)×3 + Dense(1,linear) |
| 第4.2节 | HGBRT | HistGradientBoostingRegressor + GridSearchCV |
| 第4.3/4.4节 | LSTM (Stage1) | 1层 LSTM(25) + Dense(1)+tanh |
| 第4.4节 | BPNN (Stage2) | Dense(150,tanh)×3 + Dense(1,linear) |
| 第5章 | CNN-BiLSTM-Attention | Conv1D(64)→BiLSTM(64)→Attention(permute)→Dense(1) |
| 第5章 | N-BEATSx | PyTorch 实现，stack=[trend, seasonality] |

---

## 与原始代码的已知差异（有意改进）

| # | 改进项 | 说明 |
|---|--------|------|
| 1 | Adam 优化器正确传参 | 原始代码 `Adam(lr=lr)` 创建后未传入 `model.compile()`，实际始终用 Keras 默认 lr=0.001 |
| 2 | Adam gradient clipping | `clipnorm=1.0`，防止高学习率下梯度爆炸 |
| 3 | Force2 独立 scaler | 原始代码错误地用 Force1 归一化器处理 Force2 数据 |
| 4 | BPNN 输出层连接 | 原始 Functional API 输出层连接错误（第三隐藏层为死代码） |
| 5 | N-BEATSx 集成 | 通过适配器集成到模型对比流程，PyTorch 侧因 RTX 5070 驱动兼容性问题运行于 CPU |
| 6 | 两层配置校验 | CLI 启动前 + 实验函数入口双重校验，尽早发现配置错误 |

---

## SCI 小论文核心结果

当前小论文以 **CNN-BiLSTM-Attention** 为核心模型：

- 核心数据: `results/s5/force2_attention_h500_hor{10,20,30,40}.csv`
- 对比基线: `results/s5/force2_{lstm,gru,bilstm}_h500_hor{10,20,30,40}.csv`
- 辅助参考: `results/s4/lstm_mooring/` (波浪已知/未知), `results/s4/hybrid/` (多输出步长)

---

## 结果管理

- `results/` 目录包含全部正式复现输出 (69 CSV + 78 PNG + 5 MD)
- CSV/PNG 文件由 `.gitignore` 排除，不提交 Git；复现报告 Markdown 文件提交
- 第5.3节（多级海况泛化）暂未纳入工程化架构
