# CagePredict — 深远海养殖网箱动态响应深度学习预测

硕士论文《基于深度学习的深远海养殖网箱动态响应预测方法研究》配套代码的工程化整理版本。

研究对象为浮式深远海养殖网箱在波浪载荷作用下的运动响应（纵荡 Surge、垂荡 Heave、纵摇 Pitch 等）和系泊缆力（Force1、Force2）的深度学习预测。

工程化目标：支持终端命令行运行、YAML 配置文件驱动实验、结果自动保存至 `results/`。

---

## 论文章节与代码对应

| 原始目录 | 章节 | 内容 | CLI 入口 |
|----------|------|------|----------|
| `第三章/` | 第3章 | 基于 LSTM 的网箱运动响应预测（原始 Notebook） | `python -m cage_predict run-s3` |
| `第四章/` | 第4章 | 系泊受力预测（原始 Notebook：4.2 回归 + 4.3 时序 + 4.4 混合） | `run-s4-regression` / `run-s4-lstm` / `run-s4-hybrid` |
| `第五章/` | 第5章 | CNN-BiLSTM-Attention 两阶段联合预测（原始 Notebook） | `python -m cage_predict run-s5-attention` |

工程化代码位于 `src/cage_predict/`，原始 Notebook 代码保留在 `第三章/`、`第四章/`、`第五章/` 中未做任何修改。

---

## 项目结构

```
Wudaqing_project/
├── README.md                        # 本文件
├── pyproject.toml                   # 项目元数据与 pip install -e . 支持
├── requirements.txt                 # 依赖清单
├── .gitignore
│
├── configs/                         # YAML 实验配置文件（5个章节各1个）
│   ├── s3_motion.yaml               #   第3章 LSTM 运动预测
│   ├── s4_regression.yaml           #   第4.2章 HGBRT/BPNN/XGBoost 回归
│   ├── s4_lstm_mooring.yaml         #   第4.3章 LSTM 系泊力时序预测（3层）
│   ├── s4_hybrid.yaml               #   第4.4章 混合模型联合预测
│   └── s5_attention.yaml            #   第5章 两阶段Attention模型对比
│
├── data/                            # 数据目录
│   └── raw/                         #   原始输入 CSV
│       ├── t_1.csv                  #     第3章用（复制自 s3/t_1.csv）
│       └── t_2_11.2_50.csv          #     第4、5章用（复制自 s4/4.2/）
│
├── results/                         # 实验输出（程序自动生成）
│   ├── s3/
│   ├── s4/
│   └── s5/
│
├── src/cage_predict/                # 工程化主包
│   ├── __init__.py                  #   包初始化
│   ├── __main__.py                  #   使 python -m cage_predict 可用
│   ├── cli.py                       #   命令行入口（5个子命令）
│   ├── config.py                    #   YAML 配置加载 + smoke_test 覆盖
│   ├── data.py                      #   数据IO、归一化、时序切片、集划分
│   ├── metrics.py                   #   MAE/MAPE/MSE/RMSE/Rtrapz(Acc)
│   ├── plotting.py                  #   预测曲线图、loss曲线图
│   ├── utils.py                     #   随机种子、日志、目录工具
│   ├── models/                      #   模型定义
│   │   ├── lstm.py                  #     LSTM（1层/3层可配）+ tanh 输出
│   │   ├── gru.py                   #     GRU（3层）+ tanh 输出
│   │   ├── bilstm.py                #     BiLSTM + LeakyReLU + tanh 输出
│   │   ├── bpnn.py                  #     BPNN（3层150单元）
│   │   ├── hgbdt.py                 #     HGBRT + GridSearchCV
│   │   ├── xgboost.py               #     XGBoost + GridSearchCV（4.2节对比模型）
│   │   └── attention.py             #     CNN(relu)-BiLSTM-Attention (attention_3d_block2)
│   └── experiments/                 #   实验运行脚本
│       ├── s3_motion.py             #     第3章
│       ├── s4_regression.py         #     第4.2章
│       ├── s4_lstm_mooring.py       #     第4.3章
│       ├── s4_hybrid.py             #     第4.4章
│       └── s5_attention.py          #     第5章
│
├── tests/                           # 测试
│   ├── test_data.py                 #   数据模块（23 tests）
│   ├── test_metrics.py              #   指标模块（19 tests）
│   └── test_smoke.py                #   模型冒烟（5 tests, 需 TensorFlow）
│
├── s3/                              # 原始第3章代码（未修改，保留）
├── s4/                              # 原始第4章代码（未修改，保留）
├── s5/                              # 原始第5章代码（未修改，保留）
│
├── legacy/                          # 预留：原始 .py 脚本归档
├── notebooks/                       # 预留：原始 Jupyter Notebook 归档
└── third_party/                     # 预留：外部 SOTA 模型参考代码
```

---

## CSV 数据字段说明

所有原始数据文件为时序 CSV，列定义如下：

| 列名 | 单位 | 说明 |
|------|------|------|
| `T` | s | 时间（步长 0.1s） |
| `H` | mm | 波高（波浪信息） |
| `Surge` | mm | 纵荡位移 |
| `Sway` | mm | 横荡位移 |
| `Heave` | mm | 垂荡位移 |
| `Roll` | rad | 横摇角（原始数据中需×1e-6 显示） |
| `Pitch` | rad | 纵摇角 |
| `Yaw` | rad | 艏摇角（原始数据中需×1e-6 显示） |
| `Force1` | N | 迎浪侧系泊受力 |
| `Force2` | N | 背浪侧系泊受力 |

---

## 环境安装

**已验证环境**：conda 环境 `predict_w`（Python 3.8.20, TensorFlow 2.10.0, Keras 2.10.0, GPU: RTX 5070 Laptop）。

### 方式 A：推荐（可编辑安装）

```bash
conda activate predict_w
cd Wudaqing_project
pip install -e . --no-deps
```

安装后 `python -m cage_predict` 可直接在任意目录运行。

### 方式 B：临时（不安装包）

```bash
PYTHONPATH=src python -m cage_predict --help
```

每次运行需带 `PYTHONPATH=src` 前缀。

---

## 快速开始

### 查看命令帮助

```bash
python -m cage_predict --help
```

### 冒烟测试（验证流程是否畅通）

冒烟测试使用极少数据（100条训练、少量epoch），**结果不代表论文精度**，仅用于验证数据加载→预处理→模型构建→训练→预测→评估全流程无报错。

```bash
# 第3章：LSTM 预测 Heave（~2 min GPU）
python -m cage_predict run-s3 --config configs/s3_motion.yaml --smoke-test

# 第4.2章：BPNN 预测 Force1（~2 min）
python -m cage_predict run-s4-regression --config configs/s4_regression.yaml --smoke-test

# 第4.3章：LSTM 时序预测 Force1（~3 min）
python -m cage_predict run-s4-lstm --config configs/s4_lstm_mooring.yaml --smoke-test

# 第4.4章：混合模型预测 Force1（~5 min，训练3个LSTM+1个BPNN）
python -m cage_predict run-s4-hybrid --config configs/s4_hybrid.yaml --smoke-test

# 第5章：LSTM vs Attention 对比（~8 min）
python -m cage_predict run-s5-attention --config configs/s5_attention.yaml --smoke-test
```

### 完整训练（复现论文）

去掉 `--smoke-test` 即运行完整训练。

| 章节 | 命令 | 训练规模 | GPU 估算耗时 |
|------|------|----------|-------------|
| 第3章 | `run-s3` | 7500样本×60epoch | ~15 min |
| 第4.2章 BPNN | `run-s4-regression` | 10000样本×1000epoch | ~10 min |
| 第4.2章 HGBRT | 同上（config 中改 model.name 为 hgbdt） | GridSearchCV ~2700组合 | ~20 min |
| 第4.2章 XGBoost | 同上（config 中改 model.name 为 xgboost） | GridSearchCV ~11200组合 | ~30 min |
| 第4.3章 | `run-s4-lstm` | 600样本×30epoch（3层LSTM） | ~5 min |
| 第4.4章 | `run-s4-hybrid` | 3 LSTM(30epoch) + BPNN(1000epoch) | ~20 min |
| 第5章 | `run-s5-attention` | 两阶段: Stage1(3运动×4模型×30epoch) + Stage2(4×BPNN×1000epoch) | ~60 min |

### 切换预测目标

各配置文件修改对应字段即可：

| 需求 | 修改文件 | 字段 | 示例 |
|------|----------|------|------|
| 预测 Surge/Pitch | `s3_motion.yaml` | `data.target` | `"Surge"` |
| 预测 Force2 | `s4_*.yaml` | `data.target` | `"Force2"` |
| 切换模型 | `s4_regression.yaml` | `model.name` | `"hgbdt"` |
| 改变输入特征 | `s4_regression.yaml` | `data.input_features` | `["H","Surge","Heave","Pitch"]` |
| 改变时间窗 | `s4_lstm_mooring.yaml` | `data.look_back` | `100` |
| 改变训练样本数 | `s4_lstm_mooring.yaml` | `data.train_start` | `7400`（100样本） |
| 对比模型列表 | `s5_attention.yaml` | `model.compare_models` | `["lstm","gru","bilstm","attention"]` |

### 运行测试

```bash
pytest tests/ -v           # 全部 47 个测试
pytest tests/test_data.py -v     # 数据模块（23个，无需TF）
pytest tests/test_metrics.py -v  # 指标模块（19个，无需TF）
pytest tests/test_smoke.py -v    # 模型冒烟（5个，需TF）
```

---

## 冒烟测试 vs 完整训练

| 方面 | `--smoke-test` | 完整训练 |
|------|---------------|----------|
| 目的 | 验证流程端到端可运行 | 复现论文结果 |
| 训练样本 | ~100 | 600–10000 |
| 验证样本 | ~300–500 | ~3000–5000 |
| Epochs | 1–2 | 30–1000 |
| 输出文件后缀 | `_smoke.csv` | `.csv` |
| 指标有效性 | **不可用于论文** | 可用于论文 |

---

## 配置参数与原始代码一致性

所有配置文件默认值（不带 `--smoke-test`）均与原始代码对应：

| 配置文件 | 关键参数 | 对应原始 |
|----------|----------|----------|
| `s3_motion.yaml` | look_back=55, epochs=60, lr=0.01, 3层LSTM[25,100,100] | `s3/3.3.py` |
| `s4_regression.yaml` | BPNN 150×3 tanh, epochs=1000, lr=0.01 | `s4/4.2/4.2.py` |
| `s4_lstm_mooring.yaml` | look_back=55, epochs=30, 3层LSTM[25,100,100] | `s4/4.3/4.3.py` |
| `s4_hybrid.yaml` | look_back=50, Stage1 30epoch, Stage2 1000epoch, 支持按运动分设学习率 | `s4/4.4/4.4.py` |
| `s5_attention.yaml` | 两阶段联合预测: Stage1(H→运动Attention/BiLSTM/LSTM/GRU, look_back=500, epochs=30), Stage2(预测运动→力 BPNN, epochs=1000) | `s5/第五章（上）.py` |

---

## 第5章工程化现状

| 功能 | 状态 | 说明 |
|------|------|------|
| 两阶段联合预测（Stage1: H→运动, Stage2: 预测运动→力） | ✅ 已完成 | 完整复现原始代码的两阶段串联架构 |
| `model_compare`（多模型对比） | ✅ 已完成 | Stage1 支持 lstm / gru / bilstm / attention 四种模型对比 |
| 模型切换 | ✅ 已完成 | 通过 `configs/s5_attention.yaml` 中 `model.compare_models` 列表配置 |
| 按运动分量独立学习率 | ✅ 已完成 | Heave=0.1, Surge=1.3, Pitch=0.35 |
| 按运动分量独立训练起点 | ✅ 已完成 | Surge 使用 train_start=4200（更多训练数据），其余用 4500 |
| `sea_state_test`（多海况批量泛化测试） | ⏳ 保留扩展接口 | 配置文件预留了 `mode` 字段和 `generalization` 配置块。完整的多谱峰周期、多水深、多海况等级批量实验建议参考原始 `s5/` 代码或后续扩展 |

## 模型架构修正记录

以下修正确保了新工程化代码与原始论文代码的精确一致：

| 模型 | 修正项 | 原始代码 | 修正前 | 修正后 |
|------|--------|----------|--------|--------|
| LSTM | 输出层激活 | Dense+tanh | Dense (无激活) | Dense+tanh |
| GRU | 输出层激活 | Dense+tanh | Dense (无激活) | Dense+tanh |
| Attention | Conv1D激活 | relu | tanh | relu |
| Attention | 多余Softmax | 无 | Activation("softmax") | 已移除 |
| 第5章 | 预测架构 | 两阶段(H→运动→力) | 单阶段(运动→力) | 两阶段(H→运动→力) |
| 第4.3章 | LSTM层数 | 3层[25,100,100] | 1层[25] | 3层[25,100,100] |
| 第4.2章 | XGBoost对比 | 有(GridSearchCV) | 缺失 | 已添加 |
| 第4.3/4.4章 | 训练索引 | 直接使用x数组索引 | 错误减去skip_rows | 直接使用x数组索引 |

---

## 原始文件保护

- `第三章/`、`第四章/`、`第五章/` 三个原始目录**未被修改或删除**
- 原始 notebook（`.ipynb`）、脚本（`.py`）、数据/结果（`.csv`）均完整保留
- 仅复制了两个数据文件到 `data/raw/`：原始数据来源为 `第三章/t_1.csv` 和 `第四章/4.2/t_2_11.2_50.csv`
- 后续修改应优先修改 `src/cage_predict/` 中的工程化代码，不应直接改动原始文件

---

## 常见问题

### TensorFlow / Keras 版本兼容

本项目使用 Keras 2 API（`from keras.layers import ...`）。推荐 `tensorflow>=2.4,<2.16`。如使用 TF 2.16+（Keras 3），代码中的 `Adam(lr=...)` 兼容包装器会自动处理。已验证 TF 2.10.0 + Keras 2.10.0。

### GPU / CPU 差异

GPU 与 CPU 训练结果可能因浮点精度存在微小差异。设置 `CUDA_VISIBLE_DEVICES=-1` 可强制 CPU 模式。

### 直接运行原始 .py 文件

原始目录中的 `.py` 文件由 Jupyter Notebook 导出（含 `# In[1]:` 等标记），作为独立脚本运行可能失败。请使用 `python -m cage_predict` CLI 入口。

### 可选依赖

- `xgboost`：原始 s4/4.2 中包含 XGBoost 对比实验，需单独安装
- `paddlehub`、`rembg`、`paddlepaddle`：仅出现在原始 `s3/3.4.py` 末尾的图像处理代码中，与预测主流程**无关**，**不需安装**，未列入核心依赖

### 清理缓存

```bash
find . -type d -name "__pycache__" -exec rm -rf {} +
find . -type d -name ".pytest_cache" -exec rm -rf {} +
rm -rf src/cage_predict.egg-info build dist
```
