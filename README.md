# CagePredict — 深远海养殖网箱动态响应深度学习预测

硕士论文《基于深度学习的深远海养殖网箱动态响应预测方法研究》配套代码的工程化整理版本。

研究对象为浮式深远海养殖网箱在波浪载荷作用下的运动响应（纵荡 Surge、垂荡 Heave、纵摇 Pitch 等）和系泊缆力（Force1、Force2）的深度学习预测。

---

## 论文章节与代码对应

| 原始目录 | 章节 | 内容 | CLI 入口 |
|----------|------|------|----------|
| `第三章/` | 第3章 | LSTM/GRU/BiLSTM 网箱运动响应预测（H→运动） | `run-s3` |
| `第四章/` | 第4.2节 | HGBRT/BPNN/XGBoost 系泊力回归预测（运动→力） | `run-s4-regression` |
| `第四章/` | 第4.3节 | LSTM 系泊力时间序列预测（运动历史+力历史→力） | `run-s4-lstm` |
| `第四章/` | 第4.4节 | 两阶段混合联合预测（H→运动→力） | `run-s4-hybrid` |
| `第五章/` | 第5.1-5.2节 | CNN-BiLSTM-Attention 模型对比（H→运动→力） | `run-s5-attention` |
| `第五章/` | 第5.3节 | 多级海况泛化测试 | 暂未纳入新架构 |

工程化代码位于 `src/cage_predict/`。原始 Notebook 和代码完整保留在 `第三章/`、`第四章/`、`第五章/` 中。

---

## 项目结构

```
Wudaqing_project/
├── README.md
├── pyproject.toml                   # pip install -e . 支持
├── requirements.txt
├── .gitignore
│
├── configs/                         # YAML 实验配置（每章一个）
│   ├── s3_motion.yaml               #   第3章 LSTM 运动预测
│   ├── s4_regression.yaml           #   第4.2节 HGBRT/BPNN/XGBoost 回归
│   ├── s4_lstm_mooring.yaml         #   第4.3节 LSTM 系泊力时序预测（1层）
│   ├── s4_hybrid.yaml               #   第4.4节 两阶段混合预测
│   └── s5_attention.yaml            #   第5章 Attention 模型对比
│
├── data/
│   └── raw/
│       ├── t_1.csv                  #   第3章用（11000行）
│       └── t_2_11.2_50.csv          #   第4、5章用（40000行）
│
├── results/                         # 实验输出（程序自动生成）
│   ├── s3/
│   ├── s4/
│   └── s5/
│
├── src/cage_predict/                # 工程化主包
│   ├── __init__.py
│   ├── __main__.py                  #   python -m cage_predict 入口
│   ├── cli.py                       #   命令行（5个子命令）
│   ├── config.py                    #   YAML 加载 + smoke_test 合并
│   ├── data.py                      #   数据IO、归一化、序列转换、集划分
│   ├── metrics.py                   #   MAE/MAPE/MSE/RMSE/Acc(Rtrapz)
│   ├── plotting.py                  #   预测曲线、loss曲线
│   ├── utils.py                     #   随机种子、日志、目录工具
│   ├── models/
│   │   ├── lstm.py                  #     LSTM（1层/3层可配）
│   │   ├── gru.py                   #     GRU（3层）
│   │   ├── bilstm.py                #     BiLSTM + LeakyReLU
│   │   ├── bpnn.py                  #     BPNN（3×150）
│   │   ├── hgbdt.py                 #     HGBRT + GridSearchCV
│   │   ├── xgboost.py               #     XGBoost + GridSearchCV
│   │   ├── attention.py             #     CNN-BiLSTM-Attention
│   │   └── nbeatsx_adapter.py       #     N-BEATSx (PyTorch)
│   └── experiments/
│       ├── s3_motion.py             #     第3章
│       ├── s4_regression.py         #     第4.2节
│       ├── s4_lstm_mooring.py       #     第4.3节
│       ├── s4_hybrid.py             #     第4.4节
│       └── s5_attention.py          #     第5章
│
├── tests/
│   ├── test_data.py
│   ├── test_metrics.py
│   └── test_smoke.py
│
├── 第三章/                          # 原始 Notebook（未修改）
├── 第四章/                          # 原始 Notebook（未修改）
├── 第五章/                          # 原始 Notebook（未修改）
├── legacy/                          # 原始代码归档
├── notebooks/
└── third_party/
```

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

**已验证环境**：conda 环境 `predict_w`（Python 3.8.20, TensorFlow 2.10.0, Keras 2.10.0, GPU: RTX 5070 Laptop）

```bash
# 推荐：可编辑安装
conda activate predict_w
pip install -e . --no-deps

# 安装可选依赖
pip install -e ".[dev]"      # pytest + xgboost
pip install -e ".[torch]"    # PyTorch（N-BEATSx 模型需要）
pip install -e ".[all]"      # 全部依赖
```

---

## 快速开始

### 查看帮助

```bash
python -m cage_predict --help
```

### 烟雾测试（验证流程可运行）

烟雾测试使用极少数据（~100训练样本、1-2 epoch），**结果不代表论文精度**，仅验证全流程无报错。

```bash
# 第3章：LSTM 运动预测
python -m cage_predict run-s3 --config configs/s3_motion.yaml --smoke-test

# 第4.2节：HGBRT/BPNN 回归预测
python -m cage_predict run-s4-regression --config configs/s4_regression.yaml --smoke-test

# 第4.3节：LSTM 系泊力时序预测
python -m cage_predict run-s4-lstm --config configs/s4_lstm_mooring.yaml --smoke-test

# 第4.4节：两阶段混合预测
python -m cage_predict run-s4-hybrid --config configs/s4_hybrid.yaml --smoke-test

# 第5章：Attention 模型对比
python -m cage_predict run-s5-attention --config configs/s5_attention.yaml --smoke-test
```

### 完整训练（复现论文）

去掉 `--smoke-test` 即可。

| 章节 | 命令 | 训练规模 | GPU 估算 |
|------|------|----------|----------|
| 第3章 | `run-s3` | 7500样本×60epoch | ~15 min |
| 第4.2节 BPNN | `run-s4-regression` | 10000样本×1000epoch | ~10 min |
| 第4.2节 HGBRT | `run-s4-regression`（config 中改 `model.name: hgbdt`） | GridSearchCV | ~20 min |
| 第4.2节 XGBoost | `run-s4-regression`（config 中改 `model.name: xgboost`） | GridSearchCV | ~30 min |
| 第4.3节 | `run-s4-lstm` | 600样本×30epoch | ~5 min |
| 第4.4节 | `run-s4-hybrid` | Stage1: 3 LSTM×30epoch + Stage2: BPNN×1000epoch | ~20 min |
| 第5章 | `run-s5-attention` | Stage1: 3运动×4模型×30epoch + Stage2: 4 BPNN×1000epoch | ~60 min |

### 切换预测目标

修改配置文件对应字段：

| 需求 | 文件 | 字段 | 可选值 |
|------|------|------|--------|
| 切换运动目标 | `s3_motion.yaml` | `data.target` | `Heave`, `Surge`, `Pitch`, `Sway`, `Roll`, `Yaw` |
| 切换系泊力目标 | `s4_*.yaml` / `s5_attention.yaml` | `data.target` / `data.mooring_target` | `Force1`, `Force2` |
| 切换模型类型 | `s4_regression.yaml` | `model.name` | `bpnn`, `hgbdt`, `xgboost` |
| 切换 Stage1 模型 | `s3_motion.yaml` | `model.name` | `lstm`, `gru`, `bilstm` |
| 改变输入特征 | `s4_regression.yaml` | `data.input_features` | `[Surge, Heave, Pitch]` 等 |
| 改变时间窗口 | `s4_lstm_mooring.yaml` | `data.look_back` | `1`, `10`, `30`, `50`, `100`, `200` |
| 调整训练样本量 | `s4_lstm_mooring.yaml` | `data.train_start` | 增大 = 减少训练样本 |
| 模型对比列表 | `s5_attention.yaml` | `model.compare_models` | `[lstm, bilstm, attention, nbeatsx]` |

### 运行测试

```bash
pytest tests/ -v
pytest tests/test_data.py -v     # 数据模块（无需TF）
pytest tests/test_metrics.py -v  # 指标模块（无需TF）
pytest tests/test_smoke.py -v    # 模型冒烟（需TF）
```

---

## 烟雾测试 vs 完整训练

| 方面 | `--smoke-test` | 完整训练 |
|------|---------------|----------|
| 目的 | 验证流程端到端可运行 | 复现论文结果 |
| 训练样本 | ~100 | 600–10000 |
| Epochs | 1–2 | 30–1000 |
| 输出文件后缀 | `_smoke` | 无后缀 |
| 指标有效性 | 不可用于论文 | 可用于论文 |

---

## 配置参数与原始代码对应

| 配置文件 | 关键参数 | 对应原始参考 |
|----------|----------|-------------|
| `s3_motion.yaml` | look_back=55, epochs=60, lr=0.01, 3层LSTM[25,100,100] | `第三章（上：~3.3）.ipynb` + `第三章（下：3.4~）.ipynb` |
| `s4_regression.yaml` | BPNN 150×3 tanh, epochs=1000, lr=0.01; HGBRT GridSearchCV | `第四章（上：~4.2非线性）.ipynb` |
| `s4_lstm_mooring.yaml` | look_back=55, epochs=30, 1层LSTM[25], lr=0.01 | `第四章（中：~4.3时序单LSTM）.ipynb` |
| `s4_hybrid.yaml` | look_back=50, Stage1: 1层LSTM(30epoch, 按运动分lr), Stage2: BPNN(1000epoch) | `第四章（下：~4.4联合H）.ipynb` |
| `s5_attention.yaml` | look_back=500, horizon=10, Stage1: 4模型×30epoch, Stage2: BPNN×1000epoch | `第五章/0 5.2/Attention/` + `第五章/0 5.2/BILSTM - 30epoch/` |

---

## 模型架构说明

### 各章模型

| 章节 | 模型 | 架构 |
|------|------|------|
| 第3章 | LSTM | 3层 LSTM(25→100→100) + Dropout(0.3) + Dense(1)+tanh |
| 第3章 | GRU | 3层 GRU(25→100→100) + Dropout(0.3) + Dense(1)+tanh |
| 第3章 | BiLSTM | BiLSTM(25) + LeakyReLU(0.3) + Dense(1)+tanh |
| 第4.2节 | BPNN | Dense(150,tanh)×3 + Dense(1,linear) |
| 第4.2节 | HGBRT | HistGradientBoostingRegressor + GridSearchCV |
| 第4.2节 | XGBoost | XGBRegressor + GridSearchCV |
| 第4.3/4.4节 | LSTM | 1层 LSTM(25) + Dense(1)+tanh |
| 第5章 | CNN-BiLSTM-Attention | Conv1D(64, relu) → Dropout(0.1) → BiLSTM(64) → Dropout(0.1) → Attention(tanh) → Flatten → Dense(1) |
| 第5章 | N-BEATSx | PyTorch 实现，stack=[trend, seasonality]，通过适配器提供 Keras 风格接口 |

### 与原始代码的已知差异（有意改进）

| # | 改进项 | 说明 |
|---|--------|------|
| 1 | Adam 优化器正确传参 | 原始代码 `Adam(lr=lr)` 创建后未使用，`model.compile(optimizer='adam')` 用字符串忽略传入 lr，实际始终用 Keras 默认 lr=0.001。重组代码将 Adam 对象正确传入 `model.compile()` |
| 2 | Adam gradient clipping | `clipnorm=1.0`，防止高学习率下梯度爆炸 |
| 3 | Force2 独立 scaler | 原始代码 `Force2 = Force1_scaler.fit_transform(...)` 错误地用 Force1 归一化器处理 Force2 数据。重组代码为每个变量使用独立 scaler |
| 4 | BPNN 输出层连接 | 原始 `Model_NN`（Functional API）输出层错误连接至 D_layer2 而非 D_layer3（第三隐藏层实际为死代码）。重组代码正确连接所有隐藏层 |
| 5 | XGBoost 正式启用 | 原始代码中 XGBoost 被定义但注释掉未使用，重组作为正式对比选项 |
| 6 | Roll/Yaw 未乘 1e6 | 核心实验不使用 Roll/Yaw 作为输入特征，故未实现原始 4.3/4.4 节中 `*1e6` 的缩放处理 |
| 7 | HGBRT loss 参数名 | `least_squares` → `squared_error`（sklearn API 升级，功能等价） |
| 8 | N-BEATSx 集成 | 原始代码作为独立测试存在，重组通过适配器集成到模型对比流程中 |
| 9 | GRU 排除出模型对比 | 原始代码 GRU 未在模型对比实验中使用（因高学习率下 NaN 问题，Adam bug 掩盖了此问题） |

---

## 原始文件保护

- `第三章/`、`第四章/`、`第五章/` 三个原始目录**未被修改或删除**
- 原始 notebook（`.ipynb`）、脚本（`.py`）、数据/结果（`.csv`）均完整保留
- 数据文件复制自原始目录：`t_1.csv` 来自 `第三章/`，`t_2_11.2_50.csv` 来自 `第四章/4.2/`
- 后续修改应优先修改 `src/cage_predict/` 中的工程化代码

---

## 常见问题

### TensorFlow / Keras 版本兼容

本项目使用 Keras 2 API（`from keras.layers import ...`）。推荐 `tensorflow>=2.4,<2.16`。`_build_adam()` 兼容 `lr` / `learning_rate` 两种参数名。已验证 TF 2.10.0 + Keras 2.10.0。

### GPU / CPU 差异

GPU 与 CPU 训练结果可能因浮点精度存在微小差异。设置 `CUDA_VISIBLE_DEVICES=-1` 可强制 CPU 模式。

### 直接运行原始 .py 文件

原始目录中的 `.py` 文件由 Jupyter Notebook 导出（含 `# In[1]:` 等标记），作为独立脚本运行可能失败。请使用 `python -m cage_predict` CLI 入口。

### 可选依赖

- `xgboost`：第4.2节 XGBoost 对比实验需要
- `torch`：第5章 N-BEATSx 模型对比需要
- `paddlehub`、`rembg`、`paddlepaddle`：仅出现在原始 `第三章/第三章（下：3.4~）.ipynb` 末尾图像处理代码中，与预测主流程**无关**，不需安装

### 清理缓存

```bash
find . -type d -name "__pycache__" -exec rm -rf {} +
find . -type d -name ".pytest_cache" -exec rm -rf {} +
rm -rf src/cage_predict.egg-info build dist
```
