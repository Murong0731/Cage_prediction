"""第3章 —— 基于 LSTM 的网箱运动响应预测。

预测目标：从波浪高度 H 预测 Heave（垂荡）/ Surge（纵荡）/ Pitch（纵摇）。
使用模型：LSTM / GRU / BiLSTM。

实验流程（8步）：
  1. 加载 CSV 数据
  2. 将 H 和目标运动变量分别归一化到 [-1, 1]
  3. 将时间序列转换为监督学习序列格式
  4. 按时间顺序切分训练集和验证集
  5. 构建并训练 LSTM/GRU/BiLSTM 模型
  6. 在验证集上预测
  7. 反归一化预测结果并计算评估指标
  8. 保存预测值 CSV 和可视化图表

原始代码参考：s3/3.3.py, s3/3.4.py
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from ..config import load_config
from ..data import (
    apply_minmax_scaler,
    deal_data2,
    inverse_scale,
    load_csv_data,
    save_results_csv,
    split_sequence,
    split_train_valid,
)
from ..metrics import evaluate_predictions, print_metrics
from ..plotting import plot_loss_curve, plot_prediction_curve
from ..utils import ensure_dir, set_seed

logger = logging.getLogger(__name__)

# 数据列索引映射 —— t_1.csv / t_2_11.2_50.csv 中的列顺序（去掉第一列 T 后）
# 数据共9列：H(波高), Surge(纵荡), Sway(横荡), Heave(垂荡), Roll(横摇),
#            Pitch(纵摇), Yaw(艏摇), Force1(系泊力1), Force2(系泊力2)
_COLUMN_MAP = {
    "H": 0,
    "Surge": 1,
    "Sway": 2,
    "Heave": 3,
    "Roll": 4,
    "Pitch": 5,
    "Yaw": 6,
    "Force1": 7,
    "Force2": 8,
}


def _build_model(model_cfg: dict, input_shape: tuple[int, int]):
    """根据配置构建模型（lstm / gru / bilstm）。

    参数
    ----------
    model_cfg : 模型配置字典，包含 name, num_layers, units 等参数。
    input_shape : (时间步数, 特征数) 输入形状。
    """
    name = model_cfg.get("name", "lstm")
    lr = model_cfg.get("learning_rate", 0.01)

    if name == "lstm":
        from ..models.lstm import build_lstm_model

        return build_lstm_model(
            input_shape=input_shape,
            num_layers=model_cfg.get("num_layers", 3),
            units=model_cfg.get("units", [25, 100, 100]),
            dropout=model_cfg.get("dropout", 0.3),
            activation=model_cfg.get("activation", "tanh"),
            learning_rate=lr,
        )
    elif name == "gru":
        from ..models.gru import build_gru_model

        return build_gru_model(
            input_shape=input_shape,
            units=model_cfg.get("units", [25, 100, 100]),
            dropout=model_cfg.get("dropout", 0.3),
            activation=model_cfg.get("activation", "tanh"),
            learning_rate=lr,
        )
    elif name == "bilstm":
        from ..models.bilstm import build_bilstm_model

        return build_bilstm_model(
            input_shape=input_shape,
            lstm_units=model_cfg.get("units", [25])[0],
            learning_rate=lr,
        )
    else:
        raise ValueError(f"未知的模型名称: {name}")


def run(config_path: str, smoke_test: bool = False) -> None:
    """运行第3章运动响应预测实验。

    实验假设：波浪是网箱运动的主要驱动力，因此只用波高 H 作为输入，
    预测六自由度运动（Surge, Sway, Heave, Roll, Pitch, Yaw）中的某一个。

    核心思想：
      - 输入 = 过去的波高序列
      - 输出 = 当前时刻的目标运动值
      - 这是一个单输入单输出的时间序列预测问题

    参数
    ----------
    config_path : YAML 配置文件的路径。
    smoke_test : 如果为 True，运行快速烟雾测试（减少数据量和训练轮数）。
    """
    cfg = load_config(config_path, smoke_test=smoke_test)
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    out_cfg = cfg["output"]

    set_seed(train_cfg.get("random_seed", 42))

    target = data_cfg["target"]            # 预测目标，如 "Heave"
    look_back = data_cfg["look_back"]      # 回顾窗口大小（使用过去多少步的数据）
    skip_rows = data_cfg.get("skip_rows", 0)
    train_size = data_cfg["train_size"]    # 训练集大小
    valid_size = data_cfg["valid_size"]    # 验证集大小
    train_start = data_cfg.get("train_start", 0)
    output_dir = Path(out_cfg["output_dir"])

    logger.info("=== 第3章: LSTM 运动响应预测 ===")
    logger.info("数据: %s", data_cfg["data_path"])
    logger.info("目标: %s | 模型: %s | 回顾窗口: %d | 训练轮数: %d",
                target, model_cfg["name"], look_back, train_cfg["epochs"])
    if smoke_test:
        logger.info("*** 烟雾测试模式 —— 使用减少的数据/轮数/输出 ***")

    # ── 第1步：加载数据 ─────────────────────────────────────────────────
    df = load_csv_data(data_cfg["data_path"])
    # 拼接运动数据（第1-7列）和力数据（第8-9列）
    data_array = np.hstack([
        np.array(df)[:, 1:8],   # H..Yaw（7列运动数据）
        np.array(df)[:, 8:10],  # Force1, Force2（2列力数据）
    ])
    logger.info("数据已加载: %d 行, %d 列", *data_array.shape)

    target_col = _COLUMN_MAP[target]
    h_col = _COLUMN_MAP["H"]

    # 第2步：分别归一化 H 和目标变量（与原始代码完全一致）
    # 每个变量独立归一化的原因是它们有不同的物理量纲
    h_scaled, h_scaler = apply_minmax_scaler(data_array[:, h_col:h_col + 1])
    t_scaled, t_scaler = apply_minmax_scaler(data_array[:, target_col:target_col + 1])

    # ── 第3步：创建监督学习序列 ────────────────────────────────────────
    # 将归一化后的 H 和目标变量拼接，形成 [H_scaled, Target_scaled] 数据
    combined = np.hstack([h_scaled, t_scaled])
    combined = combined[skip_rows:, :]  # 跳过前导行（数据质量较差的部分）

    # deal_data2 将时间序列转换为监督学习格式，只保留第一个输入特征的滞后值
    processed = deal_data2(combined, features_number=2, time_steps=look_back)
    # split_sequence 将二维数据切分为 (X, y) 滑动窗口
    x, y = split_sequence(processed, n_past=look_back)
    logger.info("序列数据: X=%s y=%s", x.shape, y.shape)

    # 第4步：按时间顺序切分训练集和验证集
    train_x, train_y, valid_x, valid_y = split_train_valid(
        x, y, train_start=train_start, train_end=train_size, valid_end=valid_size,
    )
    logger.info("训练集: X=%s y=%s | 验证集: X=%s y=%s",
                train_x.shape, train_y.shape, valid_x.shape, valid_y.shape)

    # ── 第5步：构建模型 ────────────────────────────────────────────────────
    input_shape = (train_x.shape[1], train_x.shape[2])
    lr_map = train_cfg.get("learning_rates", {})
    lr = lr_map.get(target, train_cfg.get("learning_rate", 0.01))
    model = _build_model({**model_cfg, "learning_rate": lr}, input_shape)

    # ── 第6步：训练模型 ────────────────────────────────────────────────────
    # verbose=2: 每个 epoch 输出一行进度（比 verbose=1 的进度条更适合日志）
    history = model.fit(
        train_x, train_y,
        epochs=train_cfg["epochs"],
        batch_size=train_cfg["batch_size"],
        validation_data=(valid_x, valid_y),
        verbose=2,
        shuffle=train_cfg.get("shuffle", False),  # 时序数据通常不打乱
    )
    loss = history.history["loss"][-1]
    val_loss = history.history.get("val_loss", [float("nan")])[-1]
    logger.info("训练完成 —— 最终损失: %.6f, 验证损失: %.6f", loss, val_loss)

    # ── 第7步：预测 ────────────────────────────────────────────────────────
    pre_y = model.predict(valid_x, verbose=0)

    # ── 第8步：反归一化并评估 ──────────────────────────────────────────────
    # 关键：用目标变量的 scaler 进行反归一化，恢复原始物理量纲
    fan_real = inverse_scale(valid_y, t_scaler)
    fan_pred = inverse_scale(pre_y, t_scaler)

    metrics = evaluate_predictions(fan_real, fan_pred, use_sklearn=True)
    print_metrics(metrics, title=target)

    # ── 第9步：保存结果 ────────────────────────────────────────────────────
    suffix = "_smoke" if smoke_test else ""
    n_plot = min(200, len(fan_real))  # 限制绘图点数（烟雾测试中数据较少）

    if out_cfg.get("save_predictions", True):
        ensure_dir(output_dir)
        csv_path = output_dir / f"{target.lower()}_h{look_back}_{model_cfg['name']}{suffix}.csv"
        save_results_csv(csv_path, fan_real, fan_pred)
        logger.info("预测结果已保存 (%d 行) → %s", len(fan_real), csv_path)

    if out_cfg.get("save_figures", True):
        fig_dir = ensure_dir(output_dir / "figures")
        plot_prediction_curve(
            fan_real, fan_pred,
            title=f"{target} — {model_cfg['name']} (回顾窗口={look_back}){suffix}",
            n_points=n_plot,
            save_path=fig_dir / f"{target.lower()}_prediction{suffix}.png",
        )
        plot_loss_curve(
            history.history["loss"],
            history.history.get("val_loss"),
            save_path=fig_dir / f"{target.lower()}_loss{suffix}.png",
        )

    logger.info("输出目录: %s", output_dir.resolve())
    logger.info("=== 第3章完成: RMSE=%.6f  Acc=%.6f ===", metrics["rmse"], metrics["acc"])
