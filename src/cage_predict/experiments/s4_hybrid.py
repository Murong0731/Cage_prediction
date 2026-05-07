"""第4.4节 —— 混合联合预测（波浪 → 运动 → 系泊力）。

这是论文中最复杂的实验设计，采用两级串联预测架构：

  第一阶段（Stage 1）：LSTM 从波浪 H 预测运动（Heave, Surge, Pitch）
  第二阶段（Stage 2）：BPNN 从预测的运动（非真实运动！）预测系泊力

关键创新点：第二阶段使用的是第一阶段"预测出来"的运动值，而不是真实的
           运动测量值。这使得整个流程可以仅从波浪数据出发，端到端地
           预测系泊力，而不需要实际测量运动。

如果第二阶段使用"真实"运动值，就等价于第4.2节的回归预测。
而这里使用"预测"运动值，模拟的是实际工程中只有波浪测量数据、没有
运动测量数据的情况。

工程意义：
  养殖网箱在实际运营中，波浪数据相对容易获取（浮标、卫星），
  但六自由度运动的精确测量需要昂贵的惯性导航设备。
  如果能仅从波浪预测系泊力，就能大幅降低监测成本。

原始代码参考：s4/4.4/4.4.py
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

_COLUMN_MAP = {
    "H": 0, "Surge": 1, "Sway": 2, "Heave": 3,
    "Roll": 4, "Pitch": 5, "Yaw": 6, "Force1": 7, "Force2": 8,
}


def run(config_path: str, smoke_test: bool = False) -> None:
    """运行第4.4节混合联合预测实验。

    实验流程（6步）：
      1. 归一化所有变量
      2. 第一阶段：训练 3 个独立的 LSTM（H → 每个运动分量）
      3. 第一阶段：在训练集和验证集上预测运动值
      4. 第二阶段：构建力数据集 → 用预测的运动值对齐
      5. 第二阶段：用 BPNN 训练（预测的运动 → 力）
      6. 评估 → 保存输出

    为什么每个运动分量用单独的 LSTM？
      - 不同运动分量对波浪的响应特性不同
      - Heave（垂荡）主要受波高的垂直分量影响
      - Surge（纵荡）主要受波浪的水平推力影响
      - 各自独立建模可以获得更好的精度

    参数
    ----------
    config_path : YAML 配置文件路径。
    smoke_test : 是否运行烟雾测试。
    """
    cfg = load_config(config_path, smoke_test=smoke_test)
    data_cfg = cfg["data"]
    s1_cfg = cfg["stage1"]      # 第一阶段配置
    s2_cfg = cfg["stage2"]      # 第二阶段配置
    out_cfg = cfg["output"]

    set_seed(42)

    wave_feat = data_cfg["wave_feature"]         # 波浪特征（如 "H"）
    motion_targets = data_cfg["motion_targets"]  # 运动目标列表，如 ["Surge", "Heave", "Pitch"]
    mooring_target = data_cfg["mooring_target"]  # 系泊力目标（如 "Force1"）
    look_back = data_cfg["look_back"]
    skip_rows = data_cfg.get("skip_rows", 0)
    train_start = data_cfg["train_start"]
    train_end = data_cfg["train_end"]
    valid_end = data_cfg["valid_end"]
    output_dir = Path(out_cfg["output_dir"])

    logger.info("=== 第4.4节: 混合联合预测 ===")
    logger.info("第一阶段: %s → %s  |  第二阶段: %s → %s  |  回顾窗口: %d",
                wave_feat, motion_targets, motion_targets, mooring_target, look_back)
    if smoke_test:
        logger.info("*** 烟雾测试模式 ***")

    # ── 第0步：加载并归一化所有变量 ────────────────────────────────────
    df = load_csv_data(data_cfg["data_path"])
    data_array = np.hstack([np.array(df)[:, 1:8], np.array(df)[:, 8:10]])
    all_vars = [wave_feat] + list(motion_targets) + [mooring_target]

    # 每个变量独立归一化，方便后续各自反归一化
    scaled = {}
    scalers = {}
    for name in all_vars:
        s, sc = apply_minmax_scaler(data_array[:, _COLUMN_MAP[name]:_COLUMN_MAP[name] + 1])
        scaled[name] = s
        scalers[name] = sc

    # config 中的索引值直接对应 x/y 数组的行号，与原始代码中的用法完全一致
    t_start = train_start
    t_end = train_end
    v_end = valid_end

    # ── 第1步：第一阶段 —— 为每个运动目标训练 LSTM(H → 运动) ────────
    from ..models.lstm import build_lstm_model

    pretrain_motions = {}   # 训练集上的运动预测值
    pre_motions = {}        # 验证集上的运动预测值

    for motion in motion_targets:
        logger.info("—— 第一阶段: 训练 %s 的 LSTM ——", motion)

        # 构建该运动的序列数据集：[波浪特征, 运动变量]
        combined = np.hstack([scaled[wave_feat], scaled[motion]])[skip_rows:, :]
        proc = deal_data2(combined, features_number=2, time_steps=look_back)
        mx, my = split_sequence(proc, n_past=look_back)

        train_mx, train_my, valid_mx, valid_my = split_train_valid(
            mx, my, train_start=t_start, train_end=t_end, valid_end=v_end,
        )

        # 每个运动分量的学习率不同（原始代码中的经验设定）
        # Heave=0.1, Surge=1.3, Pitch=0.35 —— 因为不同运动对波浪的敏感度不同
        lr_map = s1_cfg.get("learning_rates", {})
        lr = lr_map.get(motion, s1_cfg.get("learning_rate", 0.1))

        model = build_lstm_model(
            input_shape=(train_mx.shape[1], train_mx.shape[2]),
            num_layers=s1_cfg.get("num_layers", 1),
            units=s1_cfg.get("units", [25]),
            dropout=s1_cfg.get("dropout", 0.0),
            activation=s1_cfg.get("activation", "tanh"),
            learning_rate=lr,
        )
        model.fit(
            train_mx, train_my,
            epochs=s1_cfg["epochs"],
            batch_size=s1_cfg["batch_size"],
            validation_data=(valid_mx, valid_my),
            verbose=2, shuffle=False,
        )

        # 保存预测的运动值（注意：不是真实值！第二阶段用这些预测值作为输入）
        pretrain_motions[motion] = model.predict(train_mx, verbose=0)
        pre_motions[motion] = model.predict(valid_mx, verbose=0)

        logger.info("第一阶段 [%s] 完成: 训练预测=%s 验证预测=%s",
                    motion, pretrain_motions[motion].shape, pre_motions[motion].shape)

    # ── 第2步：第二阶段 —— BPNN(预测的运动 → 力) ──────────────────────
    logger.info("—— 第二阶段: 训练 %s 的 BPNN ——", mooring_target)

    # 构建力数据集以获取对齐的训练/验证标签
    force_combined = np.hstack(
        [scaled[name] for name in motion_targets] + [scaled[mooring_target]]
    )[skip_rows:, :]
    f_proc = deal_data2(force_combined, features_number=len(motion_targets) + 1, time_steps=look_back)
    fx, fy = split_sequence(f_proc, n_past=look_back)
    _, _, _, valid_fy = split_train_valid(fx, fy, t_start, t_end, v_end)

    # ★ 关键设计：第二阶段输入 = 预测的运动值（不是真实值！）
    # 这模拟了实际工程场景：只有波浪数据，运动是算出来的
    train_x_s2 = np.hstack([pretrain_motions[m] for m in motion_targets])
    valid_x_s2 = np.hstack([pre_motions[m] for m in motion_targets])
    train_y_s2 = fy[t_start:t_end].reshape(-1, 1)

    logger.info("第二阶段: train_X=%s valid_X=%s", train_x_s2.shape, valid_x_s2.shape)

    # 训练 BPNN
    from ..models.bpnn import build_bpnn_model

    nn_model = build_bpnn_model(
        input_dim=train_x_s2.shape[1],                     # 输入维度 = 运动分量数量
        hidden_units=s2_cfg.get("bpnn_units", [150, 150, 150]),
        activation=s2_cfg.get("bpnn_activation", "tanh"),
        learning_rate=s2_cfg["learning_rate"],
        use_functional_api=True,
    )
    history = nn_model.fit(
        train_x_s2, train_y_s2,
        epochs=s2_cfg["epochs"],
        batch_size=s2_cfg["batch_size"],
        validation_data=(valid_x_s2, valid_fy),
        verbose=2, shuffle=False,
    )

    pre_force = nn_model.predict(valid_x_s2, verbose=0)

    # ── 第3步：反归一化并评估 ──────────────────────────────────────────
    t_scaler = scalers[mooring_target]
    fan_real = inverse_scale(valid_fy, t_scaler)
    fan_pred = inverse_scale(pre_force, t_scaler)

    metrics = evaluate_predictions(fan_real, fan_pred, use_sklearn=True)
    print_metrics(metrics, title=f"{mooring_target}_hybrid")

    # ── 第5步：保存结果 ─────────────────────────────────────────────────
    suffix = "_smoke" if smoke_test else ""

    if out_cfg.get("save_predictions", True):
        ensure_dir(output_dir)
        csv_path = output_dir / f"{mooring_target.lower()}_hybrid_h{look_back}{suffix}.csv"
        save_results_csv(csv_path, fan_real, fan_pred)
        logger.info("预测结果已保存 (%d 行) → %s", len(fan_real), csv_path)

    if out_cfg.get("save_figures", True):
        fig_dir = ensure_dir(output_dir / "figures")
        plot_prediction_curve(
            fan_real, fan_pred,
            title=f"{mooring_target} — 混合预测 (H→运动→力, 回顾窗口={look_back}){suffix}",
            n_points=min(200, len(fan_real)),
            save_path=fig_dir / f"{mooring_target.lower()}_hybrid_prediction{suffix}.png",
        )
        plot_loss_curve(
            history.history["loss"],
            history.history.get("val_loss"),
            save_path=fig_dir / f"{mooring_target.lower()}_hybrid_loss{suffix}.png",
        )

    logger.info("输出目录: %s", output_dir.resolve())
    logger.info("=== 第4.4节完成: RMSE=%.6f  Acc=%.6f ===", metrics["rmse"], metrics["acc"])
