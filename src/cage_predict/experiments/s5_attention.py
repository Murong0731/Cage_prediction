"""第5章 —— CNN-BiLSTM-Attention 两阶段联合预测。

这是论文最后一章的核心实验，采用两级串联预测架构：

  第一阶段（Stage 1）：Attention / BiLSTM / LSTM / GRU 从波浪 H 预测
                     未来运动（Heave、Surge、Pitch）。
  第二阶段（Stage 2）：BPNN 从"预测的"运动值（非真实值）预测系泊力
                     （Force1 / Force2）。

与第4.4节（混合模型）的异同：
  - 相同：都采用两阶段（运动→力）的联合预测策略
  - 不同：第5章 Stage 1 可以使用多种模型（Attention / BiLSTM / LSTM / GRU），
         并进行模型对比；第4.4节 Stage 1 仅使用 LSTM

model_compare 模式会在同一数据集上对比 Stage 1 的多个模型，
以验证 CNN-BiLSTM-Attention 的优越性。

原始代码参考：s5/0 5.2/Attention/第五章（上：5.1~5.2混合注意力）.ipynb
              s5/0 5.2/BILSTM - 30epoch/第五章（上：5.1~5.2BiLSTM）.ipynb
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


def _build_stage1_model(name: str, s1_cfg: dict, input_shape: tuple[int, int], lr: float):
    """按名称构建 Stage 1 模型（用于运动预测）。

    参数
    ----------
    name : 模型名称 (lstm / gru / bilstm / attention)。
    s1_cfg : Stage 1 配置字典。
    input_shape : (时间步数, 特征数) 输入形状。
    lr : 学习率。
    """
    if name == "lstm":
        from ..models.lstm import build_lstm_model
        return build_lstm_model(
            input_shape=input_shape,
            num_layers=s1_cfg.get("lstm_num_layers", 1),
            units=s1_cfg.get("lstm_units", [25]),
            dropout=s1_cfg.get("lstm_dropout", 0.0),
            activation=s1_cfg.get("lstm_activation", "tanh"),
            learning_rate=lr,
        )
    elif name == "gru":
        from ..models.gru import build_gru_model
        return build_gru_model(
            input_shape=input_shape,
            units=s1_cfg.get("gru_units", [25, 100, 100]),
            dropout=s1_cfg.get("gru_dropout", 0.3),
            activation=s1_cfg.get("gru_activation", "tanh"),
            learning_rate=lr,
        )
    elif name == "bilstm":
        from ..models.bilstm import build_bilstm_model
        return build_bilstm_model(
            input_shape=input_shape,
            lstm_units=s1_cfg.get("bilstm_units", 25),
            learning_rate=lr,
        )
    elif name == "attention":
        from ..models.attention import build_attention_model
        return build_attention_model(
            input_shape=input_shape,
            conv_filters=s1_cfg.get("attn_conv_filters", 64),
            conv_kernel=s1_cfg.get("attn_conv_kernel", 1),
            lstm_units=s1_cfg.get("attn_lstm_units", 64),
            dropout_rate=s1_cfg.get("attn_dropout", 0.1),
            attention_mode=s1_cfg.get("attn_mode", "permute"),
            learning_rate=lr,
        )
    elif name == "nbeatsx":
        from ..models.nbeatsx_adapter import build_nbeatsx_model
        return build_nbeatsx_model(
            input_shape=input_shape,
            output_size=1,
            stack_types=s1_cfg.get("nbeatsx_stack_types"),
            n_blocks=s1_cfg.get("nbeatsx_n_blocks"),
            n_layers=s1_cfg.get("nbeatsx_n_layers"),
            n_hidden=s1_cfg.get("nbeatsx_n_hidden"),
            activation=s1_cfg.get("nbeatsx_activation", "relu"),
            batch_normalization=s1_cfg.get("nbeatsx_batch_normalization", True),
            dropout_prob_theta=s1_cfg.get("nbeatsx_dropout", 0.1),
            shared_weights=s1_cfg.get("nbeatsx_shared_weights", False),
            learning_rate=lr,
            lr_decay=s1_cfg.get("nbeatsx_lr_decay", 0.5),
            n_lr_decay_steps=s1_cfg.get("nbeatsx_lr_decay_steps", 5),
            weight_decay=s1_cfg.get("nbeatsx_weight_decay", 0.0),
        )
    else:
        raise ValueError(f"未知的 Stage 1 模型名称: {name}")


def run(config_path: str, smoke_test: bool = False) -> None:
    """运行第5章两阶段联合预测 + 模型比较实验。

    实验流程：
      1. 归一化所有变量（H、Heave、Surge、Pitch、Force1/Force2）
      2. Stage 1：对每个运动分量，用选定的模型从 H 预测运动
      3. Stage 1：在训练集和验证集上预测运动值
      4. Stage 2：用 BPNN 从预测的运动值预测系泊力
      5. 反归一化并评估
      6. 如果 model_compare 模式，汇总比较所有模型

    参数
    ----------
    config_path : YAML 配置文件路径。
    smoke_test : 是否运行烟雾测试。
    """
    cfg = load_config(config_path, smoke_test=smoke_test)
    data_cfg = cfg["data"]
    s1_cfg = cfg["stage1"]
    s2_cfg = cfg["stage2"]
    model_cfg = cfg["model"]
    out_cfg = cfg["output"]
    mode = cfg.get("mode", "model_compare")

    set_seed(42)

    wave_feat = data_cfg["wave_feature"]
    motion_targets = data_cfg["motion_targets"]
    mooring_target = data_cfg["mooring_target"]
    look_back = data_cfg["look_back"]
    horizon = data_cfg.get("horizon", 10)
    skip_rows = data_cfg["skip_rows"]
    train_start = data_cfg["train_start"]
    train_end = data_cfg["train_end"]
    valid_end = data_cfg["valid_end"]

    # 各运动分量的训练起始索引（原始代码中 Surge 使用 4200，其余用 4500）
    motion_train_starts = s1_cfg.get("motion_train_starts", {})
    # Stage 2 统一对齐使用的 train_start
    align_start = train_start

    models_to_run = model_cfg.get("compare_models", ["attention"])
    suffix = "_smoke" if smoke_test else ""

    logger.info("=== 第5章: 两阶段联合预测 ===")
    logger.info("Stage 1: %s → %s  |  Stage 2: %s → %s", wave_feat, motion_targets, motion_targets, mooring_target)
    logger.info("模式: %s | 对比模型: %s | 回顾窗口: %d | 预测视野: %d", mode, models_to_run, look_back, horizon)
    if smoke_test:
        logger.info("*** 烟雾测试模式 ***")

    # ══════════════════════════════════════════════════════════════════════
    # 第0步：加载并独立归一化所有变量
    # ══════════════════════════════════════════════════════════════════════
    df = load_csv_data(data_cfg["data_path"])
    data_array = np.hstack([np.array(df)[:, 1:8], np.array(df)[:, 8:10]])
    logger.info("数据已加载: %d 行, %d 列", *data_array.shape)

    all_vars = [wave_feat] + list(motion_targets) + [mooring_target]
    scaled = {}
    scalers = {}
    for name in all_vars:
        s, sc = apply_minmax_scaler(data_array[:, _COLUMN_MAP[name]:_COLUMN_MAP[name] + 1])
        scaled[name] = s
        scalers[name] = sc

    # deal_data2 的 time_steps 计算：
    #   原始代码范式: deal_data2(data, features, look_back + horizon - 1)
    #   然后 split_sequence(data, n_past=look_back)
    #   horizon = time_steps - n_past + 1
    time_steps = look_back + horizon - 1

    # ══════════════════════════════════════════════════════════════════════
    # Stage 1：为每个 Stage 1 模型类型运行完整的两阶段流程
    # ══════════════════════════════════════════════════════════════════════
    results_summary = {}
    t_scaler = scalers[mooring_target]

    for model_name in models_to_run:
        logger.info("━━━ Stage 1 模型: %s ━━━", model_name)

        # ── Stage 1: 预测每个运动分量 ──────────────────────────────────
        pretrain_motions = {}
        pre_motions = {}

        for motion in motion_targets:
            logger.info("  Stage 1: %s → %s", wave_feat, motion)

            # 每个运动分量可能有不同的训练起始索引
            m_start = motion_train_starts.get(motion, train_start)

            # 构建该运动的序列数据集: [H, Motion]
            combined = np.hstack([scaled[wave_feat], scaled[motion]])[skip_rows:, :]
            processed = deal_data2(combined, features_number=2, time_steps=time_steps)
            mx, my = split_sequence(processed, n_past=look_back)

            train_mx, train_my, valid_mx, valid_my = split_train_valid(
                mx, my,
                train_start=m_start,
                train_end=train_end,
                valid_end=valid_end,
            )

            # 每个运动分量的学习率不同（原始代码中的经验设定）
            # 原始代码对所有模型使用相同的原始学习率（公平对比）
            lr_map = s1_cfg.get("learning_rates", {})
            base_lr = lr_map.get(motion, s1_cfg.get("learning_rate", 0.1))
            lr_scales = s1_cfg.get("lr_scales", {})
            lr_scale = lr_scales.get(model_name, 1.0)
            lr = base_lr * lr_scale

            input_shape = (train_mx.shape[1], train_mx.shape[2])
            model = _build_stage1_model(model_name, s1_cfg, input_shape, lr)

            model.fit(
                train_mx, train_my,
                epochs=s1_cfg["epochs"],
                batch_size=s1_cfg["batch_size"],
                validation_data=(valid_mx, valid_my),
                verbose=2, shuffle=False,
            )

            pretrain_full = model.predict(train_mx, verbose=0)
            pre_motions[motion] = model.predict(valid_mx, verbose=0)

            # 如果该运动的 train_start 早于对齐起点，需要裁剪前面的预测
            # （原始代码中 Surge 用 4200 训练，然后 [300:, :] 对齐到 4500）
            trim = align_start - m_start
            if trim > 0:
                pretrain_motions[motion] = pretrain_full[trim:, :]
                logger.info("  Stage 1 [%s → %s] 完成: %s → %s (裁剪前%d行)",
                            wave_feat, motion, pretrain_full.shape, pre_motions[motion].shape, trim)
            else:
                pretrain_motions[motion] = pretrain_full
                logger.info("  Stage 1 [%s → %s] 完成: %s → %s",
                            wave_feat, motion, pretrain_motions[motion].shape, pre_motions[motion].shape)

        # ── Stage 2: BPNN 从预测的运动值预测系泊力 ────────────────────
        logger.info("  Stage 2: {%s} → %s", ",".join(motion_targets), mooring_target)

        # 构建力数据集获取对齐的标签
        force_cols = [scaled[m] for m in motion_targets] + [scaled[mooring_target]]
        force_combined = np.hstack(force_cols)[skip_rows:, :]
        f_proc = deal_data2(force_combined, features_number=len(motion_targets) + 1, time_steps=time_steps)
        fx, fy = split_sequence(f_proc, n_past=look_back)
        _, _, _, valid_fy = split_train_valid(fx, fy, align_start, train_end, valid_end)

        # ★ 关键设计：Stage 2 输入 = Stage 1 预测的运动值，而非真实测量值
        train_x_s2 = np.hstack([pretrain_motions[m] for m in motion_targets])
        valid_x_s2 = np.hstack([pre_motions[m] for m in motion_targets])
        train_y_s2 = fy[align_start:train_end].reshape(-1, 1)

        logger.info("  Stage 2: train_X=%s valid_X=%s", train_x_s2.shape, valid_x_s2.shape)

        from ..models.bpnn import build_bpnn_model

        nn_model = build_bpnn_model(
            input_dim=train_x_s2.shape[1],
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

        # ── 反归一化并评估 ──────────────────────────────────────────
        fan_real = inverse_scale(valid_fy, t_scaler)
        fan_pred = inverse_scale(pre_force, t_scaler)
        metrics = evaluate_predictions(fan_real, fan_pred, use_sklearn=True)
        print_metrics(metrics, title=f"{mooring_target}_{model_name}")
        results_summary[model_name] = metrics

        # ── 保存每个模型的输出 ──────────────────────────────────────
        if out_cfg.get("save_predictions", True):
            output_dir = Path(out_cfg["output_dir"])
            ensure_dir(output_dir)
            csv_path = output_dir / f"{mooring_target.lower()}_{model_name}_h{look_back}_hor{horizon}{suffix}.csv"
            save_results_csv(csv_path, fan_real, fan_pred)
            logger.info("  预测结果已保存 → %s", csv_path)

        if out_cfg.get("save_figures", True):
            fig_dir = ensure_dir(Path(out_cfg["output_dir"]) / "figures")
            plot_prediction_curve(
                fan_real, fan_pred,
                title=f"{mooring_target} — {model_name} (horizon={horizon}){suffix}",
                n_points=min(200, len(fan_real)),
                save_path=fig_dir / f"{mooring_target.lower()}_{model_name}_prediction{suffix}.png",
            )
            plot_loss_curve(
                history.history["loss"],
                history.history.get("val_loss"),
                save_path=fig_dir / f"{mooring_target.lower()}_{model_name}_loss{suffix}.png",
            )

    # ══════════════════════════════════════════════════════════════════════
    # 模型对比汇总
    # ══════════════════════════════════════════════════════════════════════
    logger.info("=== 模型对比汇总（Stage 1 模型 → Stage 2 %s） ===", mooring_target)
    logger.info("%-12s  %8s  %8s  %8s", "Stage 1 模型", "RMSE", "MAE", "Acc")
    for name, m in results_summary.items():
        logger.info("%-12s  %8.3f  %8.3f  %8.4f", name, m["rmse"], m["mae"], m["acc"])

    logger.info("输出目录: %s", Path(out_cfg["output_dir"]).resolve())
    logger.info("=== 第5章完成 ===")
