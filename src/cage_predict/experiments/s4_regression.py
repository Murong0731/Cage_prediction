"""第4.2节 —— HGBRT / BPNN 系泊力回归预测。

预测目标：从运动变量（Surge, Heave, Pitch 等）预测系泊力（Force1 / Force2）。
使用模型：HGBRT（直方图梯度提升回归树）/ BPNN（反向传播神经网络）。

与第3章的关键区别：
  - 第3章是时间序列预测（输入过去的波高 → 输出当前的运动）
  - 第4.2节是回归预测（输入当前的运动量 → 输出当前的力）
  - 不使用时间滑动窗口，而是用当前时刻的多维特征直接回归

这相当于在问："给定当前的网箱姿态，系泊缆受力多大？"

原始代码参考：s4/4.2/4.2.py
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from ..config import load_config, validate_or_raise
from ..data import (
    apply_minmax_scaler,
    inverse_scale,
    load_csv_data,
    save_predictions_csv,
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


def run(config_path: str, smoke_test: bool = False, output_dir: str | None = None) -> None:
    """运行第4.2节回归实验。

    实验流程（8步）：
      1. 加载 CSV → 2. 归一化特征 → 3. 训练/验证集切分
      → 4. 训练 BPNN 或 HGBRT → 5. 预测 → 6. 反归一化
      → 7. 评估 → 8. 保存 CSV + 图表

    HGBRT vs BPNN：
      - HGBRT：基于树的集成方法，自动处理特征交互，需要网格搜索调参
      - BPNN：神经网络方法，可以学习复杂的非线性关系，需要更多调参

    参数
    ----------
    config_path : YAML 配置文件路径。
    smoke_test : 是否运行烟雾测试。
    output_dir : 如果提供，覆盖 output.output_dir，用于重定向到临时目录。
    """
    cfg = load_config(config_path, smoke_test=smoke_test, output_dir_override=output_dir)
    validate_or_raise(cfg, task_name="s4_regression")
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    out_cfg = cfg["output"]

    set_seed(train_cfg.get("random_seed", 42))

    target = data_cfg["target"]                # 预测目标，如 "Force1"
    features = data_cfg["input_features"]      # 输入特征，如 ["Surge", "Heave", "Pitch"]
    model_name = model_cfg["name"]
    train_start = data_cfg["train_start"]
    train_end = data_cfg["train_end"]
    valid_end = data_cfg["valid_end"]
    output_dir = Path(out_cfg["output_dir"])

    logger.info("=== 第4.2节: 系泊力回归预测 ===")
    logger.info("数据: %s", data_cfg["data_path"])
    logger.info("特征: %s | 目标: %s | 模型: %s", features, target, model_name)
    if smoke_test:
        logger.info("*** 烟雾测试模式 ***")

    # ── 第1步：加载并归一化数据 ─────────────────────────────────────────
    df = load_csv_data(data_cfg["data_path"])
    data_array = np.hstack([
        np.array(df)[:, 1:8],   # 运动数据
        np.array(df)[:, 8:10],  # 力数据
    ])
    logger.info("数据已加载: %d 行, %d 列", *data_array.shape)

    # 每个输入特征独立归一化（与原始代码一致）
    x_cols = [_COLUMN_MAP[f] for f in features]
    x_scaled_list, x_scalers = [], []
    for col in x_cols:
        s, sc = apply_minmax_scaler(data_array[:, col:col + 1])
        x_scaled_list.append(s)
        x_scalers.append(sc)

    x_train_raw = np.hstack(x_scaled_list)  # 拼接所有归一化后的特征

    # 归一化目标变量
    t_col = _COLUMN_MAP[target]
    y_scaled, y_scaler = apply_minmax_scaler(data_array[:, t_col:t_col + 1])

    # ── 第2步：训练/验证集切分（扁平特征，无时间窗口） ──────────────────
    # 注意：这里不使用滑动窗口，因为回归预测用的是当前时刻的特征
    train_x, train_y, valid_x, valid_y = split_train_valid(
        x_train_raw, y_scaled,
        train_start=train_start, train_end=train_end, valid_end=valid_end,
    )
    logger.info("训练集: X=%s y=%s | 验证集: X=%s y=%s",
                train_x.shape, train_y.shape, valid_x.shape, valid_y.shape)

    # ── 第3步：构建并训练模型 ─────────────────────────────────────────
    suffix = "_smoke" if smoke_test else ""

    if model_name == "bpnn":
        from ..models.bpnn import build_bpnn_model

        model = build_bpnn_model(
            input_dim=train_x.shape[1],               # 输入维度 = 特征数量
            hidden_units=model_cfg.get("bpnn_units", [150, 150, 150]),
            activation=model_cfg.get("bpnn_activation", "tanh"),
            learning_rate=train_cfg["learning_rate"],
            use_functional_api=True,                  # 4.2 节使用 Functional API
        )
        history = model.fit(
            train_x, train_y,
            epochs=train_cfg["epochs"],
            batch_size=train_cfg["batch_size"],
            validation_data=(valid_x, valid_y),
            verbose=2,
            shuffle=False,  # 时序相关数据不打乱
        )
        pre_y = model.predict(valid_x, verbose=0)
        logger.info("BPNN 训练完成 —— 最终损失: %.6f", history.history["loss"][-1])

        if out_cfg.get("save_figures", True):
            fig_dir = ensure_dir(output_dir / "figures")
            plot_loss_curve(
                history.history["loss"],
                history.history.get("val_loss"),
                save_path=fig_dir / f"{target.lower()}_bpnn_loss{suffix}.png",
            )

    elif model_name == "hgbdt":
        from ..models.hgbdt import build_and_fit_hgbdt, predict_hgbdt

        # HGBRT 通过网格搜索自动调优超参数
        model, best_params = build_and_fit_hgbdt(
            train_x, train_y,
            smoke_test=smoke_test,
            random_state=train_cfg.get("random_seed", 42),
        )
        pre_y = predict_hgbdt(model, valid_x)
        logger.info("HGBRT 训练完成 —— 最佳参数=%s", best_params)

    elif model_name == "xgboost":
        from ..models.xgboost import build_and_fit_xgboost, predict_xgboost

        model, best_params = build_and_fit_xgboost(
            train_x, train_y,
            smoke_test=smoke_test,
            random_state=train_cfg.get("random_seed", 42),
        )
        pre_y = predict_xgboost(model, valid_x)
        logger.info("XGBoost 训练完成 —— 最佳参数=%s", best_params)

    else:
        raise ValueError(f"未知的模型名称: {model_name}。请选择 'bpnn'、'hgbdt' 或 'xgboost'。")

    # ── 第4步：反归一化并评估 ──────────────────────────────────────────
    fan_real = inverse_scale(valid_y, y_scaler)
    fan_pred = inverse_scale(pre_y, y_scaler)

    metrics = evaluate_predictions(fan_real, fan_pred, use_sklearn=True)
    print_metrics(metrics, title=f"{target}_{model_name}")

    # ── 第5步：保存结果 ─────────────────────────────────────────────────
    if out_cfg.get("save_predictions", True):
        ensure_dir(output_dir)
        # 特征名缩写：如 Surge+Pitch+Heave → SPH
        feature_tag = "".join(f[:1] for f in features)
        csv_path = output_dir / f"{target.lower()}_{model_name}_{feature_tag}{suffix}.csv"
        save_predictions_csv(
            csv_path, fan_real, fan_pred,
            experiment_name="s4_regression",
            target=target,
            model_name=model_name,
        )
        logger.info("预测结果已保存 (%d 行) → %s", len(fan_real), csv_path)

    if out_cfg.get("save_figures", True) and model_name != "hgbdt":
        # HGBRT 没有损失曲线（树模型不是迭代训练）
        fig_dir = ensure_dir(output_dir / "figures")
        plot_prediction_curve(
            fan_real, fan_pred,
            title=f"{target} — {model_name} (特征={features}){suffix}",
            n_points=min(200, len(fan_real)),
            save_path=fig_dir / f"{target.lower()}_{model_name}_prediction{suffix}.png",
        )

    logger.info("输出目录: %s", output_dir.resolve())
    logger.info("=== 第4.2节完成: RMSE=%.6f  Acc=%.6f ===", metrics["rmse"], metrics["acc"])
