"""第4.3节 —— LSTM 时间序列系泊力预测。

预测目标：从运动历史 + 力历史预测当前时刻的系泊力（Force1 / Force2）。
使用模型：单层 LSTM。

与第4.2节的区别：
  - 第4.2节：纯回归（当前运动 → 当前力），不使用历史信息
  - 第4.3节：时间序列预测（历史运动+力 → 当前力），利用时间维度信息

实验假设：系泊力不仅和当前网箱姿态有关，还与之前的运动状态有关，
        即力是有"记忆"的（缆绳的张力-位移关系是非线性和滞后的）。

原始代码参考：s4/4.3/4.3.py
"""

from __future__ import annotations

import csv
import logging
from datetime import datetime
from pathlib import Path

import numpy as np

from ..config import load_config, validate_or_raise
from ..data import (
    apply_minmax_scaler,
    deal_data2,
    inverse_scale,
    load_csv_data,
    save_predictions_csv,
    split_sequence,
    split_train_valid,
)
from ..metrics import compute_rmse, compute_rtrapz, evaluate_predictions, print_metrics
from ..plotting import plot_loss_curve, plot_prediction_curve
from ..utils import ensure_dir, set_seed

logger = logging.getLogger(__name__)

_COLUMN_MAP = {
    "H": 0, "Surge": 1, "Sway": 2, "Heave": 3,
    "Roll": 4, "Pitch": 5, "Yaw": 6, "Force1": 7, "Force2": 8,
}


def run(config_path: str, smoke_test: bool = False, output_dir: str | None = None) -> None:
    """运行第4.3节 LSTM 系泊力时间序列预测实验。

    实验流程（9步）：
      1. 加载 CSV → 2. 归一化所有特征 → 3. 构建序列数据集
      → 4. 训练/验证切分 → 5. LSTM 训练 → 6. 预测
      → 7. 反归一化 → 8. 评估 → 9. 保存输出

    关键设计：
      - 输入特征列排在前面，目标列放在最后（split_sequence 的约定）
      - 使用单层 LSTM（原始代码中的默认配置）
      - 目标变量也可以是输入特征的一部分（自回归方式）

    参数
    ----------
    config_path : YAML 配置文件路径。
    smoke_test : 是否运行烟雾测试。
    output_dir : 如果提供，覆盖 output.output_dir，用于重定向到临时目录。
    """
    cfg = load_config(config_path, smoke_test=smoke_test, output_dir_override=output_dir)
    validate_or_raise(cfg, task_name="s4_lstm_mooring")
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    out_cfg = cfg["output"]

    set_seed(train_cfg.get("random_seed", 42))

    target = data_cfg["target"]               # 预测目标，如 "Force1"
    features = data_cfg["input_features"]     # 输入特征，如 ["Surge", "Heave", "Force1"]
    wave_known = data_cfg.get("wave_known", False)  # 是否利用波浪信息(H)作为输入特征
    look_back = data_cfg["look_back"]         # 回顾窗口大小
    skip_rows = data_cfg.get("skip_rows", 0)
    train_start = data_cfg["train_start"]
    train_end = data_cfg["train_end"]
    valid_end = data_cfg["valid_end"]
    output_dir = Path(out_cfg["output_dir"])

    logger.info("=== 第4.3节: LSTM 系泊力时间序列预测 ===")
    logger.info("数据: %s", data_cfg["data_path"])
    logger.info("特征: %s | 目标: %s | 回顾窗口: %d | 训练样本: %d | 波浪已知: %s",
                features, target, look_back, train_end - train_start, wave_known)
    if smoke_test:
        logger.info("*** 烟雾测试模式 ***")

    # ── 第1步：加载并归一化所有变量 ────────────────────────────────────
    df = load_csv_data(data_cfg["data_path"])
    data_array = np.hstack([
        np.array(df)[:, 1:8],   # 运动数据
        np.array(df)[:, 8:10],  # 力数据
    ])
    logger.info("数据已加载: %d 行", data_array.shape[0])

    # 构建完整的特征集合：输入特征... + 目标变量（目标必须在最后！）
    all_cols = features + [target]
    scaled_columns = {}
    scalers = {}
    for col_name in all_cols:
        idx = _COLUMN_MAP[col_name]
        s, sc = apply_minmax_scaler(data_array[:, idx:idx + 1])
        scaled_columns[col_name] = s
        scalers[col_name] = sc

    # 按顺序拼接：特征在前，目标在最后
    combined = np.hstack([scaled_columns[name] for name in all_cols])

    # ── 第2步：创建监督学习序列 ───────────────────────────────────────
    combined = combined[skip_rows:, :]
    n_features = len(all_cols)  # deal_data2 中的变量数量

    processed = deal_data2(combined, features_number=n_features, time_steps=look_back)
    x, y = split_sequence(processed, n_past=look_back)
    logger.info("序列数据: X=%s y=%s", x.shape, y.shape)

    # config 中的索引值直接对应 x/y 数组的行号，与原始代码中的用法完全一致
    train_x, train_y, valid_x, valid_y = split_train_valid(
        x, y,
        train_start=train_start,
        train_end=train_end,
        valid_end=valid_end,
    )
    logger.info("训练集: X=%s y=%s | 验证集: X=%s y=%s",
                train_x.shape, train_y.shape, valid_x.shape, valid_y.shape)

    # ── 第3步：构建并训练 LSTM ───────────────────────────────────────────
    from ..models.lstm import build_lstm_model

    input_shape = (train_x.shape[1], train_x.shape[2])
    model = build_lstm_model(
        input_shape=input_shape,
        num_layers=model_cfg.get("num_layers", 1),    # 单层 LSTM
        units=model_cfg.get("units", [25]),
        dropout=model_cfg.get("dropout", 0.0),
        activation=model_cfg.get("activation", "tanh"),
        learning_rate=train_cfg["learning_rate"],
    )

    history = model.fit(
        train_x, train_y,
        epochs=train_cfg["epochs"],
        batch_size=train_cfg["batch_size"],
        validation_data=(valid_x, valid_y),
        verbose=2,
        shuffle=train_cfg.get("shuffle", False),
    )
    logger.info("LSTM 训练完成 —— 最终损失: %.6f  验证损失: %.6f",
                history.history["loss"][-1],
                (history.history.get("val_loss") or [float("nan")])[-1])

    # ── 第4步：预测 ────────────────────────────────────────────────────
    pre_y = model.predict(valid_x, verbose=0)

    # 归一化空间下的 RMSE（用于跨实验对比）
    rmse_normalized = compute_rmse(valid_y, pre_y)

    # ── 第5步：反归一化并评估 ──────────────────────────────────────────
    t_scaler = scalers[target]
    fan_real = inverse_scale(valid_y, t_scaler)
    fan_pred = inverse_scale(pre_y, t_scaler)

    metrics = evaluate_predictions(fan_real, fan_pred, use_sklearn=True)
    print_metrics(metrics, title=f"{target}_lstm")

    # ── 第6步：保存结果 ─────────────────────────────────────────────────
    suffix = "_smoke" if smoke_test else ""
    wave_tag = "_wave_known" if wave_known else ""
    train_n = train_end - train_start
    feature_tag = "".join(f[:1] for f in features)

    if out_cfg.get("save_predictions", True):
        ensure_dir(output_dir)
        csv_path = output_dir / f"{target.lower()}_h{look_back}_lstm_t{train_n}{wave_tag}{suffix}.csv"
        save_predictions_csv(
            csv_path, fan_real, fan_pred,
            experiment_name="s4_lstm_mooring",
            target=target,
            model_name="lstm",
        )
        logger.info("预测结果已保存 (%d 行) → %s", len(fan_real), csv_path)

    # 保存统一格式的指标 CSV
    metrics_csv_path = output_dir / "lstm_wave_known_metrics.csv"
    _append_metrics_row(
        csv_path=metrics_csv_path,
        experiment_id=f"s4_lstm_mooring_{target.lower()}{wave_tag}{suffix}",
        target=target,
        wave_condition="known" if wave_known else "unknown",
        input_features=features,
        look_back=look_back,
        output_steps=data_cfg.get("output_steps", 1),
        rmse_normalized=rmse_normalized,
        rmse_physical=metrics["rmse"],
        rtrapz=metrics["rtrapz"],
        config_path=config_path,
        result_csv=str(csv_path) if out_cfg.get("save_predictions", True) else "",
        random_seed=train_cfg.get("random_seed", 42),
        notes="",
    )
    logger.info("指标已追加 → %s", metrics_csv_path)

    if out_cfg.get("save_figures", True):
        fig_dir = ensure_dir(output_dir / "figures")
        plot_prediction_curve(
            fan_real, fan_pred,
            title=f"{target} — LSTM (回顾窗口={look_back}, 训练样本={train_n}, 波浪={'已知' if wave_known else '未知'}){suffix}",
            n_points=min(200, len(fan_real)),
            save_path=fig_dir / f"{target.lower()}_lstm_h{look_back}_prediction{wave_tag}{suffix}.png",
        )
        plot_loss_curve(
            history.history["loss"],
            history.history.get("val_loss"),
            save_path=fig_dir / f"{target.lower()}_lstm_h{look_back}_loss{wave_tag}{suffix}.png",
        )

    logger.info("输出目录: %s", output_dir.resolve())
    logger.info("=== 第4.3节完成: RMSE=%.6f  Acc=%.6f ===", metrics["rmse"], metrics["acc"])


def _append_metrics_row(
    csv_path: Path,
    experiment_id: str,
    target: str,
    wave_condition: str,
    input_features: list,
    look_back: int,
    output_steps: int,
    rmse_normalized: float,
    rmse_physical: float,
    rtrapz: float,
    config_path: str,
    result_csv: str,
    random_seed: int,
    notes: str,
) -> None:
    """向指标汇总 CSV 追加一行记录。

    如果文件不存在则自动创建并写入表头；如果已存在则追加数据行。
    使用 append 模式确保多次实验运行不会互相覆盖。
    """
    fieldnames = [
        "experiment_id", "target", "side", "wave_condition",
        "input_features", "look_back", "output_steps",
        "rmse_normalized", "rmse_physical", "rtrapz",
        "config_path", "result_csv", "random_seed", "notes",
    ]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists()

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "experiment_id": experiment_id,
            "target": target,
            "side": target,             # Force1/Force2 即缆绳侧标识
            "wave_condition": wave_condition,
            "input_features": "+".join(input_features),
            "look_back": look_back,
            "output_steps": output_steps,
            "rmse_normalized": round(rmse_normalized, 8),
            "rmse_physical": round(rmse_physical, 4),
            "rtrapz": round(rtrapz, 6),
            "config_path": config_path,
            "result_csv": result_csv,
            "random_seed": random_seed,
            "notes": notes,
        })
