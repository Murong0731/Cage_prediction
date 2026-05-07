"""时间序列预测的评估指标。

精确复现原始论文中的指标计算方法：
  - MAE  （平均绝对误差，手动实现 + sklearn）
  - MAPE （平均绝对百分比误差，sklearn 实现）
  - MSE  （均方误差，手动实现 + sklearn）
  - RMSE （均方根误差，sklearn 实现）
  - Acc  （基于梯形积分的自定义准确率指标，等价于 Rtrapz）

指标说明：
  MAE  = (1/n) * Σ|y_true - y_pred|          —— 误差的绝对平均值
  MSE  = (1/n) * Σ(y_true - y_pred)²         —— 误差平方的平均值
  RMSE = √MSE                                  —— MSE 的平方根，与原始数据同量纲
  MAPE = (100/n) * Σ|(y_true - y_pred)/y_true| —— 百分比误差
  Acc  = 1 - |1 - 积分面积比|                  —— 自定义积分准确率

原始代码来源：
  s3/3.3.py:529  evaluate()      — MAE, MAPE, MSE, RMSE, Acc
  s5/5.2  :167   RNSE()          — RMSE only
  s5/5.2  :180   Acc()           — Acc only
  s5/5.3  :226   evaluate()      — MAE, MAPE, MSE, RMSE, Acc (与 3.3 相同)
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """计算平均绝对误差 (MAE) —— 手动实现。

    MAE 衡量预测值与真实值之间的平均绝对偏差，对异常值不敏感。
    值越小表示预测越准确，最小值为 0（完美预测）。

    原始代码来源：s3/3.3.py:532 中的手动计算。
    """
    diff = np.abs(y_true.ravel() - y_pred.ravel())
    return float(diff.sum() / diff.shape[0])


def compute_mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """计算均方误差 (MSE) —— 手动实现。

    MSE 通过平方惩罚放大较大的误差，是神经网络训练中最常用的损失函数。
    与 MAE 相比，MSE 对大误差更敏感（因为误差被平方了）。

    原始代码来源：s3/3.3.py:538 中的手动计算。
    """
    diff = y_true.ravel() - y_pred.ravel()
    return float((diff * diff).sum() / diff.shape[0])


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """计算均方根误差 (RMSE) —— 通过 sklearn 实现。

    RMSE 是 MSE 的平方根，量纲与原始数据一致，是最常用的回归评估指标之一。
    例如：如果预测的是力（单位kN），RMSE的单位也是kN，便于直观理解。

    原始代码来源：s3/3.3.py:546 中的 sklearn 调用。
    """
    from sklearn.metrics import mean_squared_error

    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def compute_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """计算平均绝对百分比误差 (MAPE) —— 通过 sklearn 实现。

    MAPE 以百分比形式表示误差，便于跨不同尺度的数据进行比较。
    但 MAPE 在真实值接近 0 时会趋近无穷大，这是其局限性。

    原始代码来源：s3/3.3.py:538 中的 sklearn 调用。
    """
    from sklearn.metrics import mean_absolute_percentage_error

    return float(mean_absolute_percentage_error(y_true, y_pred))


def compute_rtrapz(
    y_true: np.ndarray, y_pred: np.ndarray, dx: float = 0.1
) -> float:
    """使用梯形积分法计算自定义准确率指标（Rtrapz / Acc）。

    这是论文中使用的自定义准确率指标，通过比较真实曲线和预测曲线
    偏离均值的积分面积来衡量预测质量。

    原始公式（s3/3.3.py:553-555, s5/5.2:180-185）：
        real_area = ∫ |y_true - mean(y_true)| dx    —— 真实值的波动面积
        pred_area = ∫ |y_pred - mean(y_true)| dx    —— 预测值的波动面积（注意：都用真实均值）
        Acc = 1 - |1 - pred_area / real_area|        —— 面积比的偏离度

    解读：
      - Acc = 1：完美预测（预测曲线的波动面积与真实曲线完全一致）
      - Acc < 1：预测与真实有偏差
      - Acc 可能为负：预测非常差（预测面积的偏离超过100%）

    参数
    ----------
    y_true : 真实值数组。
    y_pred : 预测值数组。
    dx : 积分步长，默认 0.1（与原始代码一致）。

    返回
    -------
    float : Acc 值，范围通常在 (-∞, 1]，1为最佳。
    """
    yt = y_true.ravel()
    yp = y_pred.ravel()
    yt_mean = yt.mean()

    # 利用 numpy.trapz 进行梯形数值积分
    # np.trapz(|y - mean|, dx) 近似计算曲线与均值围成的面积
    real_area = float(np.trapz(np.abs(yt - yt_mean), dx=dx))
    pred_area = float(np.trapz(np.abs(yp - yt_mean), dx=dx))

    if real_area == 0.0:
        return 0.0      # 真实值为常数时，面积比为无穷大，返回 0

    acc = 1.0 - abs(1.0 - pred_area / real_area)
    return acc


# 别名的别名 —— 原始代码中有时叫 Acc，有时叫 Rtrapz，两者等价
compute_acc = compute_rtrapz


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    dx: float = 0.1,
    use_sklearn: bool = True,
) -> dict:
    """计算所有评估指标，返回包含各项结果的字典。

    将原始代码中的 evaluate()（s3/3.3.py:529）和独立的 RNSE()/Acc()（s5/5.2）
    合并为一个统一函数，方便实验脚本调用。

    参数
    ----------
    y_true : 真实值（一维或二维数组均可）。
    y_pred : 预测值。
    dx : Rtrapz 指标的积分步长，默认 0.1。
    use_sklearn : 如果为 True，使用 sklearn 计算 MAE/MAPE/MSE/RMSE；
                  如果为 False，仅使用手动实现（适合没有 sklearn 的环境）。

    返回
    -------
    dict : 包含如下键的字典：
        - mae (float)    : 平均绝对误差
        - mape (float)   : 平均绝对百分比误差（仅 sklearn 模式下）
        - mse (float)    : 均方误差
        - rmse (float)   : 均方根误差
        - rtrapz (float) : 积分准确率
        - acc (float)    : 与 rtrapz 相同（兼容原始代码中的变量名）
    """
    yt = y_true.ravel()
    yp = y_pred.ravel()

    # 手动实现 —— 始终可用，不依赖 sklearn
    mae_manual = compute_mae(yt, yp)
    mse_manual = compute_mse(yt, yp)
    rmse_manual = float(np.sqrt(mse_manual))
    rtrapz = compute_rtrapz(yt, yp, dx=dx)

    results = {
        "mae": mae_manual,
        "mse": mse_manual,
        "rmse": rmse_manual,
        "rtrapz": rtrapz,
        "acc": rtrapz,  # 别名，匹配原始变量名
    }

    # sklearn 提供的指标（更标准、更可靠）
    if use_sklearn:
        try:
            from sklearn.metrics import (
                mean_absolute_error,
                mean_absolute_percentage_error,
                mean_squared_error,
            )

            results["mae"] = float(mean_absolute_error(yt, yp))
            results["mape"] = float(mean_absolute_percentage_error(yt, yp))
            results["mse"] = float(mean_squared_error(yt, yp))
            results["rmse"] = float(np.sqrt(mean_squared_error(yt, yp)))
        except ImportError:
            results["mape"] = float("nan")
            logger.debug("sklearn 不可用 —— 使用手动计算的指标")

    return results


def print_metrics(results: dict, title: str = "") -> None:
    """按照原始代码的输出格式打印评估指标。

    原始输出格式（各章节约定俗成）：
        MAE: 0.123  MAE(sklearn): 0.123
        MAPE(sklearn): 0.456
        MSE: 0.789  MSE(sklearn): 0.789
        RMSE(sklearn): 0.888
        Acc: 0.923

    参数
    ----------
    results : evaluate_predictions() 返回的结果字典。
    title : 可选的标题前缀，如 "Force1_lstm"。
    """
    prefix = f"[{title}] " if title else ""
    logger.info(
        "%sMAE=%.6f  MAPE=%.6f  MSE=%.6f  RMSE=%.6f  Acc=%.6f",
        prefix,
        results.get("mae", float("nan")),
        results.get("mape", float("nan")),
        results.get("mse", float("nan")),
        results.get("rmse", float("nan")),
        results.get("acc", float("nan")),
    )
