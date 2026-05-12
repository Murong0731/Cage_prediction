"""数据加载、预处理和序列转换工具。

本模块包含时间序列预测中最核心的数据处理操作：
  1. CSV 数据的读取和保存
  2. MinMaxScaler 归一化/反归一化（将数据映射到 [-1, 1] 区间）
  3. 时间序列 → 监督学习格式的转换（series_to_supervised）
  4. 特定列提取/删除的 deal_data1 / deal_data2
  5. 滑动窗口序列的切分（split_sequence）
  6. 训练集/验证集的按时间索引切分

从原始代码 s3/3.3.py, s3/3.4.py, s4/4.3/4.3.py, s4/4.4/4.4.py,
s5/第五章（上）.py, s5/第五章（下）.py 中提取，保留了精确的原始逻辑。
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# 一、文件输入输出 (I/O)
# ═══════════════════════════════════════════════════════════════════════════════

def load_csv_data(data_path: str | Path) -> pd.DataFrame:
    """加载 CSV 数据文件并返回 DataFrame。

    如果文件路径不存在，则抛出 FileNotFoundError。

    参数
    ----------
    data_path : CSV 文件的路径。

    返回
    -------
    pd.DataFrame : 读取的数据表格。
    """
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"数据文件未找到: {path}")
    return pd.read_csv(path)


def save_results_csv(path: str | Path, real: np.ndarray, predicted: np.ndarray) -> None:
    """将真实值与预测值保存到 CSV 文件（两列带表头格式）。

    原始代码中每个章节都使用此模式：
        np.savetxt('xxx.csv', np.hstack((real, pre)), delimiter=',')

    第一列为真实值，第二列为预测值，便于后续分析和绘图。
    现改用 pandas 保存带表头的 CSV，方便论文复现和结果追踪。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({
        "y_true": real.ravel(),
        "y_pred": predicted.ravel(),
    })
    df.to_csv(path, index=False)
    logger.info("结果已保存到 %s", path)


def save_predictions_csv(
    path: str | Path,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    experiment_name: str | None = None,
    target: str | None = None,
    model_name: str | None = None,
    experiment_id: str | None = None,
) -> pd.DataFrame:
    """将预测结果保存为带完整表头的 CSV 文件，适用于论文复现和结果追踪。

    保存的列包括：
      - sample_index : 样本序号（0-based）
      - y_true       : 真实值（反归一化后的原始量纲）
      - y_pred       : 预测值
      - error        : y_pred - y_true（有符号误差）
      - abs_error    : |y_pred - y_true|（绝对误差）
      - experiment_name / target / model_name / experiment_id（可选元数据）

    使用 pandas 保存，确保文件可以被任何数据分析工具直接读取。

    参数
    ----------
    path : CSV 文件的保存路径。
    y_true : 真实值数组。
    y_pred : 预测值数组。
    experiment_name : 实验名称（如 "s3_motion"、"s5_attention"）。
    target : 预测目标变量名（如 "Heave"、"Force1"）。
    model_name : 模型名称（如 "lstm"、"attention"）。
    experiment_id : 实验标识符（如配置哈希或时间戳），可选。

    返回
    -------
    pd.DataFrame : 保存的 DataFrame，方便后续进一步处理。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    yt = y_true.ravel()
    yp = y_pred.ravel()
    err = yp - yt
    abs_err = np.abs(err)

    df = pd.DataFrame({
        "sample_index": range(len(yt)),
        "y_true": yt,
        "y_pred": yp,
        "error": err,
        "abs_error": abs_err,
    })

    if experiment_name is not None:
        df["experiment_name"] = experiment_name
    if target is not None:
        df["target"] = target
    if model_name is not None:
        df["model_name"] = model_name
    if experiment_id is not None:
        df["experiment_id"] = experiment_id

    df.to_csv(path, index=False)
    logger.info("预测结果已保存到 %s (%d 行, %d 列)", path, len(df), df.shape[1])
    return df


def load_predictions_csv(path: str | Path) -> pd.DataFrame:
    """从 CSV 文件加载预测结果。

    兼容新旧两种格式：
      - 新格式：带 sample_index / y_true / y_pred 等表头
      - 旧格式：两列无表头（np.savetxt 风格，自动赋予 y_true / y_pred 列名）

    参数
    ----------
    path : CSV 文件的路径。

    返回
    -------
    pd.DataFrame : 加载的预测结果。
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"预测结果文件未找到: {path}")

    df = pd.read_csv(path)
    first_col = str(df.columns[0])

    # 兼容旧的无表头两列格式：如果第一列名不是已知字段，且看起来像数值，
    # 则说明原始文件没有表头行，需要用 header=None 重新读取
    if first_col not in {"sample_index", "y_true"}:
        try:
            float(first_col)
            # 第一列名为数值 → 旧格式没有表头，重新读取
            df = pd.read_csv(path, header=None)
        except ValueError:
            pass
        if df.shape[1] == 2:
            df.columns = ["y_true", "y_pred"]

    return df


def select_features(df: pd.DataFrame, columns: list[str]) -> np.ndarray:
    """从 DataFrame 中提取指定列作为 NumPy 数组。

    如果某列在 DataFrame 中不存在，则抛出 KeyError。

    参数
    ----------
    df : 输入数据表格。
    columns : 要提取的列名列表。

    返回
    -------
    np.ndarray : 提取后的二维数组。
    """
    missing = set(columns) - set(df.columns)
    if missing:
        raise KeyError(f"以下列未在数据中找到: {missing}")
    return df[columns].values


# ═══════════════════════════════════════════════════════════════════════════════
# 二、数据归一化/反归一化 —— 使用 sklearn MinMaxScaler
# ═══════════════════════════════════════════════════════════════════════════════

def apply_minmax_scaler(
    data: np.ndarray, feature_range: tuple[float, float] = (-1.0, 1.0)
) -> tuple[np.ndarray, "MinMaxScaler"]:
    """对数据拟合 MinMaxScaler 并返回 (归一化后数据, 归一化器)。

    归一化的作用：将不同量纲的特征缩放到统一范围（默认 [-1, 1]），
    使神经网络训练更稳定、收敛更快。

    原始代码模式（各章节通用）：
        H_scaler = MinMaxScaler(feature_range=(-1, 1))
        H = H_scaler.fit_transform(data[:, 0:1])

    参数
    ----------
    data : 形状为 (n_samples,) 或 (n_samples, 1) 的原始数据。
    feature_range : 归一化后的目标范围，默认 (-1.0, 1.0)。

    返回
    -------
    tuple : (归一化后的数据, 拟合好的 MinMaxScaler 对象)
    """
    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler(feature_range=feature_range)
    scaled = scaler.fit_transform(data)
    return scaled, scaler


def apply_minmax_scalers(
    data_columns: list[np.ndarray], feature_range: tuple[float, float] = (-1.0, 1.0)
) -> tuple[list[np.ndarray], list["MinMaxScaler"]]:
    """对多个列数组分别独立应用 MinMaxScaler。

    这是 apply_minmax_scaler 的批量版本。当需要对 H（波高）、Surge（纵荡）、
    Heave（垂荡）等多个变量分别进行归一化时，每个变量拥有独立的归一化器，
    以便后续各自反归一化。

    为什么要分别归一化？
      - 不同物理量（波高、位移、力）的量纲和幅值差异巨大
      - 分别归一化保留了各变量的相对变化模式
      - 反归一化时需要用各自的 scaler 恢复原始量纲

    参数
    ----------
    data_columns : 多个待归一化的列数组列表。
    feature_range : 归一化目标范围。

    返回
    -------
    tuple : (归一化后的数组列表, 归一化器列表)
    """
    from sklearn.preprocessing import MinMaxScaler

    scaled_list, scaler_list = [], []
    for col_data in data_columns:
        scaler = MinMaxScaler(feature_range=feature_range)
        scaled_list.append(scaler.fit_transform(col_data))
        scaler_list.append(scaler)
    return scaled_list, scaler_list


def inverse_scale(data: np.ndarray, scaler: "MinMaxScaler") -> np.ndarray:
    """使用训练好的 MinMaxScaler 对数据进行反归一化。

    反归一化将 [-1, 1] 范围内的预测值恢复到原始物理量纲。
    所有原始代码的 FanGuiHua_* 函数中都使用此操作：
        scaler.inverse_transform(valid_y)

    参数
    ----------
    data : 归一化后的数据（1D 或 2D）。
    scaler : 之前拟合好的 MinMaxScaler 对象。

    返回
    -------
    np.ndarray : 恢复到原始量纲的数据。
    """
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    return scaler.inverse_transform(data)


# ═══════════════════════════════════════════════════════════════════════════════
# 三、时间序列 → 监督学习格式转换
# ═══════════════════════════════════════════════════════════════════════════════

def series_to_supervised(
    data: np.ndarray | list, n_in: int, n_out: int = 1, dropnan: bool = True
) -> np.ndarray:
    """将时间序列转换为监督学习格式（带滞后特征）。

    这是时间序列预测中最核心的数据预处理步骤。它将连续的时序数据
    重新组织为"过去→未来"的输入-输出对。

    工作原理（以 n_in=3, n_out=1 为例）：
        原始序列: [t1, t2, t3, t4, t5, t6, t7, t8]
        转换后:
            输入(t-3, t-2, t-1)  →  输出(t)
            [t1, t2, t3]         →  t4
            [t2, t3, t4]         →  t5
            [t3, t4, t5]         →  t6
            ...

    shift(n) 的含义：
      - shift(+n)：将数据向上移动 n 行（即取未来的值）
      - shift(-n)：将数据向下移动 n 行（即取过去的值）

    精确复现原始 s3/3.3.py:437 中的 series_to_supervised() 函数。

    参数
    ----------
    data : 形状为 (n_samples, n_vars) 的数组，或单变量的列表。
    n_in : 回顾窗口大小（使用过去多少个时间步作为输入）。
    n_out : 预测未来多少个时间步（默认为 1，即单步预测）。
    dropnan : 是否删除因 shift 操作产生的含 NaN 行。

    返回
    -------
    reframed : 形状为 (n_samples - n_in - n_out + 1, (n_in + n_out) * n_vars) 的数组。
               列名格式: var{j}(t-{k}) 表示第j个变量的过去第k步的值，
                       var{j}(t+{k}) 表示第j个变量的未来第k步的值。
    """
    n_vars = 1 if isinstance(data, list) else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = [], []

    # 构建输入特征（历史滞后值）
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))       # shift(n): 向过去取n步
        names += [f"var{j+1}(t-{i})" for j in range(n_vars)]

    # 构建输出标签（当前值和未来值）
    for i in range(0, n_out):
        cols.append(df.shift(-i))      # shift(-n): 向未来取n步
        if i == 0:
            names += [f"var{j+1}(t)" for j in range(n_vars)]       # 当前时刻
        else:
            names += [f"var{j+1}(t+{i})" for j in range(n_vars)]   # 未来时刻

    agg = pd.concat(cols, axis=1)
    agg.columns = names
    if dropnan:
        agg.dropna(inplace=True)       # 删除因位移操作产生的 NaN 行
    return agg.values


def deal_data1(data: np.ndarray, features_number: int, time_steps: int) -> np.ndarray:
    """处理数据用于单/多特征预测 —— 保留全部输入特征滞后列。

    适用场景：单特征或多特征，单步或多步预测。

    列选择逻辑（原始 s3/3.3.py:458 中的精确实现）：
        删除介于 features_number 和 features_number*(time_steps+1)-1 之间的列，
        只保留最前面的 features_number 个滞后列（输入特征）和最后一列（输出标签）。

    直观理解：
        假设有2个特征 [f1, f2]，time_steps=3：
        series_to_supervised 产生 8 列：
          [f1(t-3), f2(t-3), f1(t-2), f2(t-2), f1(t-1), f2(t-1), f1(t), f2(t)]
        删除索引 [2,3,4,5]（中间列），保留 [0,1,6,7]：
          [f1(t-3), f2(t-3), f1(t), f2(t)]
        即保留所有特征的过去第一步 + 所有特征的当前值

    参数
    ----------
    data : 原始多变量数据。
    features_number : 特征变量数量。
    time_steps : 回顾窗口大小。

    返回
    -------
    np.ndarray : 处理后的数据。
    """
    process_data = series_to_supervised(data, time_steps, 1, dropnan=True)
    df = pd.DataFrame(process_data)
    # 只保留输入特征（前 features_number 个滞后列）和输出（最后一个列）
    drop_cols = list(range(features_number, features_number * (time_steps + 1) - 1))
    df.drop(df.columns[drop_cols], axis=1, inplace=True)
    return df.values


def deal_data2(data: np.ndarray, features_number: int, time_steps: int) -> np.ndarray:
    """处理数据用于多特征预测 —— 只保留第一个输入特征的滞后列。

    适用场景：多特征，单步或多步预测。不适用于单特征。

    与 deal_data1 的关键区别：
      - deal_data1 保留所有输入特征的历史滞后值
      - deal_data2 只保留第一个输入特征的历史滞后值（更激进的降维）

    列选择逻辑（原始 s3/3.3.py:473 中的精确实现）：
        删除介于 features_number-1 和 features_number*(time_steps+1)-1 之间的列，
        只保留第一个特征的历史值 + 当前输出值。

    参数
    ----------
    data : 原始多变量数据。
    features_number : 特征变量数量。
    time_steps : 回顾窗口大小。

    返回
    -------
    np.ndarray : 处理后的数据。
    """
    process_data = series_to_supervised(data, time_steps, 1, dropnan=True)
    df = pd.DataFrame(process_data)
    # 只保留第一个输入特征的滞后值和输出列
    drop_cols = list(range(features_number - 1, features_number * (time_steps + 1) - 1))
    df.drop(df.columns[drop_cols], axis=1, inplace=True)
    return df.values


def split_sequence(dataset: np.ndarray, n_past: int) -> tuple[np.ndarray, np.ndarray]:
    """将二维数据转换为 (X, y) 滑动窗口序列。

    这是时间序列预测中构建训练样本的标准方法。

    工作原理（n_past=5 为例）：
        数据: [d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, ...]
        生成序列:
          X[0] = [d0:d5, :-1]  → y[0] = d0[-1]   （前5行除最后一列作输入）
          X[1] = [d1:d6, :-1]  → y[1] = d1[-1]   （滑动1步）
          X[2] = [d2:d7, :-1]  → y[2] = d2[-1]
          ...

    原始代码（s3/3.3.py:488）的精确复现：
        X[i] = dataset[i : i+n_past, :-1]   —— 除最后一列外的所有列为输入特征
        y[i] = dataset[i, -1]               —— 最后一列为预测目标

    参数
    ----------
    dataset : 形状为 (n_samples, n_features) 的二维数据，
              其中最后一列为预测目标，其余为输入特征。
    n_past : 滑动窗口大小（每个样本包含多少个连续时间步）。

    返回
    -------
    tuple : (X, y)，其中
            X.shape = (n_samples - n_past, n_past, n_features - 1) —— 3D 张量
            y.shape = (n_samples - n_past,) —— 1D 数组
    """
    x, y = [], []
    for i in range(len(dataset)):
        end_ix = i + n_past
        if end_ix > len(dataset):
            break                       # 超出数据范围则停止
        seq_x = dataset[i:end_ix, :-1]  # 取连续 n_past 行，除最后一列作特征
        seq_y = dataset[i, -1]          # 取当前行的最后一列作标签
        x.append(seq_x)
        y.append(seq_y)
    return np.array(x), np.array(y)


# ═══════════════════════════════════════════════════════════════════════════════
# 四、训练集/验证集按时间索引切分
# ═══════════════════════════════════════════════════════════════════════════════

def split_train_valid(
    data_X: np.ndarray,
    data_Y: np.ndarray,
    train_start: int,
    train_end: int,
    valid_end: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """按时间索引将序列数据切分为训练集和验证集。

    在时间序列预测中，不能用随机切分（会泄露未来信息），
    必须按时间顺序切分：训练集在前，验证集在后。

    统一的 5 参数版本（原始 s3/3.4.py:340, s4/4.3/4.3.py:290 的逻辑）。

    切分示意：
        数据索引: [0 ──── train_start ──── train_end ──── valid_end ────]
                      ← 不使用的数据 →
                                     ← 训练集 →
                                                     ← 验证集 →

    参数
    ----------
    data_X : 输入特征序列，形状为 (n_samples, n_past, n_features)。
    data_Y : 目标值序列，形状为 (n_samples,) 或 (n_samples, 1)。
    train_start : 训练集的起始索引（0 表示与原始三参数版本兼容）。
    train_end : 训练集的结束索引（也是验证集的起始索引）。
    valid_end : 验证集的结束索引。

    返回
    -------
    train_x, train_y, valid_x, valid_y : 训练和验证数据。
        train_x.shape = (train_end - train_start, n_past, n_features)
        train_y.shape = (train_end - train_start, 1)
        valid_x.shape = (valid_end - train_end, n_past, n_features)
        valid_y.shape = (valid_end - train_end, 1)
    """
    train_x = data_X[train_start:train_end, :]
    valid_x = data_X[train_end:valid_end, :]
    train_y = data_Y[train_start:train_end]
    valid_y = data_Y[train_end:valid_end]
    train_y = train_y.reshape(-1, 1)   # 确保是列向量
    valid_y = valid_y.reshape(-1, 1)
    return train_x, train_y, valid_x, valid_y
