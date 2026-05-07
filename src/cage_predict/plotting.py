"""绘图工具 —— 预测曲线对比图、损失曲线图等。

所有函数默认将图像保存到磁盘而非调用 plt.show()，
以便实验可以在非交互终端模式下运行（如服务器、CI/CD 环境）。

使用 matplotlib 的 Agg 后端 —— 不依赖图形界面，纯后台渲染。
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# 设置非交互式后端 —— 必须在导入 pyplot 之后立即设置
# Agg 后端使用抗锯齿图形渲染器，在没有显示器的服务器上也能工作
matplotlib.use("Agg")

logger = logging.getLogger(__name__)


def _setup_chinese_font() -> None:
    """配置 matplotlib 的中文字体支持。

    按优先级尝试多个中文字体，以确保在不同操作系统上都能正确显示中文。
    """
    plt.rcParams["font.sans-serif"] = ["SimHei", "Songti SC", "Arial Unicode MS", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False  # 防止负号显示为方块


def plot_prediction_curve(
    real: np.ndarray,
    predicted: np.ndarray,
    title: str = "Real vs Prediction",
    xlabel: str = "Time step",
    ylabel: str = "Value",
    n_points: int = 200,
    save_path: str | Path | None = None,
    show: bool = False,
) -> None:
    """绘制真实值与预测值的对比曲线。

    蓝色曲线 = 真实值，橙色曲线 = 预测值。
    通常只绘制前 n_points 个数据点，以便清晰观察局部细节。

    参数
    ----------
    real : 真实值数组。
    predicted : 预测值数组。
    title : 图表标题。
    xlabel : x 轴标签。
    ylabel : y 轴标签。
    n_points : 绘制的数据点数量（默认 200，与原始代码一致）。
    save_path : 如果提供，则将图像保存到此路径。
    show : 如果为 True，调用 plt.show() 显示图像（在交互模式下会阻塞）。
    """
    _setup_chinese_font()
    plt.figure(figsize=(30, 4), dpi=100)   # 宽图，适合观察时间序列趋势
    n = min(n_points, len(real))
    plt.plot(real[:n], color="blue", label="真实值")
    plt.plot(predicted[:n], color="orange", label="预测值")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()                            # 显示图例
    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path, bbox_inches="tight")  # bbox_inches 去除多余白边
        logger.info(f"图像已保存到 {path}")
    if show:
        plt.show()
    plt.close()                             # 关闭图形释放内存


def plot_loss_curve(
    history_loss: list[float],
    history_val_loss: list[float] | None = None,
    save_path: str | Path | None = None,
    show: bool = False,
) -> None:
    """绘制训练和验证损失曲线。

    损失曲线是判断模型是否过拟合或欠拟合的重要工具：
      - 训练损失持续下降、验证损失上升 → 过拟合（需要正则化或早停）
      - 训练和验证损失都下降 → 模型仍在学习
      - 两条曲线趋于平稳 → 模型已收敛

    参数
    ----------
    history_loss : 每个 epoch 的训练损失值列表。
    history_val_loss : 每个 epoch 的验证损失值列表（可选）。
    save_path : 如果提供，则将图像保存到此路径。
    show : 是否显示图像。
    """
    _setup_chinese_font()
    plt.figure(figsize=(10, 6))
    plt.plot(history_loss, label="训练损失")
    if history_val_loss:
        plt.plot(history_val_loss, label="验证损失")
    plt.title("模型损失曲线")
    plt.ylabel("损失值 (MSE)")
    plt.xlabel("训练轮次 (Epoch)")
    plt.legend()
    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path, bbox_inches="tight")
        logger.info(f"图像已保存到 {path}")
    if show:
        plt.show()
    plt.close()
