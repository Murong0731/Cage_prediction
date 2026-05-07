"""通用工具函数：随机种子设置、日志配置、目录创建。

从原始代码 s3/、s4/、s5/ 各章节中提取的通用模式。
设置随机种子是深度学习实验可重复性的关键步骤。
"""

from __future__ import annotations

import logging
import os
import random
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    """设置 Python、NumPy 和 TensorFlow 的随机种子。

    在模型构建或数据打乱之前调用此函数，可确保实验结果的可重复性。
    相同种子 + 相同代码 + 相同数据 = 相同结果。

    参数
    ----------
    seed : 随机种子值，默认为 42。
    """
    random.seed(seed)                   # Python 内置 random 模块
    np.random.seed(seed)                # NumPy 随机数生成器
    os.environ["PYTHONHASHSEED"] = str(seed)  # Python 哈希种子（影响字典迭代顺序）
    try:
        import tensorflow as tf

        tf.random.set_seed(seed)        # TensorFlow 随机种子
        logger.debug("TensorFlow 随机种子已设置为 %d", seed)
    except ImportError:
        pass
    logger.debug("随机种子已设置为 %d", seed)


def setup_logging(level: int = logging.INFO) -> None:
    """配置根日志记录器，使用标准格式输出。

    日志格式示例：
        2025-05-04 14:30:00 [INFO] cage_predict.data: 数据已加载

    参数
    ----------
    level : 日志级别，默认为 logging.INFO。
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def ensure_dir(path: str | Path) -> Path:
    """创建目录（包括所有父目录），如果已存在则不报错。

    类似 mkdir -p 命令的行为，常用于确保输出目录在写入文件前已就绪。

    参数
    ----------
    path : 要创建的目录路径。

    返回
    -------
    Path : 创建后的路径对象。
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
