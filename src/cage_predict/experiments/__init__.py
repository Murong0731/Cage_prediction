"""实验运行器入口 —— 各章节实验的运行函数。

每个函数对应论文中的一个实验章节，可以通过 CLI 命令或直接调用运行。
"""

from .s3_motion import run as run_s3
from .s4_regression import run as run_s4_regression
from .s4_lstm_mooring import run as run_s4_lstm
from .s4_hybrid import run as run_s4_hybrid
from .s5_attention import run as run_s5_attention

__all__ = [
    "run_s3",
    "run_s4_regression",
    "run_s4_lstm",
    "run_s4_hybrid",
    "run_s5_attention",
]
