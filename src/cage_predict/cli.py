"""命令行接口 —— 为 cage_predict 实验运行器提供命令行入口。

使用 argparse 构建命令和子命令，支持从终端直接运行各章实验。
每个子命令对应一个实验章节，通过 YAML 配置文件驱动实验参数。
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器。

    返回
    -------
    argparse.ArgumentParser : 配置好的参数解析器。
    """
    parser = argparse.ArgumentParser(
        prog="cage_predict",
        description="基于深度学习的网箱动力响应预测 —— 从命令行运行实验。",
    )
    # 子命令：每个实验章节一个子命令
    sub = parser.add_subparsers(dest="command", help="可用命令")

    # run-s3：第3章 —— LSTM 运动响应预测
    p_s3 = sub.add_parser("run-s3", help="第3章: LSTM 运动响应预测")
    p_s3.add_argument("--config", required=True, help="YAML 配置文件路径")
    p_s3.add_argument("--smoke-test", action="store_true", help="运行最小烟雾测试（快速验证）")

    # run-s4-regression：第4.2节 —— HGBRT/BPNN 回归预测
    p_s4r = sub.add_parser("run-s4-regression", help="第4.2节: HGBRT/BPNN 回归")
    p_s4r.add_argument("--config", required=True, help="YAML 配置文件路径")
    p_s4r.add_argument("--smoke-test", action="store_true")

    # run-s4-lstm：第4.3节 —— LSTM 系泊力时间序列预测
    p_s4l = sub.add_parser("run-s4-lstm", help="第4.3节: LSTM 系泊力时间序列预测")
    p_s4l.add_argument("--config", required=True, help="YAML 配置文件路径")
    p_s4l.add_argument("--smoke-test", action="store_true")

    # run-s4-hybrid：第4.4节 —— 混合联合预测
    p_s4h = sub.add_parser("run-s4-hybrid", help="第4.4节: 混合联合预测")
    p_s4h.add_argument("--config", required=True, help="YAML 配置文件路径")
    p_s4h.add_argument("--smoke-test", action="store_true")

    # run-s5-attention：第5章 —— Attention 模型比较与泛化测试
    p_s5 = sub.add_parser("run-s5-attention", help="第5章: Attention 模型比较和泛化测试")
    p_s5.add_argument("--config", required=True, help="YAML 配置文件路径")
    p_s5.add_argument("--smoke-test", action="store_true")

    return parser


def main(argv: list[str] | None = None) -> None:
    """命令行入口函数。

    解析命令行参数，根据子命令分发到对应的实验运行器。

    使用示例：
        python -m cage_predict run-s3 --config configs/s3_motion.yaml
        python -m cage_predict run-s3 --config configs/s3_motion.yaml --smoke-test

    参数
    ----------
    argv : 命令行参数列表，None 表示使用 sys.argv。
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()     # 没有提供任何命令时，打印帮助信息
        return

    # 配置日志格式：时间戳 + 级别 + 消息内容
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    # 根据子命令分发到对应的实验模块
    if args.command == "run-s3":
        from .experiments.s3_motion import run
        run(args.config, smoke_test=args.smoke_test)
    elif args.command == "run-s4-regression":
        from .experiments.s4_regression import run
        run(args.config, smoke_test=args.smoke_test)
    elif args.command == "run-s4-lstm":
        from .experiments.s4_lstm_mooring import run
        run(args.config, smoke_test=args.smoke_test)
    elif args.command == "run-s4-hybrid":
        from .experiments.s4_hybrid import run
        run(args.config, smoke_test=args.smoke_test)
    elif args.command == "run-s5-attention":
        from .experiments.s5_attention import run
        run(args.config, smoke_test=args.smoke_test)


if __name__ == "__main__":
    main()
