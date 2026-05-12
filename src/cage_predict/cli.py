"""命令行接口 —— 为 cage_predict 实验运行器提供命令行入口。

使用 argparse 构建命令和子命令，支持从终端直接运行各章实验。
每个子命令对应一个实验章节，通过 YAML 配置文件驱动实验参数。
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .config import load_config, validate_or_raise

logger = logging.getLogger(__name__)

# 所有实验的有序列表：(子命令名, 章节标签, 默认配置文件)
_EXPERIMENTS = [
    ("run-s3", "第3章: LSTM 运动响应预测", "configs/s3_motion.yaml"),
    ("run-s4-regression", "第4.2节: HGBRT/BPNN 回归", "configs/s4_regression.yaml"),
    ("run-s4-lstm", "第4.3节: LSTM 系泊力时序预测", "configs/s4_lstm_mooring.yaml"),
    ("run-s4-hybrid", "第4.4节: 混合联合预测", "configs/s4_hybrid.yaml"),
    ("run-s5-attention", "第5章: Attention 模型对比", "configs/s5_attention.yaml"),
]


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
    sub = parser.add_subparsers(dest="command", help="可用命令")

    # run-s3：第3章 —— LSTM 运动响应预测
    p_s3 = sub.add_parser("run-s3", help="第3章: LSTM 运动响应预测")
    p_s3.add_argument("--config", required=True, help="YAML 配置文件路径")
    p_s3.add_argument("--smoke-test", action="store_true", help="运行最小烟雾测试（快速验证）")
    p_s3.add_argument("--output-dir", default=None, help="覆盖输出目录（用于测试等场景）")

    # run-s4-regression：第4.2节 —— HGBRT/BPNN 回归预测
    p_s4r = sub.add_parser("run-s4-regression", help="第4.2节: HGBRT/BPNN 回归")
    p_s4r.add_argument("--config", required=True, help="YAML 配置文件路径")
    p_s4r.add_argument("--smoke-test", action="store_true")
    p_s4r.add_argument("--output-dir", default=None, help="覆盖输出目录（用于测试等场景）")

    # run-s4-lstm：第4.3节 —— LSTM 系泊力时间序列预测
    p_s4l = sub.add_parser("run-s4-lstm", help="第4.3节: LSTM 系泊力时间序列预测")
    p_s4l.add_argument("--config", required=True, help="YAML 配置文件路径")
    p_s4l.add_argument("--smoke-test", action="store_true")
    p_s4l.add_argument("--output-dir", default=None, help="覆盖输出目录（用于测试等场景）")

    # run-s4-hybrid：第4.4节 —— 混合联合预测
    p_s4h = sub.add_parser("run-s4-hybrid", help="第4.4节: 混合联合预测")
    p_s4h.add_argument("--config", required=True, help="YAML 配置文件路径")
    p_s4h.add_argument("--smoke-test", action="store_true")
    p_s4h.add_argument("--output-dir", default=None, help="覆盖输出目录（用于测试等场景）")

    # run-s5-attention：第5章 —— Attention 模型比较与泛化测试
    p_s5 = sub.add_parser("run-s5-attention", help="第5章: Attention 模型比较和泛化测试")
    p_s5.add_argument("--config", required=True, help="YAML 配置文件路径")
    p_s5.add_argument("--smoke-test", action="store_true")
    p_s5.add_argument("--output-dir", default=None, help="覆盖输出目录（用于测试等场景）")

    # run-all：一键顺序运行全部实验
    p_all = sub.add_parser("run-all", help="一键顺序运行第3、4、5章全部核心实验")
    p_all.add_argument("--smoke-test", action="store_true", help="烟雾测试模式（快速验证所有实验流程）")
    p_all.add_argument(
        "--config-dir",
        default="configs",
        help="配置文件目录（默认: configs/）",
    )
    p_all.add_argument("--output-dir", default=None, help="覆盖输出目录（用于测试等场景）")

    return parser


# ── 子命令到实验模块的映射 ────────────────────────────────────────────────
_COMMAND_RUNNER = {
    "run-s3":           "cage_predict.experiments.s3_motion",
    "run-s4-regression": "cage_predict.experiments.s4_regression",
    "run-s4-lstm":      "cage_predict.experiments.s4_lstm_mooring",
    "run-s4-hybrid":    "cage_predict.experiments.s4_hybrid",
    "run-s5-attention": "cage_predict.experiments.s5_attention",
}

# 子命令到 task_name 的映射（用于配置校验）
_COMMAND_TASK = {
    "run-s3": "s3_motion",
    "run-s4-regression": "s4_regression",
    "run-s4-lstm": "s4_lstm_mooring",
    "run-s4-hybrid": "s4_hybrid",
    "run-s5-attention": "s5_attention",
}


def _dispatch_experiment(
    command: str, config: str, smoke_test: bool, output_dir: str | None = None,
) -> None:
    """将单个实验子命令分发到对应的实验模块运行。

    在调用实验模块之前，先加载配置并进行校验，尽早发现配置问题。
    """
    import importlib

    # CLI 层预校验：在导入实验模块前检查配置文件合法性
    cfg = load_config(config, smoke_test=smoke_test, output_dir_override=output_dir)
    task_name = _COMMAND_TASK.get(command)
    validate_or_raise(cfg, task_name=task_name)
    logger.info("配置校验通过 [%s] → %s", task_name or "basic", config)

    mod_name = _COMMAND_RUNNER[command]
    mod = importlib.import_module(mod_name)
    mod.run(config, smoke_test=smoke_test, output_dir=output_dir)


def _run_all_experiments(
    config_dir: str, smoke_test: bool, output_dir: str | None = None,
) -> int:
    """顺序运行所有实验，返回失败个数。"""
    failed = 0
    total = len(_EXPERIMENTS)

    for i, (cmd, label, default_cfg) in enumerate(_EXPERIMENTS, 1):
        config_path = str(Path(config_dir) / Path(default_cfg).name)
        logger.info("")
        logger.info("=" * 60)
        logger.info("[%d/%d] %s", i, total, label)
        logger.info("=" * 60)

        try:
            _dispatch_experiment(cmd, config_path, smoke_test=smoke_test, output_dir=output_dir)
        except Exception:
            logger.exception("❌ 实验失败 [%s]: %s", cmd, label)
            failed += 1

    logger.info("")
    logger.info("=" * 60)
    logger.info("全部实验执行完毕: %d 成功, %d 失败 (共 %d)", total - failed, failed, total)
    logger.info("=" * 60)
    return failed


def main(argv: list[str] | None = None) -> None:
    """命令行入口函数。

    解析命令行参数，根据子命令分发到对应的实验运行器。

    使用示例：
        python -m cage_predict run-s3 --config configs/s3_motion.yaml
        python -m cage_predict run-s3 --config configs/s3_motion.yaml --smoke-test
        python -m cage_predict run-all --smoke-test
        python -m cage_predict run-all

    参数
    ----------
    argv : 命令行参数列表，None 表示使用 sys.argv。
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    if args.command == "run-all":
        failed = _run_all_experiments(
            args.config_dir, smoke_test=args.smoke_test, output_dir=args.output_dir,
        )
        if failed > 0:
            sys.exit(1)
    else:
        _dispatch_experiment(
            args.command, args.config,
            smoke_test=args.smoke_test, output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()
