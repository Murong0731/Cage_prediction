"""YAML 配置文件加载器，支持烟雾测试（smoke test）参数覆盖。

烟雾测试是一种快速验证流程是否可运行的轻量测试，通过 smoke_test 选项
用较小的参数（如更少的epoch、更小的搜索空间）替换原始配置中的对应值，
从而在几秒或几分钟内完成验证。
"""

from __future__ import annotations

from pathlib import Path

import yaml


def load_config(config_path: str | Path, smoke_test: bool = False) -> dict:
    """加载 YAML 配置文件并返回字典。

    参数
    ----------
    config_path : YAML 配置文件的路径。
    smoke_test : 如果为 True，则将配置中的 'smoke_test' 节浅合并到顶层键中，
                 这样实验运行器会看到减少后的参数（如更少的训练轮数、更小的搜索空间）。

    返回
    -------
    dict : 解析后的配置字典。
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"配置文件未找到: {path}")

    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)  # 使用 safe_load 防止 YAML 注入攻击

    # 烟雾测试：如果启用了 smoke_test 且配置中有 smoke_test 节，
    # 则将其覆盖到主配置的对应节中
    if smoke_test and "smoke_test" in cfg:
        smoke = cfg.pop("smoke_test")
        _merge_smoke(cfg, smoke)

    return cfg


def _merge_smoke(cfg: dict, smoke: dict) -> None:
    """将烟雾测试的覆盖参数浅合并到主配置的各个节中。

    浅合并意味着只替换第一层的键值对，不会进行深层次的嵌套合并。
    例如：smoke 中的 {'training': {'epochs': 2}} 会完全替换 cfg['training'] 中的 epochs 键。
    """
    for section, overrides in smoke.items():
        if section in cfg and isinstance(cfg[section], dict) and isinstance(overrides, dict):
            cfg[section].update(overrides)
