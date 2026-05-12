"""YAML 配置文件加载器，支持烟雾测试（smoke test）参数覆盖。

烟雾测试是一种快速验证流程是否可运行的轻量测试，通过 smoke_test 选项
用较小的参数（如更少的epoch、更小的搜索空间）替换原始配置中的对应值，
从而在几秒或几分钟内完成验证。

配置校验：
  validate_config() 提供两层校验 —— 基础校验和任务特定校验。
  - 基础校验：所有配置文件都必须通过的检查（data_path 存在、必需的节存在、数值字段格式正确）
  - 任务特定校验：根据 task_name 检查各章节独有字段
  校验在 CLI 启动实验前和实验函数入口处各执行一次，确保尽早发现配置错误。
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml


def load_config(
    config_path: str | Path,
    smoke_test: bool = False,
    output_dir_override: str | None = None,
) -> dict:
    """加载 YAML 配置文件并返回字典。

    参数
    ----------
    config_path : YAML 配置文件的路径。
    smoke_test : 如果为 True，则将配置中的 'smoke_test' 节浅合并到顶层键中，
                 这样实验运行器会看到减少后的参数（如更少的训练轮数、更小的搜索空间）。
    output_dir_override : 如果提供，覆盖 output.output_dir 字段，
                          用于将输出重定向到临时目录（如测试中）。

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

    # 覆盖 output_dir（用于测试等场景重定向输出）
    if output_dir_override is not None:
        cfg.setdefault("output", {})["output_dir"] = output_dir_override

    return cfg


def _merge_smoke(cfg: dict, smoke: dict) -> None:
    """将烟雾测试的覆盖参数浅合并到主配置的各个节中。

    浅合并意味着只替换第一层的键值对，不会进行深层次的嵌套合并。
    例如：smoke 中的 {'training': {'epochs': 2}} 会完全替换 cfg['training'] 中的 epochs 键。
    """
    for section, overrides in smoke.items():
        if section in cfg and isinstance(cfg[section], dict) and isinstance(overrides, dict):
            cfg[section].update(overrides)


# ═══════════════════════════════════════════════════════════════════════════════
# 配置校验
# ═══════════════════════════════════════════════════════════════════════════════

def _is_positive_int(value: object) -> bool:
    """检查 value 是否为严格正整数（不接受 0 或负数）。"""
    return isinstance(value, int) and value > 0


def _is_positive_number(value: object) -> bool:
    """检查 value 是否为严格正数（int 或 float，不接受 0 或负数）。"""
    return isinstance(value, (int, float)) and value > 0


def _dig(cfg: dict, *keys: str) -> object:
    """安全地从嵌套字典中取值，任一中间键缺失时返回 None。"""
    node: object = cfg
    for k in keys:
        if isinstance(node, dict):
            node = node.get(k)
        else:
            return None
    return node


def validate_config(config: dict, task_name: str | None = None) -> list[str]:
    """校验配置字典的合法性，返回错误消息列表。

    基础校验对所有配置生效；task_name 不为 None 时附加各章节特定校验。
    返回空列表表示配置通过所有检查。

    参数
    ----------
    config : 已加载的配置字典（load_config 的返回值）。
    task_name : 可选的任务标识符，触发章节特定校验。
                目前支持: s3_motion, s4_regression, s4_lstm_mooring,
                         s4_hybrid, s5_attention。

    返回
    -------
    list[str] : 错误消息列表，每一项描述一个具体校验失败项。
    """
    errors: list[str] = []

    # ── 基础校验 ──────────────────────────────────────────────────────────
    # 1. data 节必须存在
    if "data" not in config or not isinstance(config["data"], dict):
        errors.append("缺少顶层节 'data'（或格式非字典）")
        return errors  # 后续校验依赖 data 节，直接返回
    data = config["data"]

    # 2. data_path 必须存在
    dp = data.get("data_path")
    if not isinstance(dp, str) or not dp.strip():
        errors.append("data.data_path 缺失或为空字符串")
    elif not Path(dp).exists():
        errors.append(f"data.data_path 指向的文件不存在: {dp}")

    # 3. output_dir 必须可创建（父目录存在）
    out = config.get("output")
    if isinstance(out, dict):
        od = out.get("output_dir")
        if isinstance(od, str) and od.strip():
            out_path = Path(od)
            try:
                out_path.mkdir(parents=True, exist_ok=True)
            except (OSError, PermissionError) as e:
                errors.append(f"output.output_dir 无法创建: {od} ({e})")
        elif not isinstance(od, str) or not od.strip():
            errors.append("output.output_dir 缺失或为空字符串")

    # 4. 模型配置节必须存在（model 或 stage1 + stage2）
    has_model = "model" in config and isinstance(config.get("model"), dict)
    has_stage1 = "stage1" in config and isinstance(config.get("stage1"), dict)
    has_stage2 = "stage2" in config and isinstance(config.get("stage2"), dict)
    if not has_model and not has_stage1:
        errors.append("缺少模型配置节: 需要顶层 'model' 或 'stage1' 节")

    # 5. 训练配置节必须存在（training 或 stage1 + stage2）
    has_training = "training" in config and isinstance(config.get("training"), dict)
    if not has_training and not has_stage1:
        errors.append("缺少训练配置节: 需要顶层 'training' 或 'stage1' 节")

    # 6. random_seed 或 seed（若存在须为有效非负整数，不要求必须存在）
    for section_name in ("training", "stage1", "model"):
        sec = config.get(section_name)
        if isinstance(sec, dict):
            for key in ("random_seed", "seed"):
                val = sec.get(key)
                if val is not None:
                    if not isinstance(val, int) or val < 0:
                        errors.append(
                            f"{section_name}.{key} 须为非负整数，当前值: {val} ({type(val).__name__})"
                        )

    # 7. target / targets 非空
    def _check_target(d: dict, path: str) -> None:
        t = d.get("target")
        ts = d.get("targets")
        mt = d.get("mooring_target")
        mts = d.get("motion_targets")
        has_target = isinstance(t, str) and t.strip()
        has_targets = isinstance(ts, list) and len(ts) > 0
        has_mt = isinstance(mt, str) and mt.strip()
        has_mts = isinstance(mts, list) and len(mts) > 0
        if not (has_target or has_targets or has_mt or has_mts):
            errors.append(
                f"{path}.target / .targets / .mooring_target / .motion_targets "
                "均为空或缺失，至少需要一个非空的预测目标"
            )
    _check_target(data, "data")

    # 8. 输入特征列非空
    def _check_features(d: dict, path: str) -> None:
        feats = d.get("input_features")
        wf = d.get("wave_feature")
        mts = d.get("motion_targets")
        has_feats = isinstance(feats, list) and len(feats) > 0
        has_wf = isinstance(wf, str) and wf.strip()
        has_mts = isinstance(mts, list) and len(mts) > 0
        if not (has_feats or has_wf or has_mts):
            errors.append(
                f"{path}.input_features / .wave_feature / .motion_targets "
                "均为空或缺失，至少需要一个非空的输入特征配置"
            )
    _check_features(data, "data")

    # 9. 数值参数校验：look_back（如果存在）
    lb = data.get("look_back")
    if lb is not None and not _is_positive_int(lb):
        errors.append(f"data.look_back 应为正整数，当前值: {lb}")

    # 10. 数值参数校验：epochs / batch_size
    def _check_training_nums(train_src: dict, path: str) -> None:
        ep = train_src.get("epochs")
        bs = train_src.get("batch_size")
        if ep is not None and not _is_positive_int(ep):
            errors.append(f"{path}.epochs 应为正整数，当前值: {ep}")
        if bs is not None and not _is_positive_int(bs):
            errors.append(f"{path}.batch_size 应为正整数，当前值: {bs}")

    if has_training:
        _check_training_nums(config["training"], "training")
    if has_stage1:
        _check_training_nums(config["stage1"], "stage1")
    if has_stage2:
        _check_training_nums(config["stage2"], "stage2")

    # 11. learning_rate 应为正数（如果存在）
    def _check_lr(sec: dict, path: str) -> None:
        lr = sec.get("learning_rate")
        if lr is not None and not _is_positive_number(lr):
            errors.append(f"{path}.learning_rate 应为正数，当前值: {lr}")

    if has_training:
        _check_lr(config["training"], "training")
    if has_stage1:
        _check_lr(config["stage1"], "stage1")
    if has_stage2:
        _check_lr(config["stage2"], "stage2")

    # ── 章节特定校验 ──────────────────────────────────────────────────────
    if task_name is not None:
        _validate_task_specific(config, task_name, errors)

    return errors


def _validate_task_specific(config: dict, task_name: str, errors: list[str]) -> None:
    """各章节（实验任务）特有的配置校验规则。"""
    data = config.get("data", {})
    model = config.get("model", {})
    training = config.get("training", {})
    stage1 = config.get("stage1", {})
    stage2 = config.get("stage2", {})

    if task_name == "s3_motion":
        # 必须字段: data.target, data.look_back, data.train_size, data.valid_size
        if not isinstance(data.get("target"), str) or not data["target"].strip():
            errors.append("s3_motion: data.target 必须为非空字符串（如 Heave/Surge/Pitch）")
        if not _is_positive_int(data.get("look_back")):
            errors.append(f"s3_motion: data.look_back 必须为正整数，当前值: {data.get('look_back')}")
        if not _is_positive_int(data.get("train_size")):
            errors.append(f"s3_motion: data.train_size 必须为正整数，当前值: {data.get('train_size')}")
        if not _is_positive_int(data.get("valid_size")):
            errors.append(f"s3_motion: data.valid_size 必须为正整数，当前值: {data.get('valid_size')}")
        if not isinstance(model.get("name"), str) or not model["name"].strip():
            errors.append("s3_motion: model.name 必须为非空字符串（如 lstm/gru/bilstm）")
        if not isinstance(model.get("units"), list) or len(model["units"]) == 0:
            errors.append(f"s3_motion: model.units 必须为非空列表，当前值: {model.get('units')}")
        if not isinstance(training.get("random_seed"), int):
            errors.append(f"s3_motion: training.random_seed 必须为整数，当前值: {training.get('random_seed')}")

    elif task_name == "s4_regression":
        if not isinstance(data.get("target"), str) or not data["target"].strip():
            errors.append("s4_regression: data.target 必须为非空字符串（如 Force1/Force2）")
        input_feats = data.get("input_features", [])
        if not isinstance(input_feats, list) or len(input_feats) == 0:
            errors.append("s4_regression: data.input_features 必须为非空列表")
        if not _is_positive_int(data.get("train_start")):
            errors.append(f"s4_regression: data.train_start 必须为正整数，当前值: {data.get('train_start')}")
        else:
            train_start = int(data["train_start"])
            train_end = data.get("train_end")
            if not _is_positive_int(train_end):
                errors.append(f"s4_regression: data.train_end 必须为正整数，当前值: {train_end}")
            elif train_start >= int(train_end):
                errors.append(
                    f"s4_regression: data.train_start ({train_start}) 必须小于 data.train_end ({train_end})"
                )
        if not _is_positive_int(data.get("valid_end")):
            errors.append(f"s4_regression: data.valid_end 必须为正整数，当前值: {data.get('valid_end')}")
        if not isinstance(model.get("name"), str) or not model["name"].strip():
            errors.append("s4_regression: model.name 必须为非空字符串（如 hgbdt/bpnn）")
        if not isinstance(training.get("random_seed"), int):
            errors.append(f"s4_regression: training.random_seed 必须为整数，当前值: {training.get('random_seed')}")

    elif task_name == "s4_lstm_mooring":
        if not isinstance(data.get("target"), str) or not data["target"].strip():
            errors.append("s4_lstm_mooring: data.target 必须为非空字符串（如 Force1/Force2）")
        if not _is_positive_int(data.get("look_back")):
            errors.append(f"s4_lstm_mooring: data.look_back 必须为正整数，当前值: {data.get('look_back')}")
        if not _is_positive_int(data.get("train_start")):
            errors.append(f"s4_lstm_mooring: data.train_start 必须为正整数，当前值: {data.get('train_start')}")
        if not _is_positive_int(data.get("train_end")):
            errors.append(f"s4_lstm_mooring: data.train_end 必须为正整数，当前值: {data.get('train_end')}")
        if not _is_positive_int(data.get("valid_end")):
            errors.append(f"s4_lstm_mooring: data.valid_end 必须为正整数，当前值: {data.get('valid_end')}")
        if not isinstance(model.get("name"), str) or not model["name"].strip():
            errors.append("s4_lstm_mooring: model.name 必须为非空字符串（如 lstm）")
        if not isinstance(model.get("units"), list) or len(model["units"]) == 0:
            errors.append(f"s4_lstm_mooring: model.units 必须为非空列表，当前值: {model.get('units')}")

    elif task_name == "s4_hybrid":
        if not isinstance(data.get("wave_feature"), str) or not data["wave_feature"].strip():
            errors.append("s4_hybrid: data.wave_feature 必须为非空字符串（如 H）")
        has_single = isinstance(data.get("mooring_target"), str) and data["mooring_target"].strip()
        has_multi = isinstance(data.get("mooring_targets"), list) and len(data["mooring_targets"]) > 0
        if not has_single and not has_multi:
            errors.append("s4_hybrid: data.mooring_target 或 data.mooring_targets 必须为非空")
        mts = data.get("motion_targets", [])
        if not isinstance(mts, list) or len(mts) == 0:
            errors.append("s4_hybrid: data.motion_targets 必须为非空列表")
        if not _is_positive_int(data.get("look_back")):
            errors.append(f"s4_hybrid: data.look_back 必须为正整数，当前值: {data.get('look_back')}")
        if not isinstance(stage1.get("model"), str) or not stage1["model"].strip():
            errors.append(f"s4_hybrid: stage1.model 必须为非空字符串，当前值: {stage1.get('model')}")
        if not isinstance(stage2.get("model"), str) or not stage2["model"].strip():
            errors.append(f"s4_hybrid: stage2.model 必须为非空字符串，当前值: {stage2.get('model')}")
        if not _is_positive_int(data.get("train_start")):
            errors.append(f"s4_hybrid: data.train_start 必须为正整数，当前值: {data.get('train_start')}")
        if not _is_positive_int(data.get("train_end")):
            errors.append(f"s4_hybrid: data.train_end 必须为正整数，当前值: {data.get('train_end')}")
        if not _is_positive_int(data.get("valid_end")):
            errors.append(f"s4_hybrid: data.valid_end 必须为正整数，当前值: {data.get('valid_end')}")

    elif task_name == "s5_attention":
        if not isinstance(data.get("wave_feature"), str) or not data["wave_feature"].strip():
            errors.append("s5_attention: data.wave_feature 必须为非空字符串（如 H）")
        if not isinstance(data.get("mooring_target"), str) or not data["mooring_target"].strip():
            errors.append("s5_attention: data.mooring_target 必须为非空字符串（如 Force1/Force2）")
        mts = data.get("motion_targets", [])
        if not isinstance(mts, list) or len(mts) == 0:
            errors.append("s5_attention: data.motion_targets 必须为非空列表")
        if not _is_positive_int(data.get("look_back")):
            errors.append(f"s5_attention: data.look_back 必须为正整数，当前值: {data.get('look_back')}")
        compare = model.get("compare_models")
        if not isinstance(compare, list) or len(compare) == 0:
            errors.append(
                f"s5_attention: model.compare_models 必须为非空列表，当前值: {compare}"
            )
        if not isinstance(stage1.get("model") if "model" in stage1 else True, str):
            # s5_attention 的 stage1 不使用单一 model 字段（使用 compare_models），
            # 但 stage2 必须有 model
            pass
        if not isinstance(stage2.get("model"), str) or not stage2["model"].strip():
            errors.append(f"s5_attention: stage2.model 必须为非空字符串，当前值: {stage2.get('model')}")
        if not _is_positive_int(data.get("horizon")):
            errors.append(f"s5_attention: data.horizon 必须为正整数，当前值: {data.get('horizon')}")

    else:
        errors.append(f"未知的 task_name: {task_name}，当前支持: s3_motion, s4_regression, "
                      "s4_lstm_mooring, s4_hybrid, s5_attention")


def validate_or_raise(config: dict, task_name: str | None = None) -> None:
    """调用 validate_config，如有错误则合并为一条 ValueError 抛出。

    这是 validate_config 的便捷包装，适用于「校验不通过即退出」的场景。
    """
    errors = validate_config(config, task_name=task_name)
    if errors:
        header = f"配置校验失败 ({len(errors)} 项):"
        if task_name:
            header = f"配置校验失败 [{task_name}] ({len(errors)} 项):"
        details = "\n  - ".join(errors)
        raise ValueError(f"{header}\n  - {details}")
