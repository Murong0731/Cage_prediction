"""XGBoost（极端梯度提升）模型构建器。

XGBoost 是基于梯度提升树的集成学习方法，在第4.2节中作为 HGBRT /
BPNN 的对比模型。它通过网格搜索自动调优超参数。

原始代码参考：s4/4.2/4.2.py —— XGBoost() 函数和 GridSearchCV 调参。
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

# 原始代码中的网格搜索参数空间
DEFAULT_PARAM_GRID = dict(
    max_depth=[3, 5, 7, 9, 11, 13, 15],
    min_child_weight=[1, 3],
    gamma=[0, 0.05, 0.1],
    subsample=[0.6, 0.7, 0.8, 0.9, 1],
    colsample_bytree=[0.6, 0.7, 0.8, 0.9, 1],
    reg_alpha=[0, 0.001, 0.005, 0.025, 0.05],
    learning_rate=[0.025, 0.05, 0.1, 0.2],
    n_estimators=[50, 100, 200, 400],
)

# 烟雾测试模式使用更小的搜索空间以加速
SMOKE_PARAM_GRID = dict(
    max_depth=[3, 5],
    min_child_weight=[1],
    gamma=[0],
    subsample=[0.8],
    colsample_bytree=[0.8],
    reg_alpha=[0],
    learning_rate=[0.1],
    n_estimators=[50],
)


def build_and_fit_xgboost(
    train_x: np.ndarray,
    train_y: np.ndarray,
    param_grid: dict | None = None,
    smoke_test: bool = False,
    random_state: int = 42,
    n_jobs: int = -1,
) -> tuple[object, dict]:
    """使用网格搜索训练 XGBoost 模型。

    参数
    ----------
    train_x : 形状为 (n_samples, n_features) 的训练输入。
    train_y : 形状为 (n_samples,) 或 (n_samples, 1) 的训练目标。
    param_grid : 超参数网格字典。如果为 None，则使用默认的完整网格。
    smoke_test : 如果为 True，使用缩小的网格以加速。
    random_state : 随机种子，用于可重复性。
    n_jobs : 并行作业数，-1 表示使用所有核心。

    返回
    -------
    tuple : (已拟合的最佳 XGBRegressor 模型, 最佳参数字典)。
    """
    from sklearn.model_selection import GridSearchCV
    from xgboost import XGBRegressor

    if param_grid is None:
        param_grid = SMOKE_PARAM_GRID if smoke_test else DEFAULT_PARAM_GRID

    train_y = train_y.ravel()

    model = XGBRegressor(random_state=random_state, objective="reg:squarederror")
    grid = GridSearchCV(
        model, param_grid,
        scoring="neg_mean_absolute_error",
        n_jobs=n_jobs,
    )
    grid.fit(train_x, train_y)

    logger.info("XGBoost 最佳分数: %f  最佳参数: %s", grid.best_score_, grid.best_params_)

    best_params = grid.best_params_
    final_model = XGBRegressor(
        max_depth=best_params["max_depth"],
        min_child_weight=best_params["min_child_weight"],
        gamma=best_params["gamma"],
        subsample=best_params["subsample"],
        colsample_bytree=best_params["colsample_bytree"],
        reg_alpha=best_params["reg_alpha"],
        learning_rate=best_params["learning_rate"],
        n_estimators=best_params["n_estimators"],
        random_state=random_state,
        objective="reg:squarederror",
    )
    final_model.fit(train_x, train_y)
    logger.info("XGBoost 最终模型已拟合")
    return final_model, best_params


def predict_xgboost(model: object, x: np.ndarray) -> np.ndarray:
    """使用已训练的 XGBoost 模型进行预测。

    参数
    ----------
    model : 已拟合的 XGBRegressor。
    x : 形状为 (n_samples, n_features) 的输入。

    返回
    -------
    np.ndarray : 形状为 (n_samples,) 的预测值。
    """
    return model.predict(x)
