"""HGBRT（直方图梯度提升回归树）模型构建器。

HGBRT 是 Gradient Boosting 的高效实现，使用直方图加速分裂点搜索，
比传统的 GBDT（梯度提升决策树）快很多倍，同时保持相近的精度。

核心概念：
  - Gradient Boosting：每次迭代训练一棵新树来拟合前一轮的残差
  - Histogram（直方图）：将连续特征离散化为有限个分箱，加速计算
  - 回归树：每棵树的叶子节点输出一个连续值

本模块包装了 scikit-learn 的 HistGradientBoostingRegressor，
并实现了原始 4.2 节代码中的 GridSearchCV 超参数调优逻辑。

原始代码参考：s4/4.2/4.2.py:453 —— HGBRT() 函数
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def _param_grid_small() -> dict:
    """烟雾测试用的缩小版超参数搜索空间（快速验证）。

    只搜索少量参数组合，确保流程可以在短时间内完成。
    """
    return dict(max_depth=[3, 5], max_leaf_nodes=[15, 31], min_samples_leaf=[10, 20])


def _param_grid_full() -> dict:
    """完整的超参数搜索空间，与原始 4.2 节代码一致。

    搜索说明：
      - max_depth [3..25]: 树的最大深度，控制模型复杂度
      - max_leaf_nodes [25..44]: 叶节点最大数量，限制树的宽度
      - min_samples_leaf [10..24]: 叶子节点最少样本数，防止过拟合
    """
    return dict(
        max_depth=[3, 5, 6, 7, 9, 12, 15, 17, 25],
        max_leaf_nodes=[i for i in range(25, 45)],
        min_samples_leaf=[i for i in range(10, 25)],
    )


def build_and_fit_hgbdt(
    train_x: np.ndarray,
    train_y: np.ndarray,
    smoke_test: bool = False,
    random_state: int = 42,
) -> tuple[Any, dict[str, Any]]:
    """网格搜索 HGBRT 模型并拟合最佳估计器。

    原始流程（s4/4.2/4.2.py:453–469）：
        1. 使用 GridSearchCV 在参数空间中搜索最佳超参数
        2. 提取 best_params → 用这些参数构建最佳模型
        3. 最佳模型在全部训练数据上拟合
        4. 返回预训练模型和最佳参数

    参数
    ----------
    train_x : (n_samples, n_features) 训练特征。
    train_y : (n_samples, 1) 或 (n_samples,) 训练目标值。
    smoke_test : 如果为 True，使用缩小的参数网格以快速验证。
    random_state : 随机种子，确保结果可重复。

    返回
    -------
    best_model : 拟合好的 HistGradientBoostingRegressor 模型。
    best_params : 选定的超参数字典。
    """
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.model_selection import GridSearchCV

    # 根据是否烟雾测试选择搜索空间大小
    param_grid = _param_grid_small() if smoke_test else _param_grid_full()

    logger.info(
        "HGBRT 网格搜索: max_depth=%s, max_leaf_nodes=%d 个值, min_samples_leaf=%d 个值",
        param_grid["max_depth"],
        len(param_grid["max_leaf_nodes"]),
        len(param_grid["min_samples_leaf"]),
    )

    # GridSearchCV：交叉验证网格搜索
    # scoring="neg_mean_absolute_error"：使用负 MAE 评分（越大越好）
    # n_jobs=-1：使用所有 CPU 核心并行搜索
    #
    # TODO: 当前使用默认 KFold（shuffle=False）做 CV 分割。若输入数据存在时间自相关
    # （如连续时刻的测量值），普通 KFold 可能造成数据泄露。此时应改用 TimeSeriesSplit：
    #   from sklearn.model_selection import TimeSeriesSplit
    #   cv=TimeSeriesSplit(n_splits=5)
    # 上游 s4_regression.py 已按时间索引切分训练/验证集，此处 CV 仅在训练集内部评估，
    # 对于第 4.2 节的横截面式回归任务（当前时刻运动 → 当前时刻力）影响有限，
    # 故暂不改动，后续若将 HGBDT 用于纯时间序列预测时再调整。
    grid_search = GridSearchCV(
        HistGradientBoostingRegressor(random_state=random_state),
        param_grid,
        scoring="neg_mean_absolute_error",  # 负 MAE —— sklearn 中分数越高越好
        n_jobs=-1,                            # 使用所有可用的 CPU 核心
    )
    grid_result = grid_search.fit(train_x, train_y.ravel())
    logger.info("最佳得分: %.6f  最佳参数: %s", grid_result.best_score_, grid_result.best_params_)

    # 用最佳参数重新构建模型并在全部训练数据上拟合
    best_parameters = grid_search.best_estimator_.get_params()
    best_model = HistGradientBoostingRegressor(
        max_depth=best_parameters["max_depth"],
        max_leaf_nodes=best_parameters["max_leaf_nodes"],
        min_samples_leaf=best_parameters["min_samples_leaf"],
        loss="squared_error",  # "least_squares" for sklearn<1.0, "squared_error" for >=1.0
        random_state=random_state,
    )
    best_model.fit(train_x, train_y.ravel())
    logger.info("HGBRT 已在 %d 个样本上拟合 —— 最佳参数=%s", len(train_x), best_parameters)
    return best_model, best_parameters


def predict_hgbdt(model, valid_x: np.ndarray) -> np.ndarray:
    """使用拟合好的 HGBRT 模型进行预测。

    参数
    ----------
    model : 已拟合的 HistGradientBoostingRegressor。
    valid_x : 验证集的输入特征。

    返回
    -------
    np.ndarray : (n_samples, 1) 形状的预测值数组。
    """
    pre_y = model.predict(valid_x)
    if pre_y.ndim == 1:
        pre_y = pre_y.reshape(-1, 1)  # 确保是列向量
    return pre_y
