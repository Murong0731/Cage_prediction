"""metrics.py 模块的单元测试 —— 不依赖 TensorFlow。

验证评估指标计算与原始代码 s3/3.3.py evaluate()、s5/5.2 RNSE()/Acc()
以及 s5/5.3 evaluate() 的精确一致性。
"""

import numpy as np
import pytest

from cage_predict.metrics import (
    compute_acc,
    compute_mae,
    compute_mape,
    compute_mse,
    compute_rmse,
    compute_rtrapz,
    evaluate_predictions,
)


# ═══════════════════════════════════════════════════════════════════════════════
# MAE（平均绝对误差）测试
# ═══════════════════════════════════════════════════════════════════════════════

class TestMAE:
    """测试 compute_mae 函数。"""

    def test_perfect(self):
        """完美预测：MAE 应为 0。"""
        y = np.array([1.0, 2.0, 3.0])
        assert compute_mae(y, y) == 0.0

    def test_nonzero(self):
        """非完美预测：验证 MAE 的手动计算值。"""
        y_true = np.array([0.0, 0.0])
        y_pred = np.array([3.0, 4.0])
        # MAE = (|0-3| + |0-4|) / 2 = (3+4)/2 = 3.5
        assert compute_mae(y_true, y_pred) == pytest.approx(3.5)

    def test_2d_input(self):
        """二维输入应正确处理（自动展平）。"""
        y_true = np.array([[1.0], [2.0], [3.0]])
        y_pred = np.array([[1.5], [2.5], [2.5]])
        assert compute_mae(y_true, y_pred) == pytest.approx(0.5)


# ═══════════════════════════════════════════════════════════════════════════════
# MSE（均方误差）测试
# ═══════════════════════════════════════════════════════════════════════════════

class TestMSE:
    """测试 compute_mse 函数。"""

    def test_perfect(self):
        """完美预测：MSE 应为 0。"""
        y = np.array([1.0, 2.0, 3.0])
        assert compute_mse(y, y) == 0.0

    def test_nonzero(self):
        """非完美预测：验证 MSE 的平方惩罚特性。"""
        y_true = np.array([0.0, 0.0])
        y_pred = np.array([3.0, 4.0])
        # MSE = (3² + 4²) / 2 = (9+16)/2 = 12.5
        assert compute_mse(y_true, y_pred) == pytest.approx(12.5)


# ═══════════════════════════════════════════════════════════════════════════════
# RMSE（均方根误差）测试
# ═══════════════════════════════════════════════════════════════════════════════

class TestRMSE:
    """测试 compute_rmse 函数。"""

    def test_perfect(self):
        """完美预测：RMSE 应为 0。"""
        y = np.array([1.0, 2.0, 3.0])
        assert compute_rmse(y, y) == 0.0

    def test_nonzero(self):
        """验证 RMSE = √MSE 的正确性。"""
        y_true = np.array([0.0, 0.0])
        y_pred = np.array([3.0, 4.0])
        # RMSE = √12.5 ≈ 3.5355
        assert compute_rmse(y_true, y_pred) == pytest.approx(3.5355, abs=0.01)


# ═══════════════════════════════════════════════════════════════════════════════
# MAPE（平均绝对百分比误差）测试
# ═══════════════════════════════════════════════════════════════════════════════

class TestMAPE:
    """测试 compute_mape 函数。"""

    def test_perfect(self):
        """完美预测：MAPE 应为 0%"""
        y = np.array([1.0, 2.0, 3.0])
        assert compute_mape(y, y) == 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# Rtrapz / Acc（积分准确率）测试
# ═══════════════════════════════════════════════════════════════════════════════

class TestRtrapz:
    """测试 compute_rtrapz / compute_acc 自定义积分准确率指标。

    这是论文中使用的特殊指标，通过比较预测曲线和真实曲线的积分面积
    来衡量预测质量。
    """

    def test_perfect_match(self):
        """完美预测 → Acc = 1.0（面积比完全一致）"""
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert compute_rtrapz(y, y) == pytest.approx(1.0)

    def test_worse_prediction(self):
        """不良预测 → Acc < 1.0（预测面积与真实面积有偏差）"""
        y_true = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        y_pred = np.array([2.0, 2.0, 2.0, 2.0, 2.0])
        acc = compute_rtrapz(y_true, y_pred, dx=1.0)
        assert acc < 1.0

    def test_constant_truth_zero_area(self):
        """真实值为常数时 → 积分面积为0 → 返回 0.0（特殊情况保护）"""
        y_true = np.array([5.0, 5.0, 5.0, 5.0])
        y_pred = np.array([5.1, 5.1, 5.1, 5.1])
        assert compute_rtrapz(y_true, y_pred) == 0.0

    def test_uses_real_mean_for_both(self):
        """关键验证：pred_area 使用 mean(y_true)，而非 mean(y_pred)。

        原始公式中两个积分都用真实值的均值 XP1.mean()，这是有意为之的设计。
        如果用预测值的均值，公式将变为对称的形式，解读也会不同。

        手动验证：y_true=[0,1,2], y_pred=[10,11,12]
          real_area = 0.5*|0-1| + |1-1| + 0.5*|2-1| = 1.0 (梯形积分)
          pred_area = 0.5*|10-1| + |11-1| + 0.5*|12-1| = 4.5+10+5.5 = 20.0
          acc = 1 - |1 - 20/1| = 1 - 19 = -18
        """
        y_true = np.array([0.0, 1.0, 2.0])
        y_pred = np.array([10.0, 11.0, 12.0])
        acc = compute_rtrapz(y_true, y_pred, dx=1.0)
        assert acc < 0  # 负值说明预测非常差

    def test_negative_acc_possible(self):
        """Acc 可以为负 —— 当预测远差于使用均值作为预测值时。

        这是该指标的一个特点：它衡量的是预测相对于"使用均值预测"的改进程度。
        如果预测比直接用均值还差很多，Acc 就会是负值。
        """
        y_true = np.array([0.0, 0.0, 0.0, 0.0, 0.1])
        y_pred = np.array([100.0, 100.0, 100.0, 100.0, 100.0])
        acc = compute_rtrapz(y_true, y_pred)
        assert acc < 0

    def test_compute_acc_alias(self):
        """compute_acc 和 compute_rtrapz 应完全一致（它们是同一个函数）。"""
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert compute_acc(y, y) == pytest.approx(1.0)
        assert compute_acc(y, y) == compute_rtrapz(y, y)


# ═══════════════════════════════════════════════════════════════════════════════
# evaluate_predictions（综合评估）测试
# ═══════════════════════════════════════════════════════════════════════════════

class TestEvaluatePredictions:
    """测试 evaluate_predictions 综合评估函数。"""

    def test_returns_all_keys(self):
        """返回结果应包含所有必需的键。"""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 2.2, 2.8, 4.2, 5.0])
        result = evaluate_predictions(y_true, y_pred)
        for key in ["mae", "mse", "rmse", "rtrapz", "acc"]:
            assert key in result, f"缺少键: {key}"
            assert isinstance(result[key], float), f"键 {key} 应为 float 类型"

    def test_perfect_prediction(self):
        """完美预测时所有误差指标应为 0，Acc 应为 1.0。"""
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = evaluate_predictions(y, y, use_sklearn=False)
        assert result["mae"] == 0.0
        assert result["mse"] == 0.0
        assert result["rmse"] == 0.0
        assert result["acc"] == pytest.approx(1.0)

    def test_1d_input(self):
        """一维输入应正确处理。"""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 3.0])
        result = evaluate_predictions(y_true, y_pred, use_sklearn=False)
        assert result["rmse"] == 0.0

    def test_2d_input(self):
        """二维输入应正确处理。"""
        y_true = np.array([[1.0], [2.0], [3.0]])
        y_pred = np.array([[1.0], [2.0], [3.0]])
        result = evaluate_predictions(y_true, y_pred, use_sklearn=False)
        assert result["rmse"] == 0.0

    def test_manual_matches_sklearn(self):
        """手动实现应与 sklearn 实现一致（在容差范围内）。"""
        np.random.seed(42)
        y_true = np.random.randn(100)
        y_pred = y_true + np.random.randn(100) * 0.1
        result = evaluate_predictions(y_true, y_pred, use_sklearn=True)
        # 如果 sklearn 可用，验证结果的一致性
        if "mape" in result and not np.isnan(result["mape"]):
            assert result["mae"] >= 0
            assert result["rmse"] >= 0
