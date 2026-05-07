"""data.py 模块的单元测试 —— 不依赖 TensorFlow。

验证数据处理函数的精确行为是否与原始代码 s3/3.3.py, s3/3.4.py 一致。
每个测试类对应 data.py 中的一个函数族。
"""

import numpy as np
import pytest

from cage_predict.data import (
    apply_minmax_scaler,
    apply_minmax_scalers,
    deal_data1,
    deal_data2,
    inverse_scale,
    save_results_csv,
    select_features,
    series_to_supervised,
    split_sequence,
    split_train_valid,
)

# 检查可选依赖是否可用（ski-learn 是软依赖）
try:
    from sklearn.preprocessing import MinMaxScaler  # noqa: F401
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False


# ═══════════════════════════════════════════════════════════════════════════════
# series_to_supervised 函数测试
# ═══════════════════════════════════════════════════════════════════════════════

class TestSeriesToSupervised:
    """测试时间序列 → 监督学习格式的转换。

    核心功能：将连续的时间序列数据重新组织为"过去→未来"的输入-输出对。
    """

    def test_basic_shape(self):
        """测试基本形状：100个样本, 2个变量, 回顾窗口=5, 预测步数=1 → 输出应为 (95, 12)。

        形状计算：rows = 100 - 5 - 1 + 1 = 95
                 cols = (5+1) * 2 = 12
        """
        data = np.random.randn(100, 2)
        result = series_to_supervised(data, n_in=5, n_out=1, dropnan=True)
        assert result.shape == (95, 12)

    def test_n_out_3(self):
        """测试多步预测：n_out=3 时会添加 3 组未来列（t, t+1, t+2）。

        形状计算：rows = 100 - 3 - 2 = 95
                 cols = (3+3) * 2 = 12
        """
        data = np.random.randn(100, 2)
        result = series_to_supervised(data, n_in=3, n_out=3, dropnan=True)
        assert result.shape == (95, 12)

    def test_no_dropnan(self):
        """测试保留 NaN 的模式：不丢弃含 NaN 的行，输出行数应与输入相同。"""
        data = np.random.randn(100, 2)
        result = series_to_supervised(data, n_in=5, n_out=1, dropnan=False)
        assert result.shape == (100, 12)

    def test_list_input(self):
        """测试列表输入（原始代码支持单变量列表格式）。

        形状计算：rows = 10 - 2 - 1 + 1 = 8
                 cols = (2+1) * 1 = 3
        """
        data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        result = series_to_supervised(data, n_in=2, n_out=1, dropnan=True)
        assert result.shape == (8, 3)

    def test_column_structure(self):
        """测试列结构：输出的最后一组列（var1(t), var2(t)）应该是未移动的当前值。

        验证 var1(t) 和 var2(t) 等于原始数据的对应行。
        """
        data = np.column_stack([np.arange(20.0), np.arange(20.0) * 10])
        result = series_to_supervised(data, n_in=3, n_out=1, dropnan=True)
        assert result[0, -2] == data[3, 0]  # var1(t) 对应原始数据的第4行（索引3）
        assert result[0, -1] == data[3, 1]  # var2(t) 对应原始数据的第4行（索引3）

    def test_shift_correctness(self):
        """测试位移正确性：var1(t-1) 应该等于 data[row-1, 0]。

        细致验证每个滞后列的值是否正确位移。
        """
        data = np.column_stack([np.arange(20.0), np.arange(20.0)])
        result = series_to_supervised(data, n_in=3, n_out=1, dropnan=True)
        # 列布局: var1(t-3), var2(t-3), var1(t-2), var2(t-2), var1(t-1), var2(t-1), var1(t), var2(t)
        assert result[0, 0] == data[0, 0]  # var1(t-3) = 原始第1行
        assert result[0, 4] == data[2, 0]  # var1(t-1) = 原始第3行
        assert result[0, 6] == data[3, 0]  # var1(t) = 原始第4行


# ═══════════════════════════════════════════════════════════════════════════════
# deal_data1 / deal_data2 函数测试
# ═══════════════════════════════════════════════════════════════════════════════

class TestDealData:
    """测试 deal_data1 和 deal_data2 的列过滤逻辑。

    两者的关键区别：
      - deal_data1：保留所有输入特征的历史滞后列 + 所有输出列
      - deal_data2：只保留第一个输入特征的历史滞后列 + 输出列
    """

    def test_deal_data1_keeps_input_features_and_output(self):
        """deal_data1 应保留前 features_number 个滞后列 + 输出列。

        2个特征, n_in=5: 保留索引 0,1 + 最后一列 = 3列。
        """
        data = np.random.randn(100, 2)
        result = deal_data1(data, features_number=2, time_steps=5)
        assert result.shape[1] == 3

    def test_deal_data2_keeps_only_first_feature_and_output(self):
        """deal_data2 应只保留第一个特征滞后列 + 输出列。

        2个特征, n_in=5: 只保留第一列特征 + 输出列 = 2列。
        """
        data = np.random.randn(100, 2)
        result = deal_data2(data, features_number=2, time_steps=5)
        assert result.shape[1] == 2

    def test_deal_data1_vs_deal_data2_difference(self):
        """deal_data1 的列数应多于 deal_data2（保留了更多特征列）。"""
        data = np.random.randn(100, 3)
        r1 = deal_data1(data, features_number=3, time_steps=3)
        r2 = deal_data2(data, features_number=3, time_steps=3)
        assert r1.shape[1] > r2.shape[1]


# ═══════════════════════════════════════════════════════════════════════════════
# split_sequence 函数测试
# ═══════════════════════════════════════════════════════════════════════════════

class TestSplitSequence:
    """测试滑动窗口序列切分。

    将二维表格数据转换为 (X, y) 格式的3D张量序列。
    """

    def test_shape(self):
        """测试输出形状：50行3列数据, n_past=5 → X.shape=(46, 5, 2), y.shape=(46,)。

        X 的最后一维是2而不是3，因为 split_sequence 将最后一列作为标签 y。
        """
        data = np.random.randn(50, 3)
        x, y = split_sequence(data, n_past=5)
        assert x.shape == (46, 5, 2)
        assert y.shape == (46,)

    def test_y_is_last_column(self):
        """验证 y 是数据的最后一列。"""
        data = np.column_stack([np.zeros((20, 2)), np.arange(20.0)])
        x, y = split_sequence(data, n_past=3)
        assert y[0] == data[0, -1]

    def test_x_excludes_last_column(self):
        """验证 X 不包含最后一列（最后一列是 y）。"""
        data = np.random.randn(20, 4)
        x, y = split_sequence(data, n_past=3)
        assert x.shape[2] == 3  # 4 - 1 = 3列特征

    def test_boundary(self):
        """边界测试：当 n_past 超过数据长度时，应返回空数组。"""
        data = np.random.randn(10, 3)
        x, y = split_sequence(data, n_past=20)
        assert len(x) == 0
        assert len(y) == 0


# ═══════════════════════════════════════════════════════════════════════════════
# split_train_valid 函数测试
# ═══════════════════════════════════════════════════════════════════════════════

class TestSplitTrainValid:
    """测试按时间索引的训练/验证集切分。

    时间序列数据不能随机切分，必须按时间顺序：训练集在前，验证集在后。
    """

    def test_basic_split(self):
        """基本切分：100个样本, train[0:70], valid[70:100]"""
        x = np.random.randn(100, 5, 2)
        y = np.random.randn(100)
        tx, ty, vx, vy = split_train_valid(x, y, train_start=0, train_end=70, valid_end=100)
        assert tx.shape == (70, 5, 2)
        assert ty.shape == (70, 1)
        assert vx.shape == (30, 5, 2)
        assert vy.shape == (30, 1)

    def test_offset_train_start(self):
        """偏移起始索引：跳过前50个样本, train[50:120], valid[120:180]"""
        x = np.random.randn(200, 5, 2)
        y = np.random.randn(200)
        tx, ty, vx, vy = split_train_valid(x, y, train_start=50, train_end=120, valid_end=180)
        assert tx.shape == (70, 5, 2)
        assert vx.shape == (60, 5, 2)

    def test_3param_compatibility(self):
        """兼容性测试：train_start=0 应与原始三参数版本结果相同。"""
        x = np.random.randn(100, 5, 1)
        y = np.random.randn(100)
        tx, ty, vx, vy = split_train_valid(x, y, train_start=0, train_end=70, valid_end=100)
        assert tx.shape[0] == 70


# ═══════════════════════════════════════════════════════════════════════════════
# MinMaxScaler 测试
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(not _HAS_SKLEARN, reason="scikit-learn 未安装")
class TestScaler:
    """测试 MinMaxScaler 归一化和反归一化。"""

    def test_range_minus1_to_1(self):
        """归一化后数据应在 [-1, 1] 范围内。"""
        data = np.array([[0.0], [10.0], [5.0]])
        scaled, scaler = apply_minmax_scaler(data)
        assert scaled.min() == pytest.approx(-1.0)
        assert scaled.max() == pytest.approx(1.0)

    def test_inverse_roundtrip(self):
        """归一化 → 反归一化 应还原为原始数据（往返一致性）。"""
        data = np.array([[0.0], [10.0], [5.0], [7.5]])
        scaled, scaler = apply_minmax_scaler(data)
        restored = inverse_scale(scaled, scaler)
        np.testing.assert_array_almost_equal(restored.flatten(), data.flatten())

    def test_inverse_1d_input(self):
        """反归一化应兼容一维输入（自动 reshape 为二维）。"""
        data = np.array([[0.0], [10.0], [5.0]])
        scaled, scaler = apply_minmax_scaler(data)
        restored = inverse_scale(scaled.ravel(), scaler)
        np.testing.assert_array_almost_equal(restored.flatten(), data.flatten())

    def test_apply_minmax_scalers(self):
        """批量归一化：每个列应独立归一化到 [-1, 1]。

        不同列的原始值范围不同，但归一化后都应该是 [-1, 1]。
        """
        col1 = np.array([[1.0], [2.0], [3.0]])
        col2 = np.array([[10.0], [20.0], [30.0]])
        scaled_list, scaler_list = apply_minmax_scalers([col1, col2])
        assert len(scaled_list) == 2
        assert len(scaler_list) == 2
        assert scaled_list[0].min() == pytest.approx(-1.0)
        assert scaled_list[0].max() == pytest.approx(1.0)
        assert scaled_list[1].min() == pytest.approx(-1.0)
        assert scaled_list[1].max() == pytest.approx(1.0)


# ═══════════════════════════════════════════════════════════════════════════════
# select_features 函数测试
# ═══════════════════════════════════════════════════════════════════════════════

class TestSelectFeatures:
    """测试特征列选择功能。"""

    def test_basic(self):
        """基本测试：从 DataFrame 中选择指定列。"""
        import pandas as pd
        df = pd.DataFrame({"A": [1, 2], "B": [3, 4], "C": [5, 6]})
        result = select_features(df, ["A", "C"])
        assert result.shape == (2, 2)

    def test_missing_column_raises(self):
        """请求不存在的列应抛出 KeyError。"""
        import pandas as pd
        df = pd.DataFrame({"A": [1, 2]})
        with pytest.raises(KeyError):
            select_features(df, ["B"])


# ═══════════════════════════════════════════════════════════════════════════════
# save_results_csv 函数测试
# ═══════════════════════════════════════════════════════════════════════════════

class TestSaveResults:
    """测试结果保存功能。"""

    def test_save_and_read(self, tmp_path):
        """保存后应能正确读取：两列分别为真实值和预测值。

        tmp_path 是 pytest 提供的临时目录，测试后自动清理。
        """
        real = np.array([1.0, 2.0, 3.0])
        pred = np.array([1.1, 2.2, 2.8])
        path = tmp_path / "sub" / "result.csv"
        save_results_csv(path, real, pred)
        loaded = np.loadtxt(path, delimiter=",")
        assert loaded.shape == (3, 2)
        np.testing.assert_array_almost_equal(loaded[:, 0], real)
        np.testing.assert_array_almost_equal(loaded[:, 1], pred)
