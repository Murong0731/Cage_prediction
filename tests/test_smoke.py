"""烟雾测试 —— 需要 TensorFlow，验证最小化管道的完整性。

烟雾测试（Smoke Test）是一种快速验证，目的是确保代码的基本功能
可以正常运行（即"上电后不冒烟"），而不是测试所有细节。

这些测试会实际构建并训练小型神经网络模型，因此需要 TensorFlow。
"""

import numpy as np
import pytest

# TensorFlow 是软依赖：如果没安装就跳过烟雾测试
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


@pytest.mark.skipif(not TF_AVAILABLE, reason="TensorFlow 未安装")
class TestSmokeLSTM:
    """LSTM 模型的最小化端到端烟雾测试。

    验证：
      1. 模型可以成功构建（build）
      2. 模型可以成功训练（fit）
      3. BPNN 和 Attention 模型也可以构建和训练
    """

    def test_build_lstm_model(self):
        """测试单层 LSTM 模型的构建。"""
        from cage_predict.models.lstm import build_lstm_model

        model = build_lstm_model(
            input_shape=(10, 2),  # 10个时间步, 2个特征
            num_layers=1,
            units=[8],             # 小模型以加快测试
            dropout=0.0,
        )
        assert model is not None
        assert len(model.layers) >= 2  # 至少有 LSTM + Dense 两层

    def test_build_lstm_model_3layer(self):
        """测试三层 LSTM 模型的构建。

        三层模型包含：LSTM → Dropout → LSTM → Dropout → LSTM → Dropout → Dense
        """
        from cage_predict.models.lstm import build_lstm_model

        model = build_lstm_model(
            input_shape=(10, 2),
            num_layers=3,
            units=[8, 16, 16],
            dropout=0.2,
        )
        assert model is not None

    def test_lstm_fit_one_epoch(self):
        """测试 LSTM 在合成数据上训练一个 epoch —— 验证整个训练管道可运行。

        使用随机生成的合成数据，只训练1个epoch来验证代码没有错误，
        而不是检验模型的实际学习能力。
        """
        from cage_predict.models.lstm import build_lstm_model

        model = build_lstm_model(
            input_shape=(5, 1),
            num_layers=1,
            units=[4],
            dropout=0.0,
        )
        # 合成数据：50个样本，每个5个时间步，1个特征
        x = np.random.randn(50, 5, 1).astype(np.float32)
        y = np.random.randn(50, 1).astype(np.float32)
        history = model.fit(x, y, epochs=1, batch_size=8, verbose=0, shuffle=False)
        assert len(history.history["loss"]) == 1  # 1个epoch应有1个损失值

    def test_bpnn_fit_one_epoch(self):
        """测试 BPNN 在合成数据上训练一个 epoch。

        验证 BPNN 模型可以成功构建和训练，包括 Sequential API 模式。
        """
        from cage_predict.models.bpnn import build_bpnn_model

        model = build_bpnn_model(input_dim=3, hidden_units=[8, 8], use_functional_api=False)
        x = np.random.randn(50, 3).astype(np.float32)
        y = np.random.randn(50, 1).astype(np.float32)
        history = model.fit(x, y, epochs=1, batch_size=8, verbose=0)
        assert len(history.history["loss"]) == 1

    def test_attention_model_build(self):
        """测试 CNN-BiLSTM-Attention 模型的构建。

        这是第5章的核心模型，确保其可以成功构建（不验证训练）。
        """
        from cage_predict.models.attention import build_attention_model

        model = build_attention_model(
            input_shape=(20, 1),   # 20个时间步, 1个特征
            conv_filters=8,         # 小型卷积核以加快测试
            lstm_units=8,
            dropout_rate=0.0,
        )
        assert model is not None
