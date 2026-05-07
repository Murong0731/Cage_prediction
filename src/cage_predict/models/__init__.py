"""模型定义 —— LSTM, GRU, BiLSTM, BPNN, HGBRT, CNN-BiLSTM-Attention, N-BEATSx。

本包中包含项目中使用的所有预测模型：
  - LSTM (长短期记忆网络)        —— 第3章、第4.3节使用
  - GRU (门控循环单元)           —— 第5章模型比较中使用
  - BiLSTM (双向LSTM)            —— 第5章模型比较中使用
  - BPNN (反向传播神经网络)      —— 第4.2节、第4.4节使用
  - HGBRT (直方图梯度提升回归树) —— 第4.2节使用
  - Attention (CNN-BiLSTM-Attention) —— 第5章使用
  - N-BEATSx (神经基扩展分析)    —— 第5章第三方模型对比
"""

from .lstm import build_lstm_model
from .gru import build_gru_model
from .bilstm import build_bilstm_model
from .bpnn import build_bpnn_model
from .hgbdt import build_and_fit_hgbdt, predict_hgbdt
from .attention import build_attention_model
from .nbeatsx_adapter import build_nbeatsx_model

__all__ = [
    "build_lstm_model",
    "build_gru_model",
    "build_bilstm_model",
    "build_bpnn_model",
    "build_and_fit_hgbdt",
    "predict_hgbdt",
    "build_attention_model",
    "build_nbeatsx_model",
]
