"""cage_predict —— 基于深度学习的网箱动力响应预测工具包。

本工具包实现了以下章节的模型：
  - 第3章：LSTM/GRU/BiLSTM 运动响应预测
  - 第4.2节：HGBRT/BPNN 系泊力回归预测
  - 第4.3节：LSTM 系泊力时间序列预测
  - 第4.4节：混合联合预测（波浪→运动→系泊力）
  - 第5章：CNN-BiLSTM-Attention 模型比较与泛化测试

使用方式：
    python -m cage_predict run-s3 --config configs/s3_motion.yaml
"""

__version__ = "0.1.0"
