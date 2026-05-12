"""N-BEATSx 模型适配器 —— PyTorch 核心 + Keras-like API。

从原版 N-BEATSx（https://github.com/cchallu/nbeatsx）提取核心 PyTorch 模块，
去掉 EPF 电价预测专用的 filter_input_vars / include_var_dict / ExogenousBasis，
保留通用的 Identity / Trend / Seasonality 基函数和 TCN 模块。

通过 NBeatsxAdapter 提供与现有 Keras 模型兼容的 fit() / predict() 接口，
使其可直接注册到 s5_attention.py 的 _build_stage1_model() 分发机制中。
"""

from __future__ import annotations

import logging
import math
import os
from typing import Tuple

import numpy as np

# PyTorch CUDA DLL 可能因驱动不兼容而加载失败 (RTX 5070 compute 12.0)。
# N-BEATSx 参数量小（~35K），CPU 训练即可满足需求，因此强制排除 GPU 避免导入失败。
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch as t           # noqa: E402
import torch.nn as nn       # noqa: E402
import torch.nn.functional as F  # noqa: E402

logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# PyTorch nn.Module 核心（提取自原版 nbeats_model.py + tcn.py）
# ══════════════════════════════════════════════════════════════════════════════


class Chomp1d(nn.Module):
    """裁切因果卷积的尾部填充，保证输入输出长度一致。"""

    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, : -self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """TCN 残差块：膨胀因果卷积 + 残差连接。"""

    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super().__init__()
        self.conv1 = nn.utils.weight_norm(
            nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.utils.weight_norm(
            nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.relu2, self.dropout2,
        )
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    """时间卷积网络：多层膨胀因果卷积堆叠。"""

    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super().__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2**i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers.append(
                TemporalBlock(
                    in_channels, out_channels, kernel_size,
                    stride=1, dilation=dilation_size,
                    padding=(kernel_size - 1) * dilation_size, dropout=dropout,
                )
            )
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# ══════════════════════════════════════════════════════════════════════════════
# N-BEATS 基函数（Basis）
# ══════════════════════════════════════════════════════════════════════════════


class IdentityBasis(nn.Module):
    """恒等基函数：直接将 theta 切分为 backcast + forecast，不做变换。"""

    def __init__(self, backcast_size: int, forecast_size: int):
        super().__init__()
        self.backcast_size = backcast_size
        self.forecast_size = forecast_size

    def forward(self, theta: t.Tensor, insample_x_t=None, outsample_x_t=None) -> Tuple[t.Tensor, t.Tensor]:
        backcast = theta[:, : self.backcast_size]
        forecast = theta[:, -self.forecast_size:]
        return backcast, forecast


class TrendBasis(nn.Module):
    """多项式趋势基函数：用 Legendre 多项式展开时间趋势。"""

    def __init__(self, degree_of_polynomial: int, backcast_size: int, forecast_size: int):
        super().__init__()
        polynomial_size = degree_of_polynomial + 1
        self.backcast_basis = nn.Parameter(
            t.tensor(
                np.concatenate([
                    np.power(np.arange(backcast_size, dtype=np.float64) / backcast_size, i)[None, :]
                    for i in range(polynomial_size)
                ]),
                dtype=t.float32,
            ),
            requires_grad=False,
        )
        self.forecast_basis = nn.Parameter(
            t.tensor(
                np.concatenate([
                    np.power(np.arange(forecast_size, dtype=np.float64) / forecast_size, i)[None, :]
                    for i in range(polynomial_size)
                ]),
                dtype=t.float32,
            ),
            requires_grad=False,
        )

    def forward(self, theta: t.Tensor, insample_x_t=None, outsample_x_t=None) -> Tuple[t.Tensor, t.Tensor]:
        cut_point = self.forecast_basis.shape[0]
        backcast = t.einsum("bp,pt->bt", theta[:, cut_point:], self.backcast_basis)
        forecast = t.einsum("bp,pt->bt", theta[:, :cut_point], self.forecast_basis)
        return backcast, forecast


class SeasonalityBasis(nn.Module):
    """季节性基函数：用傅里叶级数（sin/cos）展开周期性模式。"""

    def __init__(self, harmonics: int, backcast_size: int, forecast_size: int):
        super().__init__()
        frequency = np.append(
            np.zeros(1, dtype=np.float32),
            np.arange(harmonics, harmonics / 2 * forecast_size, dtype=np.float32) / harmonics,
        )[None, :]
        backcast_grid = (
            -2 * np.pi * (np.arange(backcast_size, dtype=np.float32)[:, None] / forecast_size) * frequency
        )
        forecast_grid = (
            2 * np.pi * (np.arange(forecast_size, dtype=np.float32)[:, None] / forecast_size) * frequency
        )

        backcast_cos = t.tensor(np.transpose(np.cos(backcast_grid)), dtype=t.float32)
        backcast_sin = t.tensor(np.transpose(np.sin(backcast_grid)), dtype=t.float32)
        backcast_template = t.cat([backcast_cos, backcast_sin], dim=0)

        forecast_cos = t.tensor(np.transpose(np.cos(forecast_grid)), dtype=t.float32)
        forecast_sin = t.tensor(np.transpose(np.sin(forecast_grid)), dtype=t.float32)
        forecast_template = t.cat([forecast_cos, forecast_sin], dim=0)

        self.backcast_basis = nn.Parameter(backcast_template, requires_grad=False)
        self.forecast_basis = nn.Parameter(forecast_template, requires_grad=False)

    def forward(self, theta: t.Tensor, insample_x_t=None, outsample_x_t=None) -> Tuple[t.Tensor, t.Tensor]:
        cut_point = self.forecast_basis.shape[0]
        backcast = t.einsum("bp,pt->bt", theta[:, cut_point:], self.backcast_basis)
        forecast = t.einsum("bp,pt->bt", theta[:, :cut_point], self.forecast_basis)
        return backcast, forecast


# ══════════════════════════════════════════════════════════════════════════════
# N-BEATS Block + 主模型
# ══════════════════════════════════════════════════════════════════════════════


class _StaticFeaturesEncoder(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features=in_features, out_features=out_features),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.encoder(x)


class NBeatsBlock(nn.Module):
    """N-BEATS 基本块：FC 网络 → theta → Basis 函数 → (backcast, forecast)。

    backcast 用于从残差中减去已解释成分，forecast 累加到最终预测。
    """

    ACTIVATIONS = {
        "relu": nn.ReLU,
        "softplus": nn.Softplus,
        "tanh": nn.Tanh,
        "selu": nn.SELU,
        "lrelu": nn.LeakyReLU,
        "prelu": nn.PReLU,
        "sigmoid": nn.Sigmoid,
    }

    def __init__(
        self,
        x_t_n_inputs: int,
        x_s_n_inputs: int,
        x_s_n_hidden: int,
        theta_n_dim: int,
        basis: nn.Module,
        n_layers: int,
        theta_n_hidden: list,
        batch_normalization: bool,
        dropout_prob: float,
        activation: str,
    ):
        super().__init__()

        if x_s_n_inputs == 0:
            x_s_n_hidden = 0
        theta_n_hidden = [x_t_n_inputs + x_s_n_hidden] + theta_n_hidden

        self.x_s_n_inputs = x_s_n_inputs
        self.x_s_n_hidden = x_s_n_hidden

        hidden_layers = []
        for i in range(n_layers):
            hidden_layers.append(nn.Linear(theta_n_hidden[i], theta_n_hidden[i + 1]))
            hidden_layers.append(self.ACTIVATIONS[activation]())
            if batch_normalization:
                hidden_layers.append(nn.BatchNorm1d(theta_n_hidden[i + 1]))
            if dropout_prob > 0:
                hidden_layers.append(nn.Dropout(p=dropout_prob))

        output_layer = [nn.Linear(theta_n_hidden[-1], theta_n_dim)]
        self.layers = nn.Sequential(*(hidden_layers + output_layer))
        # 正交初始化，防止大输入维度下前向传播输出爆炸
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight)
                nn.init.zeros_(layer.bias)

        if (self.x_s_n_inputs > 0) and (self.x_s_n_hidden > 0):
            self.static_encoder = _StaticFeaturesEncoder(x_s_n_inputs, x_s_n_hidden)
        self.basis = basis

    def forward(self, insample_y: t.Tensor, insample_x_t=None,
                outsample_x_t=None, x_s=None) -> Tuple[t.Tensor, t.Tensor]:
        if (self.x_s_n_inputs > 0) and (self.x_s_n_hidden > 0):
            x_s = self.static_encoder(x_s)
            insample_y = t.cat((insample_y, x_s), 1)

        theta = self.layers(insample_y)
        backcast, forecast = self.basis(theta, insample_x_t, outsample_x_t)
        return backcast, forecast


class NBeats(nn.Module):
    """N-BEATS 主模型：堆叠多个 NBeatsBlock，逐块分解残差。"""

    def __init__(self, blocks: nn.ModuleList):
        super().__init__()
        self.blocks = blocks

    def forward(self, insample_y, insample_x_t, insample_mask,
                outsample_x_t, x_s, return_decomposition=False):
        residuals = insample_y.flip(dims=(-1,))
        if insample_x_t is not None:
            insample_x_t = insample_x_t.flip(dims=(-1,))
        insample_mask = insample_mask.flip(dims=(-1,))

        forecast = insample_y[:, -1:]  # Level with Naive1
        block_forecasts = []
        for block in self.blocks:
            backcast, block_forecast = block(
                insample_y=residuals, insample_x_t=insample_x_t,
                outsample_x_t=outsample_x_t, x_s=x_s,
            )
            residuals = (residuals - backcast) * insample_mask
            forecast = forecast + block_forecast
            block_forecasts.append(block_forecast)

        block_forecasts = t.stack(block_forecasts).permute(1, 0, 2)

        if return_decomposition:
            return forecast, block_forecasts
        return forecast


# ══════════════════════════════════════════════════════════════════════════════
# 适配器 —— Keras-like API
# ══════════════════════════════════════════════════════════════════════════════

BASIS_REGISTRY = {
    "identity": IdentityBasis,
    "trend": TrendBasis,
    "seasonality": SeasonalityBasis,
}


class NBeatsxAdapter:
    """N-BEATSx PyTorch 模型适配器，提供 Keras 风格的 fit()/predict() 接口。

    内部管理 PyTorch 模型、优化器、学习率调度和设备（CUDA/CPU），
    对外暴露与现有 Keras 模型兼容的训练/预测方法。
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        stack_types: list[str] | None = None,
        n_blocks: list[int] | None = None,
        n_layers: list[int] | None = None,
        n_hidden: list[list[int]] | None = None,
        n_harmonics: list[int] | None = None,
        n_polynomials: list[int] | None = None,
        activation: str = "relu",
        batch_normalization: bool = True,
        dropout_prob_theta: float = 0.1,
        shared_weights: bool = False,
        learning_rate: float = 0.001,
        lr_decay: float = 0.5,
        n_lr_decay_steps: int = 5,
        weight_decay: float = 0.0,
        random_seed: int = 42,
        device: str | None = None,
    ):
        if stack_types is None:
            stack_types = ["identity"]
        if n_blocks is None:
            n_blocks = [3]
        if n_layers is None:
            n_layers = [4]
        if n_hidden is None:
            n_hidden = [[256, 256, 256]]

        self.input_size = input_size
        self.output_size = output_size
        self.stack_types = stack_types
        self.n_blocks = n_blocks
        self.learning_rate = learning_rate
        self.lr_decay = lr_decay
        self.n_lr_decay_steps = n_lr_decay_steps
        self.weight_decay = weight_decay
        self.random_seed = random_seed

        if device is None:
            device = "cuda" if t.cuda.is_available() else "cpu"
        # 验证 CUDA 设备确实可用（新 GPU 架构如 Blackwell sm_120 可能不被旧 PyTorch 支持）
        if device == "cuda":
            try:
                _test = t.zeros(1, device="cuda")
                del _test
            except RuntimeError:
                logger.warning("CUDA 设备不可用（内核不兼容），回退到 CPU")
                device = "cpu"
        self.device = device

        t.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)

        # 构建 stack → NBeatsBlock 列表
        block_list = []
        # 用于 Trend/Seasonality 基函数的构造参数
        n_harmonics = n_harmonics or [1] * len(stack_types)
        n_polynomials = n_polynomials or [2] * len(stack_types)

        for i, stype in enumerate(stack_types):
            for block_id in range(n_blocks[i]):
                bn = batch_normalization and (len(block_list) == 0)
                if shared_weights and block_id > 0:
                    nbeats_block = block_list[-1]
                else:
                    if stype == "identity":
                        basis = IdentityBasis(backcast_size=input_size, forecast_size=output_size)
                        theta_dim = input_size + output_size
                    elif stype == "trend":
                        basis = TrendBasis(
                            degree_of_polynomial=n_polynomials[i],
                            backcast_size=input_size, forecast_size=output_size,
                        )
                        theta_dim = 2 * (n_polynomials[i] + 1)
                    elif stype == "seasonality":
                        basis = SeasonalityBasis(
                            harmonics=n_harmonics[i],
                            backcast_size=input_size, forecast_size=output_size,
                        )
                        theta_dim = 4 * int(np.ceil(n_harmonics[i] / 2 * output_size) - (n_harmonics[i] - 1))
                    else:
                        raise ValueError(f"不支持的 stack 类型: {stype}，可选: identity/trend/seasonality")

                    nbeats_block = NBeatsBlock(
                        x_t_n_inputs=input_size,
                        x_s_n_inputs=0,
                        x_s_n_hidden=0,
                        theta_n_dim=theta_dim,
                        basis=basis,
                        n_layers=n_layers[i],
                        theta_n_hidden=n_hidden[i],
                        batch_normalization=bn,
                        dropout_prob=dropout_prob_theta,
                        activation=activation,
                    )
                block_list.append(nbeats_block)

        self.model = NBeats(nn.ModuleList(block_list)).to(self.device)
        logger.info(
            "已构建 N-BEATSx 模型: input=%d output=%d stacks=%s blocks=%s device=%s",
            input_size, output_size, stack_types, n_blocks, self.device,
        )

    def _to_tensor(self, x: np.ndarray) -> t.Tensor:
        return t.as_tensor(x, dtype=t.float32).to(self.device)

    def fit(
        self,
        train_X: np.ndarray,
        train_y: np.ndarray,
        validation_data=None,
        epochs: int = 30,
        batch_size: int = 50,
        verbose: int = 2,
        shuffle: bool = False,
    ):
        """训练模型。

        参数
        ----------
        train_X : (n_train, input_size, 1) — 输入序列。
        train_y : (n_train, output_size) — 目标值。
        validation_data : (val_X, val_y) 或 None。
        epochs : 训练轮数。
        batch_size : 批量大小。
        verbose : 0=静默，1=进度条，2=每 epoch 一行。
        """
        # 数据整形：X (n, look_back, 1) → insample_y (n, look_back)
        X_t = self._to_tensor(train_X).squeeze(-1)
        y_t = self._to_tensor(train_y)
        n_train = X_t.shape[0]

        val_X_t, val_y_t = None, None
        if validation_data is not None:
            val_X_t = self._to_tensor(validation_data[0]).squeeze(-1)
            val_y_t = self._to_tensor(validation_data[1])

        optimizer = t.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        lr_decay_steps = max(1, epochs // self.n_lr_decay_steps)
        scheduler = t.optim.lr_scheduler.StepLR(optimizer, step_size=lr_decay_steps, gamma=self.lr_decay)

        history = {"loss": [], "val_loss": []}
        best_val_loss = float("inf")
        best_state_dict = None
        early_stopping_counter = 0
        early_stopping_patience = max(5, epochs // 5)

        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0.0
            n_batches = 0

            # 手动分批（不使用 DataLoader，保持简单）
            indices = np.random.permutation(n_train) if batch_size < n_train else np.arange(n_train)
            for start in range(0, n_train, batch_size):
                batch_idx = indices[start : start + batch_size]
                batch_X = X_t[batch_idx]
                batch_y = y_t[batch_idx]

                insample_mask = t.ones_like(batch_X, device=self.device)

                optimizer.zero_grad()
                forecast = self.model(
                    insample_y=batch_X,
                    insample_x_t=None,
                    insample_mask=insample_mask,
                    outsample_x_t=None,
                    x_s=None,
                )
                loss = F.mse_loss(forecast, batch_y)
                if not t.isnan(loss):
                    loss.backward()
                    t.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            scheduler.step()
            avg_loss = epoch_loss / max(n_batches, 1)
            history["loss"].append(avg_loss)

            # 验证
            if val_X_t is not None:
                self.model.eval()
                with t.no_grad():
                    val_mask = t.ones_like(val_X_t, device=self.device)
                    val_pred = self.model(
                        insample_y=val_X_t,
                        insample_x_t=None,
                        insample_mask=val_mask,
                        outsample_x_t=None,
                        x_s=None,
                    )
                    val_loss = F.mse_loss(val_pred, val_y_t).item()
                    history["val_loss"].append(val_loss)
                self.model.train()

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state_dict = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                    early_stopping_counter = 0
                else:
                    early_stopping_counter += 1

                if early_stopping_counter >= early_stopping_patience:
                    if verbose >= 2:
                        logger.info("  Early stopping at epoch %d (best val_loss=%.6f)", epoch + 1, best_val_loss)
                    break

            if verbose >= 2:
                val_str = f" - val_loss: {history['val_loss'][-1]:.6f}" if history["val_loss"] else ""
                print(f"Epoch {epoch + 1}/{epochs} - loss: {avg_loss:.6f}{val_str}")

        # 恢复最佳模型
        if best_state_dict is not None:
            self.model.load_state_dict({k: v.to(self.device) for k, v in best_state_dict.items()})

        return history

    def predict(self, X: np.ndarray, verbose: int = 0) -> np.ndarray:
        """预测。

        参数
        ----------
        X : (n, input_size, 1) — 输入序列。
        verbose : 未使用（保持与 Keras API 兼容）。

        返回
        -------
        np.ndarray : (n, output_size) — 预测值。
        """
        X_t = self._to_tensor(X).squeeze(-1)
        self.model.eval()
        with t.no_grad():
            mask = t.ones_like(X_t, device=self.device)
            pred = self.model(
                insample_y=X_t,
                insample_x_t=None,
                insample_mask=mask,
                outsample_x_t=None,
                x_s=None,
            )
        pred = pred.cpu().numpy()
        # 裁剪极端值防止 Stage 2 BPNN 出现 NaN（数据经 MinMaxScaler 归一化至 [-1,1]）
        pred = np.clip(pred, -5.0, 5.0)
        return pred


def build_nbeatsx_model(
    input_shape: tuple[int, int],
    output_size: int = 1,
    stack_types: list[str] | None = None,
    n_blocks: list[int] | None = None,
    n_layers: list[int] | None = None,
    n_hidden: list[list[int]] | None = None,
    activation: str = "relu",
    batch_normalization: bool = True,
    dropout_prob_theta: float = 0.1,
    shared_weights: bool = False,
    learning_rate: float = 0.001,
    lr_decay: float = 0.5,
    n_lr_decay_steps: int = 5,
    weight_decay: float = 0.0,
    random_seed: int = 42,
    device: str | None = None,
) -> NBeatsxAdapter:
    """构建 N-BEATSx 模型适配器（与 build_lstm_model 等保持一致的工厂函数）。

    参数
    ----------
    input_shape : (时间步数, 特征数) — 与 Keras 模型一致的输入形状，
                  其中时间步数 = look_back。
    output_size : 预测步数，默认 1（单步预测）。
    stack_types : 堆叠类型列表，如 ["identity"] 或 ["trend", "seasonality"]。
    learning_rate : Adam 优化器的初始学习率。
    其余参数见 NBeatsxAdapter.__init__。

    返回
    -------
    NBeatsxAdapter : 已实例化的适配器（模型已构建并移至设备）。
    """
    input_size = input_shape[0]  # 时间步数
    return NBeatsxAdapter(
        input_size=input_size,
        output_size=output_size,
        stack_types=stack_types,
        n_blocks=n_blocks,
        n_layers=n_layers,
        n_hidden=n_hidden,
        activation=activation,
        batch_normalization=batch_normalization,
        dropout_prob_theta=dropout_prob_theta,
        shared_weights=shared_weights,
        learning_rate=learning_rate,
        lr_decay=lr_decay,
        n_lr_decay_steps=n_lr_decay_steps,
        weight_decay=weight_decay,
        random_seed=random_seed,
        device=device,
    )
