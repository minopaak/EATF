"""DLinear (Zeng et al., AAAI 2023).

시계열을 추세/계절성으로 분해하고 각각 선형 레이어로 예측하는 단순 강력 베이스라인.
clones/Time-Series-Library/models/DLinear.py 의 forecasting 경로를
self-contained하게 옮겨와 정리.

forward: (x_enc, x_mark_enc, x_dec, x_mark_dec, mask) -> [B, pred_len, D]
         (시간 마크/디코더 인자는 미사용, 인터페이스 통일용)
"""
import torch
import torch.nn as nn

from ..config import ModelConfig
from ..layers import series_decomp


class DLinear(nn.Module):
    def __init__(self, configs: ModelConfig, individual: bool = False):
        """individual: 변수마다 별도 선형층을 둘지 여부 (기본 공유)."""
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.channels = configs.enc_in
        self.individual = individual

        self.decompsition = series_decomp(configs.moving_avg)

        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()
            for _ in range(self.channels):
                self.Linear_Seasonal.append(self._init_linear())
                self.Linear_Trend.append(self._init_linear())
        else:
            self.Linear_Seasonal = self._init_linear()
            self.Linear_Trend = self._init_linear()

    def _init_linear(self) -> nn.Linear:
        layer = nn.Linear(self.seq_len, self.pred_len)
        # 균등 초기화 (원 논문/구현 관행): 초기엔 단순 평균에 가깝게
        layer.weight = nn.Parameter((1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
        return layer

    def encoder(self, x):
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init = seasonal_init.permute(0, 2, 1)
        trend_init = trend_init.permute(0, 2, 1)

        if self.individual:
            seasonal_output = torch.zeros(
                [seasonal_init.size(0), seasonal_init.size(1), self.pred_len],
                dtype=seasonal_init.dtype, device=seasonal_init.device,
            )
            trend_output = torch.zeros(
                [trend_init.size(0), trend_init.size(1), self.pred_len],
                dtype=trend_init.dtype, device=trend_init.device,
            )
            for i in range(self.channels):
                seasonal_output[:, i, :] = self.Linear_Seasonal[i](seasonal_init[:, i, :])
                trend_output[:, i, :] = self.Linear_Trend[i](trend_init[:, i, :])
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)

        x = seasonal_output + trend_output
        return x.permute(0, 2, 1)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        dec_out = self.encoder(x_enc)
        return dec_out[:, -self.pred_len:, :]  # [B, pred_len, D]
