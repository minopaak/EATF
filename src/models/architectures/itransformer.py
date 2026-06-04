"""iTransformer (Liu et al., ICLR 2024).

'변수(variate)를 토큰으로' 뒤집어, 각 변수의 전체 look-back 시계열을 하나의 토큰으로
임베딩하고 변수 간 self-attention으로 상관을 모델링. clones/Time-Series-Library 의
iTransformer forecasting 경로를 self-contained하게 이식 (시간 마크 임베딩은 우리 데이터에
없으므로 생략, 나머지는 동일). 입력은 instance-wise 정규화 후 인코딩, 출력에서 역정규화.

forward: (x_enc, x_mark_enc, x_dec, x_mark_dec, mask) -> [B, pred_len, D]
"""
import torch
import torch.nn as nn

from ..config import ModelConfig
from ..layers import FullAttention, AttentionLayer, Encoder, EncoderLayer


class iTransformer(nn.Module):
    def __init__(self, configs: ModelConfig):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        # inverted embedding: 변수별 [seq_len] 시계열 -> [d_model] 토큰
        self.enc_embedding = nn.Linear(configs.seq_len, configs.d_model)
        self.dropout = nn.Dropout(configs.dropout)

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor,
                                      attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads,
                    ),
                    configs.d_model, configs.d_ff,
                    dropout=configs.dropout, activation=configs.activation,
                )
                for _ in range(configs.e_layers)
            ],
            norm_layer=nn.LayerNorm(configs.d_model),
        )
        self.projection = nn.Linear(configs.d_model, configs.pred_len)

    def forecast(self, x_enc):
        # instance-wise 정규화 (Non-stationary Transformer)
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc / stdev

        # invert: [B, L, V] -> [B, V, L] -> embed -> [B, V, d_model]
        enc_out = self.dropout(self.enc_embedding(x_enc.permute(0, 2, 1)))
        enc_out, _ = self.encoder(enc_out)                       # [B, V, d_model]
        dec_out = self.projection(enc_out).permute(0, 2, 1)      # [B, pred_len, V]

        # 역정규화
        dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
        dec_out = dec_out + means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
        return dec_out

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        return self.forecast(x_enc)[:, -self.pred_len:, :]       # [B, pred_len, D]
