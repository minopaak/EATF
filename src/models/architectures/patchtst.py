"""PatchTST (Nie et al., ICLR 2023).

시계열을 패치로 나눠 채널 독립적으로 transformer 인코딩하는 모델.
clones/Time-Series-Library/models/PatchTST.py 의 forecasting 경로를
self-contained하게 옮겨와 정리. 입력은 instance-wise 정규화(Non-stationary
Transformer 방식) 후 인코딩, 출력에서 역정규화.

forward: (x_enc, x_mark_enc, x_dec, x_mark_dec, mask) -> [B, pred_len, D]
"""
import torch
import torch.nn as nn

from ..config import ModelConfig
from ..layers import (
    PatchEmbedding,
    FullAttention,
    AttentionLayer,
    Encoder,
    EncoderLayer,
    FlattenHead,
    Transpose,
)


class PatchTST(nn.Module):
    def __init__(self, configs: ModelConfig, patch_len: int = None, stride: int = None):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        # patch_len/stride는 configs에서 (짧은 look-back L=8에 맞게 작게)
        patch_len = patch_len if patch_len is not None else getattr(configs, "patch_len", 16)
        stride = stride if stride is not None else getattr(configs, "stride", 8)
        padding = stride

        # 패치 분할 + 임베딩
        self.patch_embedding = PatchEmbedding(
            configs.d_model, patch_len, stride, padding, configs.dropout
        )

        # transformer 인코더
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            False, configs.factor,
                            attention_dropout=configs.dropout, output_attention=False,
                        ),
                        configs.d_model, configs.n_heads,
                    ),
                    configs.d_model, configs.d_ff,
                    dropout=configs.dropout, activation=configs.activation,
                )
                for _ in range(configs.e_layers)
            ],
            norm_layer=nn.Sequential(
                Transpose(1, 2), nn.BatchNorm1d(configs.d_model), Transpose(1, 2)
            ),
        )

        # 예측 헤드
        self.head_nf = configs.d_model * int((configs.seq_len - patch_len) / stride + 2)
        self.head = FlattenHead(
            configs.enc_in, self.head_nf, configs.pred_len, head_dropout=configs.dropout
        )

    def forecast(self, x_enc):
        # instance-wise 정규화 (Non-stationary Transformer)
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc / stdev

        # 패치 + 임베딩 : [bs, n_vars, seq_len] -> [bs*n_vars, patch_num, d_model]
        x_enc = x_enc.permute(0, 2, 1)
        enc_out, n_vars = self.patch_embedding(x_enc)

        # 인코더
        enc_out, _ = self.encoder(enc_out)
        # [bs, n_vars, patch_num, d_model] -> [bs, n_vars, d_model, patch_num]
        enc_out = torch.reshape(enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        enc_out = enc_out.permute(0, 1, 3, 2)

        # 헤드
        dec_out = self.head(enc_out)        # [bs, n_vars, pred_len]
        dec_out = dec_out.permute(0, 2, 1)  # [bs, pred_len, n_vars]

        # 역정규화
        dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
        dec_out = dec_out + means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
        return dec_out

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        dec_out = self.forecast(x_enc)
        return dec_out[:, -self.pred_len:, :]  # [B, pred_len, D]
