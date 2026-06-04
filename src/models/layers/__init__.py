"""공용 빌딩블록 (모델 아키텍처가 공유).

대부분 Time-Series-Library(thuml/Time-Series-Library; MM-TSFlib 경유)에서 이식한
self-contained 모듈들이며, 각 파일 상단에 출처를 표기한다. 모듈 구성:

  embed              DataEmbedding 계열, TokenEmbedding, PositionalEmbedding, PatchEmbedding
  attention          FullAttention, ProbAttention, AttentionLayer
  encdec             (표준 Transformer) Encoder/Decoder/EncoderLayer/DecoderLayer/ConvLayer
  autoformer_encdec  (Autoformer) Encoder/Decoder/EncoderLayer/DecoderLayer/my_Layernorm/series_decomp
  autocorrelation    AutoCorrelation, AutoCorrelationLayer
  fourier / wavelet  FEDformer 주파수 블록
  masking / norm     attention 마스크, StandardNorm
  heads              FlattenHead, Transpose (PatchTST 헤드)

아래 re-export 는 compact 모델(DLinear/PatchTST/iTransformer)이 쓰는 공통 심볼이다.
이름 충돌(Encoder 등 표준 vs Autoformer)은 재노출하지 않으므로, enc-dec 모델은 해당
서브모듈에서 직접 import 한다.
"""
from .attention import FullAttention, AttentionLayer
from .encdec import Encoder, EncoderLayer
from .embed import PatchEmbedding, PositionalEmbedding
from .autoformer_encdec import series_decomp, moving_avg
from .heads import FlattenHead, Transpose

__all__ = [
    "FullAttention", "AttentionLayer", "Encoder", "EncoderLayer",
    "PatchEmbedding", "PositionalEmbedding", "series_decomp", "moving_avg",
    "FlattenHead", "Transpose",
]
