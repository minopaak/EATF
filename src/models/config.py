"""모델 하이퍼파라미터 설정.

TSLib(thuml/Time-Series-Library) 모델들은 `configs` 객체의 속성을 읽어
초기화된다. 이 dataclass가 그 인터페이스를 채운다. 필드별로 어떤 모델이
참조하는지 주석으로 표시.

설계 문서: MD_files/02_dataset_design.md (L=36, H={6,12,24})
"""
from dataclasses import dataclass


@dataclass
class ModelConfig:
    # ── task ───────────────────────────────────────────────
    task_name: str = "long_term_forecast"   # 모든 모델: forward 분기

    # ── sequence length ────────────────────────────────────
    seq_len: int = 8        # L (look-back) — Time-MMD monthly 설정
    label_len: int = 4      # decoder warm-up (PatchTST/DLinear 미사용, MM-TSFlib용)
    pred_len: int = 12      # H (horizon) — monthly 실험은 {6,8,10,12}로 override

    # ── data dimensions ────────────────────────────────────
    enc_in: int = 1         # 입력 변수 수 (OT 단변량=1, Agri/Econ 다변량=3)
    dec_in: int = 1
    c_out: int = 1

    # ── transformer hyperparams (PatchTST 등) ──────────────
    d_model: int = 128
    n_heads: int = 8
    e_layers: int = 2
    d_layers: int = 1
    d_ff: int = 256
    factor: int = 1         # attention factor
    dropout: float = 0.1
    activation: str = "gelu"

    # ── PatchTST patch 설정 (짧은 L=8에 맞춰 작게) ─────────
    patch_len: int = 4
    stride: int = 2

    # ── DLinear ────────────────────────────────────────────
    moving_avg: int = 25    # series decomposition kernel (홀수)

    # ── embedding (일부 모델 참조) ─────────────────────────
    embed: str = "timeF"
    freq: str = "m"         # monthly

    # ── classification (forecast엔 미사용, 생성자가 참조) ──
    num_class: int = 1

    def for_horizon(self, pred_len: int) -> "ModelConfig":
        """동일 설정에서 horizon만 바꾼 복사본 반환."""
        from dataclasses import replace
        return replace(self, pred_len=pred_len)
