"""학습용 데이터 로더 (data/processed/*.csv → 윈도우 텐서).

build_dataset.py(데이터 구축)와 구분: 이 파일은 구축된 CSV를 모델 학습용으로
로딩한다.

핵심 설계
---------
1. 슬라이딩 윈도우: (x[L,V], y[H,V]). L=look-back, H=horizon.
2. Split = TSLib `Dataset_Custom` 표준 방식 (PatchTST/DLinear/MM-TSFlib 동일):
   train/val/test = 70/10/20, val/test는 직전 구간에서 seq_len 만큼 look-back을 빌림.
       border1s = [0, num_train - L, n - num_test - L]
       border2s = [num_train, num_train + num_vali, n]
   → 예측 타깃은 각 구간 안에만(누수 없음), 입력만 과거 참조. 작은 도메인도 윈도우 0 안 됨.
3. 정규화 (normalize):
   - 'dataset' (기본): per-domain global StandardScaler. 각 도메인의 **train 행에만 fit**한
     평균/분산으로 그 도메인 전체를 표준화. MM-TSFlib(TSLib) 표준 방식과 동일.
     이벤트(레짐 변화)를 도메인 기준 편차로 보존.
   - 'instance' (옵션/ablation): RevIN. 각 윈도우를 look-back 통계로 정규화.
   - 'none': 정규화 안 함.
   어느 모드든 (x, y, mean, std)를 반환 → 예측을 원 스케일로 역정규화 가능.
4. 변수: 단변량(OT) 기본, 다변량 옵션.

빌더
----
  build_in_domain(domain, ...) : 한 도메인 내 시간순 train/val/test

모델 forward 규약: model(x, None, None, None) -> [B, H, V]
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed"

MONTHLY_DOMAINS = ["Agriculture", "Economy", "Security", "SocialGood", "Traffic"]
META_COLS = ["date", "start_date", "end_date"]
TEXT_COLS = ["report_text", "search_text", "report_pred", "search_pred"]

DEFAULT_SEQ_LEN = 8           # Time-MMD monthly 설정
DEFAULT_NORMALIZE = "dataset"  # per-domain global StandardScaler
_EPS = 1e-5


# ─────────────────────────────────────────────────────────────
#  로딩 / 변수 선택
# ─────────────────────────────────────────────────────────────
def load_domain_frame(domain: str, data_dir: Path = DATA_DIR) -> pd.DataFrame:
    path = Path(data_dir) / f"{domain}_merged.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} 없음. build_dataset.py 먼저 실행 필요.")
    return pd.read_csv(path, parse_dates=["date"])


def get_var_columns(df: pd.DataFrame, multivariate: bool):
    """numerical 변수 컬럼과 OT 인덱스 반환."""
    all_vars = [c for c in df.columns if c not in META_COLS + TEXT_COLS]
    cols = all_vars if multivariate else ["OT"]
    if "OT" not in cols:
        raise ValueError(f"OT 컬럼 없음. 사용 가능: {all_vars}")
    return cols, cols.index("OT")


# ─────────────────────────────────────────────────────────────
#  Window Dataset
# ─────────────────────────────────────────────────────────────
class WindowDataset(Dataset):
    """[T, V] 시계열을 (x[L,V], y[H,V]) 윈도우로 제공.

    normalize:
      'dataset'  : 주어진 scaler(mean,std; 도메인 train 통계)로 표준화 (상수)
      'instance' : 각 윈도우의 look-back 통계로 표준화 (RevIN)
      'none'     : mean=0, std=1
    모든 모드에서 (x, y, mean, std)를 반환.
    """

    def __init__(self, series: np.ndarray, seq_len: int, pred_len: int,
                 normalize: str = DEFAULT_NORMALIZE, scaler=None, eps: float = _EPS,
                 text_emb: np.ndarray = None):
        assert normalize in ("dataset", "instance", "none")
        if normalize == "dataset" and scaler is None:
            raise ValueError("normalize='dataset'은 scaler=(mean,std)가 필요합니다.")
        self.series = np.asarray(series, dtype=np.float32)
        # text_emb[t] = 행 t(달)의 텍스트 임베딩. 윈도우는 look-back 마지막 달(예측
        # 시점, s+L-1)의 임베딩을 사용 (MM-TSFlib 의 forecast-origin 텍스트와 동일).
        self.text_emb = None if text_emb is None else np.asarray(text_emb, dtype=np.float32)
        self.L = seq_len
        self.H = pred_len
        self.normalize = normalize
        self.scaler = scaler
        self.eps = eps
        T = len(self.series)
        self.starts = list(range(0, T - seq_len - pred_len + 1))

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, i):
        s = self.starts[i]
        x = self.series[s : s + self.L]                      # [L, V]
        y = self.series[s + self.L : s + self.L + self.H]    # [H, V]

        if self.normalize == "instance":
            mean = x.mean(axis=0)
            std = x.std(axis=0) + self.eps
        elif self.normalize == "dataset":
            mean, std = self.scaler                          # 도메인 상수 (broadcast)
        else:
            mean = np.zeros(x.shape[1], dtype=np.float32)
            std = np.ones(x.shape[1], dtype=np.float32)

        x_n = (x - mean) / std
        y_n = (y - mean) / std

        if self.text_emb is not None:
            te = self.text_emb[s + self.L - 1]               # 예측 시점 달의 텍스트
        else:
            te = np.zeros(1, dtype=np.float32)               # unimodal 더미 (모델이 무시)

        return (
            torch.from_numpy(np.asarray(x_n, dtype=np.float32)),
            torch.from_numpy(np.asarray(y_n, dtype=np.float32)),
            torch.from_numpy(np.asarray(mean, dtype=np.float32)),
            torch.from_numpy(np.asarray(std, dtype=np.float32)),
            torch.from_numpy(np.asarray(te, dtype=np.float32)),
        )


def _make_ds(series, seq_len, pred_len, normalize, scaler=None, tag="", text_emb=None):
    ds = WindowDataset(series, seq_len, pred_len, normalize=normalize, scaler=scaler,
                       text_emb=text_emb)
    if len(ds) == 0:
        print(f"  [warn] {tag}: 구간 길이 {len(series)} 로 윈도우 0개 (L+H={seq_len + pred_len})")
    return ds


def _trim_to_valid(arr: np.ndarray, tag: str = "", aux: np.ndarray = None):
    """선택된 변수들이 모두 유효한 연속 구간으로 자름 (앞/뒤 ragged NaN 제거).

    단변량(OT)은 보통 그대로 = 풀시리즈. 다변량은 늦게 시작하는 변수에 맞춰
    공통 윈도우가 됨. 내부 NaN이 남으면 경고.

    aux(예: 행별 텍스트 임베딩 [T, d])를 주면 같은 구간으로 잘라 함께 반환한다.
    aux=None: trimmed 단일 반환. aux 제공: (trimmed, aux_trimmed) 튜플 반환.
    """
    valid = ~np.isnan(arr).any(axis=1)
    if not valid.any():
        raise ValueError(f"{tag}: 전 구간 NaN")
    first = int(valid.argmax())
    last = len(valid) - 1 - int(valid[::-1].argmax())
    trimmed = arr[first:last + 1]
    n_drop = len(arr) - len(trimmed)
    internal = int((~valid[first:last + 1]).sum())
    if n_drop > 0:
        print(f"  [trim] {tag}: ragged 구간 {n_drop} rows 제거 (유효 {len(trimmed)})")
    if internal > 0:
        print(f"  [warn] {tag}: 내부 NaN {internal} rows 잔존 (impute 필요)")
    if aux is not None:
        return trimmed, aux[first:last + 1]
    return trimmed


def _standard_split(series: np.ndarray, seq_len: int, pred_len: int, normalize: str,
                    train_ratio=0.7, test_ratio=0.2, tag="", text_emb=None) -> dict:
    """TSLib Dataset_Custom 방식 train/val/test 분할 + (dataset 모드면) train-fit scaler.
    val/test는 직전 구간에서 seq_len 만큼 look-back을 빌려온다.
    text_emb([T, d])를 주면 시리즈와 동일 border로 잘라 각 WindowDataset에 부착.
    """
    n = len(series)
    num_train = int(n * train_ratio)
    num_test = int(n * test_ratio)
    num_vali = n - num_train - num_test

    border1s = [0, max(0, num_train - seq_len), max(0, n - num_test - seq_len)]
    border2s = [num_train, num_train + num_vali, n]

    # per-domain global scaler는 train 행([0:num_train])에만 fit
    scaler = None
    if normalize == "dataset":
        train_rows = series[0:num_train]
        mean = train_rows.mean(axis=0).astype(np.float32)
        std = (train_rows.std(axis=0) + _EPS).astype(np.float32)
        scaler = (mean, std)

    out = {"scaler": scaler}
    for t, name in enumerate(["train", "val", "test"]):
        seg = series[border1s[t]:border2s[t]]
        seg_emb = text_emb[border1s[t]:border2s[t]] if text_emb is not None else None
        out[name] = _make_ds(seg, seq_len, pred_len, normalize, scaler, f"{tag}/{name}",
                             text_emb=seg_emb)
    return out


# ─────────────────────────────────────────────────────────────
#  빌더
# ─────────────────────────────────────────────────────────────
def build_in_domain(domain: str, seq_len: int = DEFAULT_SEQ_LEN, pred_len: int = 12,
                    multivariate: bool = False, normalize: str = DEFAULT_NORMALIZE,
                    data_dir: Path = DATA_DIR, text: bool = False,
                    llm: str = "BERT", text_source: str = "both", text_pool: str = "avg",
                    text_layer: str = "embedding", device: str = "cpu") -> dict:
    """한 도메인 내 시간순 train/val/test (source==target).

    text=True 면 frozen LLM 텍스트 임베딩([T, d_llm])을 같은 trim/split로 부착해
    멀티모달(MM-TSFlib 계열) 학습에 쓴다. 임베딩은 도메인별 1회 계산 후 캐시.
    """
    df = load_domain_frame(domain, data_dir)
    cols, target_idx = get_var_columns(df, multivariate)
    arr = df[cols].to_numpy()
    emb = None
    if text:
        from ..models.text_encoder import encode_domain
        emb = encode_domain(domain, df, llm=llm, source=text_source,
                            pool=text_pool, layer=text_layer, device=device)
    if emb is not None:
        series, emb = _trim_to_valid(arr, tag=domain, aux=emb)
    else:
        series = _trim_to_valid(arr, tag=domain)              # 단변량=풀, 다변량=공통윈도우
    sp = _standard_split(series, seq_len, pred_len, normalize, tag=domain, text_emb=emb)
    return {
        "train": sp["train"], "val": sp["val"], "test": sp["test"],
        "var_cols": cols, "target_idx": target_idx, "mode": "in_domain",
        "domain": domain, "scaler": sp["scaler"],
    }


def make_dataloader(ds, batch_size: int = 32, shuffle: bool = False,
                    num_workers: int = 0) -> DataLoader:
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, drop_last=False)


# ─────────────────────────────────────────────────────────────
#  smoke test
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import torch
    from ..models import build_model, ModelConfig

    L, H = DEFAULT_SEQ_LEN, 12
    print(f"L={L}, H={H}, normalize={DEFAULT_NORMALIZE}\n")

    print("=== in-domain 윈도우 수 (단변량) ===")
    splits = {}
    for d in MONTHLY_DOMAINS:
        s = build_in_domain(d, L, H)
        splits[d] = s
        print(f"  {d:12s} train={len(s['train']):4d} val={len(s['val']):3d} test={len(s['test']):3d}")

    print("\n=== end-to-end ===")
    loader = make_dataloader(splits["Economy"]["train"], batch_size=16, shuffle=True)
    xb, yb, mb, sb, te = next(iter(loader))
    cfg = ModelConfig(seq_len=L, pred_len=H, enc_in=1, c_out=1)
    for name in ("PatchTST", "DLinear"):
        model = build_model(name, cfg); model.eval()
        with torch.no_grad():
            pred = model(xb, None, None, None)
        print(f"  {name:9s} x={tuple(xb.shape)} pred={tuple(pred.shape)}")
