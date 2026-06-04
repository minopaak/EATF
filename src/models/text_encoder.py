"""frozen LLM 텍스트 인코더 + 디스크 캐시.

data/processed/*.csv 의 텍스트 컬럼(report_text/search_text)을 frozen LLM으로
임베딩해 [T, d_llm] 벡터를 만든다. LLM은 frozen이고 데이터가 작으므로 **도메인별로
한 번만** 임베딩해 .npy 로 캐시한다 (학습 루프에서 재계산하지 않음).

MM-TSFlib 계열 fusion의 '무거운 frozen 부분'을 여기서 끝내고, 학습 가능한 투영
(d_llm -> d_text -> H*V)은 모델(mm_fusion) 쪽에 둔다.

풀링은 mask-aware (avg/max). 텍스트는 look-back 마지막 달(예측 시점) 것을 모델이
고르므로(loader), 여기서는 행별 임베딩 [T, d_llm] 전체를 만든다.
"""
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = PROJECT_ROOT / "data" / "text_emb"

_HF_NAME = {"BERT": "bert-base-uncased", "GPT2": "gpt2"}
_LAZY = {}   # llm -> (model, tokenizer), 프로세스당 1회 로드


def build_text_series(df, source: str = "both") -> list:
    """df의 텍스트 컬럼 -> 행별 문자열 [T]. 빈 칸은 'No information'."""
    rep = df["report_text"] if "report_text" in df.columns else None
    sea = df["search_text"] if "search_text" in df.columns else None
    out = []
    for i in range(len(df)):
        parts = []
        if source in ("report", "both") and rep is not None:
            parts.append(str(rep.iloc[i]))
        if source in ("search", "both") and sea is not None:
            parts.append(str(sea.iloc[i]))
        txt = "\n".join(p for p in parts if p and p.lower() != "nan").strip()
        out.append(txt or "No information")
    return out


def _load_llm(llm: str, device: str):
    if llm in _LAZY:
        return _LAZY[llm]
    from transformers import AutoModel, AutoTokenizer
    name = _HF_NAME[llm]
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:           # GPT2 는 pad 토큰이 없음
        tok.pad_token = tok.eos_token
    model = AutoModel.from_pretrained(name).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    _LAZY[llm] = (model, tok)
    return model, tok


@torch.no_grad()
def _embed(texts, llm, pool, device, layer="embedding", batch_size=32, max_length=256) -> np.ndarray:
    """layer='embedding': LLM 입력 임베딩 레이어만 사용(MM-TSFlib use_fullmodel=0, 기본).
       layer='hidden'   : 전체 forward 의 last_hidden_state (contextual)."""
    model, tok = _load_llm(llm, device)
    embed_layer = model.get_input_embeddings()
    chunks = []
    for i in range(0, len(texts), batch_size):
        enc = tok(texts[i:i + batch_size], return_tensors="pt", padding=True,
                  truncation=True, max_length=max_length).to(device)
        if layer == "hidden":
            hs = model(**enc).last_hidden_state                 # [B, T, d] (contextual)
        else:
            hs = embed_layer(enc["input_ids"])                  # [B, T, d] (embedding layer)
        mask = enc["attention_mask"].unsqueeze(-1).float()      # [B, T, 1]
        if pool == "max":
            v = hs.masked_fill(mask == 0, float("-inf")).max(dim=1).values
        else:  # avg (mask-aware)
            v = (hs * mask).sum(1) / mask.sum(1).clamp(min=1e-6)
        chunks.append(v.cpu().float())
    return torch.cat(chunks).numpy()


def encode_domain(domain, df, *, llm="BERT", source="both", pool="avg",
                  layer="embedding", device="cpu", cache_dir=CACHE_DIR) -> np.ndarray:
    """도메인 텍스트 -> [T, d_llm]. 캐시 있으면 로드, 없으면 임베딩 후 저장.
    layer: 'embedding'(MM-TSFlib 기본) | 'hidden'(contextual)."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{domain}_{llm}_{source}_{pool}_{layer}.npy"
    if path.exists():
        emb = np.load(path)
        if len(emb) == len(df):
            return emb                                          # 행 수 일치 → 재사용
    emb = _embed(build_text_series(df, source), llm, pool, device, layer=layer)
    np.save(path, emb)
    return emb
