"""OpenAI 유틸 (임베딩 + 생성). knowledge_pool과 같은 계정/모델을 쓴다.

- 임베딩: text-embedding-3-small (단위벡터 → 코사인 = 내적).
- 생성:  responses.create(instructions, input). 모델은 호출부에서 지정.
- 키: OPENAI_API_KEY 환경변수 → 없으면 EATF/.env → sibling knowledge_pool/.env 순으로 로드.
"""
import os
from functools import lru_cache
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _ensure_key():
    if os.environ.get("OPENAI_API_KEY"):
        return
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
    if not os.environ.get("OPENAI_API_KEY"):
        load_dotenv(PROJECT_ROOT.parent / "knowledge_pool" / ".env")
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY 없음 — 환경변수나 .env 설정 필요")


@lru_cache(maxsize=1)
def _client():
    _ensure_key()
    from openai import OpenAI
    return OpenAI()


def embed_texts(texts, model="text-embedding-3-small", batch_size=256) -> np.ndarray:
    """문자열 리스트 -> [N, d] 임베딩(단위벡터). 입력 순서 보존."""
    out = []
    for i in range(0, len(texts), batch_size):
        resp = _client().embeddings.create(model=model, input=list(texts[i:i + batch_size]))
        out += [d.embedding for d in sorted(resp.data, key=lambda x: x.index)]
    return np.asarray(out, dtype=np.float32)


def call_llm(instructions, input, model="gpt-4o-mini") -> str:
    """responses.create 단발 호출. instructions=시스템, input=사용자. 텍스트 반환."""
    resp = _client().responses.create(model=model, instructions=instructions, input=input)
    return resp.output_text
