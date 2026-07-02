"""Mechanism pool retrieval — DK pool JSON 로드 + content 임베딩 + top-K 검색.

knowledge_pool(별도 repo)이 만든 DK(Domain Knowledge) JSON을 **소비만** 한다. 각 entry의
자연어 `content`(시점무관 일반 원리·메커니즘)를 text-embedding-3-small로 임베딩해두고,
query와의 코사인 유사도로 top-K를 검색한다 (pool 임베딩은 도메인별 1회 캐시).

query는 이후 query writer가 생성한다 — 여기선 임의 query 문자열을 받는다. (`04_evaluation.md`)

실행(스모크): EATF/ 에서
    uv run python -m src.retrieval
"""
import json
from pathlib import Path

import numpy as np

from .llm import embed_texts

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POOL_DIR = PROJECT_ROOT.parent / "knowledge_pool" / "data"   # sibling repo
CACHE_DIR = PROJECT_ROOT / "data" / "pool_emb"

# EATF 도메인명 -> pool 파일명. 현재 economy만 존재; 나머지는 구축 시 파일명 확인 필요.
_POOL_FILE = {"Agriculture": "agriculture", "Economy": "economy", "Security": "security",
              "SocialGood": "social_good", "Traffic": "traffic"}


def _load_jsonl(path):
    if not path.exists():
        raise FileNotFoundError(f"pool 없음: {path}")
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _unit(v):
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-8)


class MechanismPool:
    """한 도메인의 DK entry를 로드·임베딩하고 query로 top-K를 검색한다.

    검색은 retrieve()에서 코사인 유사도(단위벡터 내적)로 수행한다.
    """

    def __init__(self, domain, pool_dir=DEFAULT_POOL_DIR, cache_dir=CACHE_DIR):
        fname = _POOL_FILE.get(domain, domain.lower())
        self.entries = _load_jsonl(Path(pool_dir) / "dk_pool" / f"{fname}.jsonl")
        self.domain = domain
        self.Z = _unit(self._pool_emb(fname, Path(cache_dir)))

    def _pool_emb(self, fname, cache_dir):
        """pool content 임베딩 [N, d]. 캐시(엔트리 수 일치)면 로드, 아니면 임베딩 후 저장."""
        cache_dir.mkdir(parents=True, exist_ok=True)
        path = cache_dir / f"{fname}_dk.npy"
        if path.exists():
            z = np.load(path)
            if len(z) == len(self.entries):
                return z
        z = embed_texts([e["content"] for e in self.entries])
        np.save(path, z)
        return z

    def __len__(self):
        return len(self.entries)

    def retrieve(self, query, k=5):
        """query 문자열에 가장 유사한 top-K entry(+score) 반환."""
        q = _unit(embed_texts([query])[0])
        sim = self.Z @ q
        idx = np.argsort(-sim)[:k]
        return [{**self.entries[i], "score": float(sim[i])} for i in idx]


if __name__ == "__main__":
    pool = MechanismPool("Economy")
    print(f"Economy DK pool: {len(pool)} entries\n")

    q = "supply chain disruption widening the trade deficit under low inventory"
    print(f"query: {q}\n=== top-5 ===")
    for r in pool.retrieve(q, k=5):
        print(f"  {r['score']:.3f}  {r['title']}")
