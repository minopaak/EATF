"""State → (summary+query) → retrieve → reasoning. 메소드 §7–11의 생성 파트.

한 예측 인스턴스에 대해:
  ① write_query : 수치+텍스트 state를 mechanism-seeking query로 (gpt-4o-mini, §21).
                  state summary와 query를 한 번에 쓴다.
  ② retrieve    : query로 DK pool top-K 검색 (retrieval.MechanismPool).
  ③ reason      : (state + 검색된 메커니즘) → 최종 영향분석 reasoning r_t (gpt-4.1, §11).

outer-loop query 적응(§18)은 아직 없음 — 초기 프롬프트 고정.

실행(스모크): EATF/ 에서
    uv run python -m src.mechanism
"""
import argparse
import json
from pathlib import Path

from .llm import call_llm
from .retrieval import MechanismPool, PROJECT_ROOT

CACHE_DIR = PROJECT_ROOT / "data" / "mechanism"

# 도메인별 타깃 변수 (knowledge_pool/domains.yaml)
TARGET = {
    "Agriculture": "Retail Broiler Composite price",
    "Economy": "International Trade Balance",
    "Security": "Disaster and Emergency Grants",
    "SocialGood": "Unemployment Rate",
    "Traffic": "Travel Volume",
}

# §7 State Summary + §8 Query Writer를 한 번에 (같은 8개월 state로 summary와 query 동시 작성)
_SUMMARY_QUERY_PROMPT = """You are given a forecasting state: a target variable, a forecast \
horizon, the look-back numerical values (oldest to newest), and the aligned textual \
evidence over the look-back window.

Do two things together, from the same state:
1. summary: summarize the current forecasting state — the numerical trend and recent \
changes, any abnormal pattern, the salient textual events (ignore irrelevant/noisy text), \
the target variable, the forecast horizon, and possible transition cues.
2. query: transform the state into a retrieval phrase that will be matched (by embedding \
similarity) against mechanism descriptions. It is NOT a question. Write it as a declarative \
statement, in the same descriptive style as a mechanism summary, that renders the current \
situation together with the candidate mechanism pathways carrying it toward the future \
target. Include the target variable, the salient state features (numerical pattern + \
textual cues), the horizon, and plausible cause->pathway->effect chains. Do not phrase it \
as a question and do not just list keywords. \
Example style: "declining trade balance under a demand-driven economic slump with falling \
exports and imports, with likely delayed demand response and countercyclical adjustment \
over a medium-term horizon".

Return only JSON: {"summary": "...", "query": "..."}"""

_REASON_PROMPT = """You are a forecasting reasoning agent. Given the current forecasting \
state summary and a set of retrieved mechanisms, produce the mechanism reasoning as a fixed \
labeled template. Use exactly these five labels, one per line, in this order, and add \
nothing else:

DIRECTION: the expected direction/shape of the future target trajectory.
MECHANISMS: the retrieved mechanisms that are actually relevant here (names, semicolon-separated).
STATE LINK: how the current numerical-textual state connects to those mechanisms.
HORIZON: horizon-specific or delayed effects over the forecast horizon.
CONFLICT: resolve disagreements between mechanisms and give the net effect (or note uncertainty).
CONCLUSION: a single self-contained natural-language sentence giving the expected future \
trajectory of the target over the horizon, covering in this fixed order: (1) direction, \
(2) qualitative magnitude (small / moderate / large), (3) timing (immediate / gradual / \
delayed), (4) persistence (transient or mean-reverting vs. persistent / level shift), and \
(5) confidence (low / medium / high). Use qualitative terms only, never fabricate numbers. \
It must stand alone without referring to the fields above.

Keep each of the first five fields to one or two sentences."""


def state_text(domain, values, text, horizon):
    """인스턴스의 8개월 수치+텍스트를 현재 state 입력 문자열로 (summary+query writer의 입력)."""
    vals = ", ".join(f"{v:.3f}" for v in values)
    return (f"Target: {TARGET.get(domain, domain)} ({domain}), monthly.\n"
            f"Forecast horizon: {horizon} months.\n"
            f"Look-back values (oldest to newest): {vals}\n"
            f"Aligned textual evidence over the look-back window:\n{text or 'No information'}")


def _parse_json(txt):
    txt = txt.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    return json.loads(txt)


def summarize_and_query(state):
    """§7+§8: state로부터 state summary와 mechanism query를 한 번에. (summary, query) 반환."""
    out = _parse_json(call_llm(_SUMMARY_QUERY_PROMPT, state, model="gpt-4o-mini"))
    return out["summary"].strip(), out["query"].strip()


def reason(summary, retrieved):
    """§11: state summary + 검색된 메커니즘 → 최종 영향분석 reasoning r_t (gpt-4o)."""
    mechs = "\n".join(f"- {r['title']}: {r['content']}" for r in retrieved)
    return call_llm(_REASON_PROMPT, f"State summary:\n{summary}\n\nRetrieved mechanisms:\n{mechs}",
                    model="gpt-4o").strip()


def _field(text, label):
    """라벨 템플릿에서 'LABEL: ...' 값 추출 (없으면 빈 문자열)."""
    for line in text.splitlines():
        if line.strip().upper().startswith(label.upper()):
            return line.split(":", 1)[1].strip()
    return ""


def mechanism_context(pool, domain, values, text, horizon, k=5):
    """① state→(summary,query) ② retrieve ③ reasoning. reasoning은 6필드 라벨 템플릿.
    conclusion = 그중 CONCLUSION 필드(Phase 4에서 임베딩되는 유일한 부분)."""
    summary, query = summarize_and_query(state_text(domain, values, text, horizon))
    retrieved = pool.retrieve(query, k=k)
    r = reason(summary, retrieved)
    return {"summary": summary, "query": query, "retrieved": retrieved,
            "reasoning": r, "conclusion": _field(r, "CONCLUSION")}


def _origins(df, texts, L, H):
    """예측 가능한 각 윈도우의 (origin_date, look-back values[L], window_text).
    origin = 윈도우 마지막 달(예측 원점); 타깃 y가 있으려면 origin+H < len."""
    ot = df["OT"].to_numpy()
    out = []
    for origin in range(L - 1, len(df) - H):
        vals = ot[origin - L + 1: origin + 1]
        wt = "\n".join(t for t in texts[origin - L + 1: origin + 1] if t != "No information")
        out.append((str(df["date"].iloc[origin].date()), vals, wt))
    return out


def precompute_domain(domain, L=8, H=6, k=5, limit=None):
    """도메인의 모든 예측 원점에 대해 r_t를 1회 생성·캐시(원점 date로 키). 이미 있으면 skip.
    캐시 = data/mechanism/{domain}_L{L}_H{H}.jsonl (record: date, query, retrieved_ids, reasoning)."""
    from .data.loader import load_domain_frame
    from .models.text_encoder import build_text_series

    df = load_domain_frame(domain)
    origins = _origins(df, build_text_series(df), L, H)
    if limit:
        origins = origins[:limit]
    pool = MechanismPool(domain)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{domain}_L{L}_H{H}.jsonl"
    cache = {}
    if path.exists():
        cache = {r["date"]: r for r in (json.loads(l) for l in open(path) if l.strip())}

    n_new = 0
    for date, vals, wt in origins:
        if date in cache:
            continue
        ctx = mechanism_context(pool, domain, vals, wt, H, k=k)
        cache[date] = {"date": date, "summary": ctx["summary"], "query": ctx["query"],
                       "retrieved_ids": [r["id"] for r in ctx["retrieved"]],
                       "reasoning": ctx["reasoning"], "conclusion": ctx["conclusion"]}
        n_new += 1
        if n_new % 20 == 0:
            print(f"  ... {n_new} new")
    with open(path, "w") as f:
        for date in sorted(cache):
            f.write(json.dumps(cache[date], ensure_ascii=False) + "\n")
    print(f"[{domain} L{L} H{H}] {len(cache)} cached (+{n_new} new) -> {path}")
    return cache


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="mechanism reasoning precompute + cache")
    ap.add_argument("--domain", default="Economy")
    ap.add_argument("--L", type=int, default=8)
    ap.add_argument("--horizon", type=int, default=6)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--limit", type=int, default=None, help="처음 N개 원점만(비용 테스트용)")
    args = ap.parse_args()
    cache = precompute_domain(args.domain, args.L, args.horizon, args.k, args.limit)
    r = cache[sorted(cache)[0]]
    print(f"\n[예시 record] {r['date']}\n  query: {r['query'][:100]}...\n"
          f"  retrieved: {r['retrieved_ids']}\n  reasoning: {r['reasoning'][:160]}...")
