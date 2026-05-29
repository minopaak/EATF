"""
Time-MMD 도메인 통합 CSV 생성 (monthly 5개 도메인 지원)

설계 문서: MD_files/02_dataset_design.md
- Climate는 weekly이므로 제외 (설계의 "down-sample 금지" 원칙)
- Monthly 5개: Agriculture, Economy, Security, SocialGood, Traffic

Usage:
    uv run python build_dataset.py --domain Agriculture
    uv run python build_dataset.py --all
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
TIME_MMD_ROOT = SCRIPT_DIR / "clones" / "Time-MMD"
OUTPUT_DIR = SCRIPT_DIR / "data" / "processed"
NO_INFO = "No information"

# 도메인별로 drop할 컬럼 (중복 date / 메타데이터 / 상수 ID)
DOMAIN_CONFIG = {
    'Agriculture': {'drop_cols': ['Date']},
    'Economy':     {'drop_cols': ['Month']},
    'Security':    {'drop_cols': []},
    'SocialGood':  {'drop_cols': []},
    'Traffic':     {'drop_cols': ['Date']},
}

MONTHLY_DOMAINS = list(DOMAIN_CONFIG.keys())


def load_numerical(domain: str) -> pd.DataFrame:
    path = TIME_MMD_ROOT / "numerical" / domain / f"{domain}.csv"
    df = pd.read_csv(path)

    drop = [c for c in DOMAIN_CONFIG[domain]['drop_cols'] if c in df.columns]
    if drop:
        df = df.drop(columns=drop)

    df['start_date'] = pd.to_datetime(df['start_date'])
    df['end_date'] = pd.to_datetime(df['end_date'])
    df['date'] = df['start_date'].dt.to_period('M').dt.to_timestamp()

    # Time-MMD numerical CSV이 도메인별로 정렬 상태가 다름 (Economy는 연도 desc) → 시간순 강제
    df = df.sort_values('date').reset_index(drop=True)

    return df


def assign_month_majority(start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.Timestamp:
    """텍스트 윈도우 [start, end]가 가장 많이 겹친 월. 동률이면 빠른 월."""
    dates = pd.date_range(start_date, end_date, freq='D')
    counts = dates.to_period('M').value_counts()
    max_count = counts.max()
    earliest_max = counts[counts == max_count].index.min()
    return earliest_max.to_timestamp()


def load_textual(domain: str, source: str) -> pd.DataFrame:
    path = TIME_MMD_ROOT / "textual" / domain / f"{domain}_{source}.csv"
    df = pd.read_csv(path)
    df['start_date'] = pd.to_datetime(df['start_date'])
    df['end_date'] = pd.to_datetime(df['end_date'])

    # Time-MMD 업스트림 버그: 연말~연초 윈도우에서 end_date의 연도가 1년 적게 들어간 row가 있음
    # 예: start=2022-12-26, end=2022-01-01 (실제론 2023-01-01이어야 함)
    inverted = df['start_date'] > df['end_date']
    if inverted.any():
        df.loc[inverted, 'end_date'] = df.loc[inverted, 'end_date'] + pd.DateOffset(years=1)
        print(f"    [fix] {domain}/{source}: 연도-경계 inverted row {inverted.sum()}개 end_date+=1y")

    # 보정 후에도 비정상이면 drop
    still_bad = df['start_date'] > df['end_date']
    if still_bad.any():
        print(f"    [warn] {domain}/{source}: 보정 불가 row {still_bad.sum()}개 drop")
        df = df[~still_bad].copy()

    df['month'] = df.apply(
        lambda r: assign_month_majority(r['start_date'], r['end_date']),
        axis=1,
    )
    return df


def group_text_by_month(text_df: pd.DataFrame, col: str) -> pd.Series:
    # 내용(col)이 비어있는 row 제외 → "날짜: " 빈 prefix artifact 방지.
    # (Time-MMD 원본에 fact/preds가 NaN인 row 존재. 예: Security report 전체, search의 빈 주)
    # 내용 있는 row가 한 달에 하나도 없으면 그 달은 series에서 빠지고 → 호출부에서 "No information"으로 채워짐
    df = text_df[text_df[col].notna() & (text_df[col].astype(str).str.strip() != '')].copy()
    if df.empty:
        return pd.Series(dtype='object')

    df['formatted'] = df.apply(
        lambda r: f"{r['start_date'].strftime('%Y-%m-%d')}: {r[col]}", axis=1
    )
    return df.groupby('month')['formatted'].apply(lambda x: '\n'.join(x))


def merge_text_to_numerical(num_df: pd.DataFrame, text_df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    fact_grouped = group_text_by_month(text_df, 'fact')
    pred_grouped = group_text_by_month(text_df, 'preds')
    num_df[f'{prefix}_text'] = num_df['date'].map(fact_grouped).fillna(NO_INFO)
    num_df[f'{prefix}_pred'] = num_df['date'].map(pred_grouped).fillna(NO_INFO)
    return num_df


def trim_to_text_coverage(df: pd.DataFrame) -> pd.DataFrame:
    """텍스트(report 또는 search)가 처음/마지막으로 등장하는 행 사이로 trim.

    이벤트-aware 멀티모달 벤치마크 원칙: 텍스트가 통째로 없는 앞/뒤 구간 제거.
    (내부의 산발적 no-info 월은 유지 — 그건 sparsity일 뿐 시대 결손이 아님.)
    Time-MMD numerical은 텍스트보다 과거까지 확장돼 있어(예: SocialGood 1948~,
    Traffic 1970~) 텍스트 시대 이전 구간이 생김. 그 구간은 멀티모달 분석 불가라 제거.
    """
    has_text = ((df['report_text'] != NO_INFO) | (df['search_text'] != NO_INFO)).to_numpy()
    if not has_text.any():
        print("    [warn] 텍스트가 전혀 없음 — trim 생략")
        return df.reset_index(drop=True)
    first = int(has_text.argmax())
    last = len(has_text) - 1 - int(has_text[::-1].argmax())
    return df.iloc[first:last + 1].reset_index(drop=True)


def build_domain(domain: str) -> Path:
    if domain not in DOMAIN_CONFIG:
        raise ValueError(
            f"Unknown or non-monthly domain: {domain}. Valid: {MONTHLY_DOMAINS}"
        )

    print(f"\n=== {domain} ===")

    num_df = load_numerical(domain)
    print(
        f"  Numerical rows: {len(num_df)}, "
        f"range: {num_df['date'].min().date()} ~ {num_df['date'].max().date()}"
    )

    for source in ['report', 'search']:
        text_df = load_textual(domain, source)
        print(f"  {source.capitalize():6s} rows: {len(text_df)}")
        num_df = merge_text_to_numerical(num_df, text_df, source)

    text_cols = ['report_text', 'search_text', 'report_pred', 'search_pred']
    meta_cols = ['date', 'start_date', 'end_date']
    var_cols = [c for c in num_df.columns if c not in meta_cols + text_cols]
    final_cols = ['date', 'start_date', 'end_date'] + var_cols + text_cols
    final_df = num_df[final_cols]

    # 텍스트 시대로 trim: 텍스트가 없는 앞/뒤 구간 제거 (이벤트-aware 벤치마크 원칙).
    # 늦게 시작하는 보조 변수의 초기 NaN은 그대로 둠 — 실험별 처리는 로더가 담당:
    #   단변량(OT) → 텍스트 시대 OT 풀,  다변량 → 그 안의 공통 윈도우(전 변수 유효 구간)
    n_before = len(final_df)
    final_df = trim_to_text_coverage(final_df)
    if len(final_df) < n_before:
        print(f"  Trim(텍스트 시대): {n_before} → {len(final_df)} rows "
              f"({final_df['date'].min().date()} ~ {final_df['date'].max().date()})")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"{domain}_merged.csv"
    final_df.to_csv(output_path, index=False)

    report_empty = (final_df['report_text'] == NO_INFO).mean()
    search_empty = (final_df['search_text'] == NO_INFO).mean()
    print(
        f"  Saved: {output_path.relative_to(SCRIPT_DIR)} (shape={final_df.shape})"
    )
    print(f"  Variables ({len(var_cols)}): {var_cols}")
    print(
        f"  No-info: report={report_empty:.1%}, search={search_empty:.1%}"
    )

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Build merged monthly CSV for Time-MMD domain(s)."
    )
    parser.add_argument(
        '--domain', type=str, default=None,
        help=f"Single domain to process. Choose from {MONTHLY_DOMAINS}",
    )
    parser.add_argument('--all', action='store_true', help='Process all monthly domains')
    args = parser.parse_args()

    if args.all:
        targets = MONTHLY_DOMAINS
    elif args.domain:
        targets = [args.domain]
    else:
        parser.error("Specify --domain DOMAIN or --all")

    for d in targets:
        build_domain(d)


if __name__ == "__main__":
    main()
