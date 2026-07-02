"""학습용 데이터 로딩 (data/processed/*.csv → 윈도우 텐서).

(데이터 구축은 루트의 build_dataset.py 담당)
"""
from .loader import (
    WindowDataset,
    load_domain_frame,
    get_var_columns,
    build_in_domain,
    make_dataloader,
    MONTHLY_DOMAINS,
)

__all__ = [
    "WindowDataset",
    "load_domain_frame",
    "get_var_columns",
    "build_in_domain",
    "make_dataloader",
    "MONTHLY_DOMAINS",
]
