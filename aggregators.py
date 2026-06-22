"""
aggregators.py
==============
Darts의 `Aggregator` 기반 클래스를 확장한 커스텀 Aggregator.

강의 14의 Aggregator 모듈(다변량 이진 시계열 → 단변량 이진 시계열)은
`OrAggregator`(하나라도 이상)와 `AndAggregator`(전부 동의)만 표준으로 제공한다.
실무에서는 "전체 Scorer 중 일정 수준 이상이 동의하면 이상"으로 판정하는
다수결(majority vote) 합의가 더 유용한 경우가 많다. 이를 Darts의
Aggregator API 규약(`_predict_core` 구현)에 맞춰 직접 구현한다.

    consensus_level = 1.0          → AndAggregator와 동일 (전원 동의)
    consensus_level → 0 (작을수록) → OrAggregator에 근접 (한 명이라도 동의)
    consensus_level = 0.5          → 과반 동의 (기본값)

즉 Or/And를 양 끝 경우로 포함하는 일반화된 합의 Aggregator다.
"""

from __future__ import annotations

import numpy as np

from darts import TimeSeries
from darts.ad.aggregators import Aggregator

try:
    from darts.utils.utils import _parallel_apply
except ImportError:  # pragma: no cover
    def _parallel_apply(args_list, fn, n_jobs=1, fn_args=None, fn_kwargs=None):
        fn_args = fn_args or ()
        fn_kwargs = fn_kwargs or {}
        return [fn(*args, *fn_args, **fn_kwargs) for args in args_list]


class MajorityVoteAggregator(Aggregator):
    """합의 수준(consensus_level) 이상이 동의해야 이상으로 판정하는 합의 Aggregator."""

    def __init__(self, consensus_level: float = 0.5, n_jobs: int = 1) -> None:
        if not (0 < consensus_level <= 1):
            raise ValueError("consensus_level은 (0, 1] 구간이어야 합니다.")
        super().__init__()
        self._consensus_level = consensus_level
        self._n_jobs = n_jobs

    def __str__(self) -> str:
        return f"MajorityVoteAggregator(consensus_level={self._consensus_level})"

    @property
    def consensus_level(self) -> float:
        return self._consensus_level

    def _predict_core(self, series, *args, **kwargs):
        def _vote(s: TimeSeries) -> TimeSeries:
            vals = s.all_values(copy=False)
            n_voters = vals.shape[1]
            votes_on = vals.sum(axis=1)
            k = int(np.ceil(self._consensus_level * n_voters))
            decided = (votes_on >= k).astype(s.dtype)
            return TimeSeries(
                times=s.time_index,
                values=decided,
                components=["majority_vote"],
                copy=False,
            )

        return _parallel_apply(
            [(s,) for s in series], _vote, n_jobs=1, fn_args=args, fn_kwargs=kwargs
        )

    @staticmethod
    def votes_needed(n_voters: int, consensus_level: float) -> int:
        return int(np.ceil(consensus_level * n_voters))
