"""
anomaly_pipeline.py
====================
시계열 이상탐지 핵심 파이프라인 (강의 14 '시계열 이상탐지' 기반).

강의에서 다룬 Darts 이상탐지 4모듈 구조를 그대로 따른다.

    Time series ─▶ [Anomaly Model: Forecasting + Scorer] ─▶ Anomaly score
    Anomaly score ─▶ [Detector] ─▶ Binary prediction
    Binary preds  ─▶ [Aggregator] ─▶ 통합 Binary prediction
    평가: eval_metric_from_scores (AUC-ROC / AUC-PR) + sklearn 지표

이 모듈은 Streamlit UI(app.py)와 분리되어 있어 단독으로도 테스트가 가능하다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import reduce
from typing import Optional

import numpy as np
import pandas as pd

from darts import TimeSeries
from darts.ad import (
    ForecastingAnomalyModel,
    NormScorer,
    KMeansScorer,
    WassersteinScorer,
    QuantileDetector,
)
from darts.ad.aggregators import OrAggregator, AndAggregator
from darts.ad.utils import eval_metric_from_scores
from darts.dataprocessing.transformers import Scaler
from darts.models import SKLearnModel

try:
    from darts.ad import PyODScorer
    from pyod.models.lof import LOF
    from pyod.models.copod import COPOD
    from pyod.models.ecod import ECOD
    PYOD_AVAILABLE = True
except ImportError:
    PYOD_AVAILABLE = False

from aggregators import MajorityVoteAggregator

from sklearn.metrics import (
    precision_recall_fscore_support,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)


# --------------------------------------------------------------------------- #
# 설정 (UI에서 전달)
# --------------------------------------------------------------------------- #
@dataclass
class ScorerConfig:
    use_norm: bool = True
    use_kmeans: bool = False
    use_wasserstein: bool = True
    window: int = 24            # KMeans / Wasserstein / PyOD 윈도우 길이
    kmeans_k: int = 2           # KMeans 군집 수
    # --- PyODScorer 확장: Darts가 PyOD 탐지기를 Scorer로 감싸는 표준 방식 ---
    use_lof: bool = False       # Local Outlier Factor (밀도 기반, 국소 패턴에 강함)
    use_copod: bool = False     # Copula-Based Outlier Detection (분포 기반, 빠름)
    use_ecod: bool = False      # Empirical CDF 기반 (가정 적음, 해석 쉬움)
    lof_neighbors: int = 20     # LOF 이웃 수


@dataclass
class AggregatorConfig:
    """여러 Scorer의 이진 예측을 통합하는 방식.

    "or"/"and"는 Darts 표준 Aggregator. "majority"는 본 파이프라인에서
    Darts의 Aggregator API를 확장해 구현한 합의 수준 기반 Aggregator로,
    consensus_level=1.0이면 and와 동일, consensus_level이 작을수록 or에 근접하는 일반화다.
    """
    method: str = "majority"           # "or" | "and" | "majority"
    consensus_level: float = 0.5       # method="majority"일 때 합의 수준 (과반=0.5)


@dataclass
class PipelineConfig:
    time_col: str
    value_cols: list[str]
    label_col: Optional[str] = None      # 정답(ground-truth) 라벨이 있으면 지정
    train_ratio: float = 0.5             # 학습/테스트 분할 비율
    lags: int = 48                       # 예측 모델 lag (강의: one_week=7*24*2)
    detector_quantile: float = 0.95      # Detector 임계 분위수
    detector_fit_mode: str = "test"      # "test" | "train" — Detector를 어느 구간 점수로 학습할지
    scorers: ScorerConfig = field(default_factory=ScorerConfig)
    aggregator: AggregatorConfig = field(default_factory=AggregatorConfig)


# --------------------------------------------------------------------------- #
# 유틸
# --------------------------------------------------------------------------- #
def _to_univariate(ts: TimeSeries, how: str = "max") -> TimeSeries:
    """다변량 이상 점수를 컴포넌트 축으로 축소하여 단변량으로 변환."""
    v = ts.values()
    if ts.n_components > 1:
        v = {"max": v.max, "mean": v.mean, "sum": v.sum}[how](axis=1)
    else:
        v = v.ravel()
    return TimeSeries.from_times_and_values(ts.time_index, v)


def _build_scorers(cfg: ScorerConfig):
    """설정에 따라 Scorer 리스트를 구성.

    NormScorer/KMeansScorer/WassersteinScorer는 강의 14에서 다룬 Darts 기본
    Scorer. LOF/COPOD/ECOD는 `PyODScorer`로 PyOD의 비지도 이상탐지 모델을
    Darts Scorer 규약(시계열 → 이상 점수)에 맞춰 감싼 확장이다. 예측오차가
    아니라 (변환 윈도우 내) 특징 분포 자체로 이상을 판단해, 예측 기반
    Scorer가 놓치는 패턴을 보완한다.
    """
    scorers, names = [], []
    pyod_skipped = False
    if cfg.use_norm:
        scorers.append(NormScorer(component_wise=True))
        names.append("NormScorer")
    if cfg.use_kmeans:
        scorers.append(KMeansScorer(k=cfg.kmeans_k, window=cfg.window, component_wise=True))
        names.append("KMeansScorer")
    if cfg.use_wasserstein:
        scorers.append(WassersteinScorer(window=cfg.window, component_wise=True))
        names.append("WassersteinScorer")
    if PYOD_AVAILABLE and cfg.use_lof:
        scorers.append(PyODScorer(
            model=LOF(n_neighbors=cfg.lof_neighbors), window=cfg.window, component_wise=True
        ))
        names.append("LOF(PyOD)")
    elif cfg.use_lof:
        pyod_skipped = True
    if PYOD_AVAILABLE and cfg.use_copod:
        scorers.append(PyODScorer(model=COPOD(), window=cfg.window, component_wise=True))
        names.append("COPOD(PyOD)")
    elif cfg.use_copod:
        pyod_skipped = True
    if PYOD_AVAILABLE and cfg.use_ecod:
        scorers.append(PyODScorer(model=ECOD(), window=cfg.window, component_wise=True))
        names.append("ECOD(PyOD)")
    elif cfg.use_ecod:
        pyod_skipped = True
    if not scorers:  # 최소 하나는 보장
        scorers.append(NormScorer(component_wise=True))
        names.append("NormScorer")
    return scorers, names, pyod_skipped


# --------------------------------------------------------------------------- #
# 데이터 로딩 & 전처리 (강의: 데이터 준비 → TimeSeries → Scaler)
# --------------------------------------------------------------------------- #
def load_and_prepare(df: pd.DataFrame, cfg: PipelineConfig):
    """CSV DataFrame → 정렬·결측 처리 → Darts TimeSeries (+ 라벨)."""
    df = df.copy()
    df[cfg.time_col] = pd.to_datetime(df[cfg.time_col], errors="coerce")
    df = df.dropna(subset=[cfg.time_col]).sort_values(cfg.time_col)
    df = df.drop_duplicates(subset=[cfg.time_col])

    # 값 컬럼 결측은 선형 보간 후 양끝 채움
    df[cfg.value_cols] = (
        df[cfg.value_cols]
        .apply(pd.to_numeric, errors="coerce")
        .interpolate(method="linear", limit_direction="both")
    )
    df = df.dropna(subset=cfg.value_cols)

    # 결측 날짜가 있으면 채우고(fill_missing_dates), 생긴 결측은 보간
    series = TimeSeries.from_dataframe(
        df, time_col=cfg.time_col, value_cols=cfg.value_cols, fill_missing_dates=True
    )
    if np.isnan(series.values()).any():
        from darts.dataprocessing.transformers import MissingValuesFiller
        series = MissingValuesFiller().transform(series)

    labels = None
    if cfg.label_col and cfg.label_col in df.columns:
        lab = pd.to_numeric(df[cfg.label_col], errors="coerce").fillna(0)
        lab = (lab > 0).astype(int)
        lab.index = pd.to_datetime(df[cfg.time_col].values)
        # 시계열 인덱스(결측 보간 후 길이 변동 가능)에 정렬, 없는 시점은 정상(0)
        lab = lab.reindex(series.time_index, fill_value=0)
        labels = TimeSeries.from_times_and_values(
            series.time_index, lab.values.reshape(-1, 1)
        )
    return series, labels, df


# --------------------------------------------------------------------------- #
# 메인 파이프라인
# --------------------------------------------------------------------------- #
def run_pipeline(df: pd.DataFrame, cfg: PipelineConfig) -> dict:
    """업로드된 데이터에 대해 전체 이상탐지 파이프라인을 실행."""
    series, labels, clean_df = load_and_prepare(df, cfg)

    n = len(series)
    if n < cfg.lags * 3:
        raise ValueError(
            f"데이터가 너무 짧습니다 (관측치 {n}개). lag({cfg.lags})를 줄이거나 "
            f"더 긴 데이터를 사용하세요. 권장: 관측치 ≥ lag × 3."
        )

    # 1) 스케일링 (강의: Scaler)
    scaler = Scaler()
    series_s = scaler.fit_transform(series)

    # 2) 학습/테스트 분할
    train, test = series_s.split_before(cfg.train_ratio)
    if len(train) <= cfg.lags:
        raise ValueError(
            f"학습 구간({len(train)})이 lag({cfg.lags})보다 짧습니다. "
            f"train_ratio를 키우거나 lag를 줄이세요."
        )

    test_labels = None
    if labels is not None:
        test_labels = labels.split_before(cfg.train_ratio)[1]

    # 3) 예측 모델 (강의: SKLearnModel with lags) + Scorer 결합 = ForecastingAnomalyModel
    scorers, scorer_names, pyod_skipped = _build_scorers(cfg.scorers)
    fmodel = SKLearnModel(lags=cfg.lags, output_chunk_length=1)
    anomaly_model = ForecastingAnomalyModel(model=fmodel, scorer=scorers)
    anomaly_model.fit(train, allow_model_training=True)

    # 4) 이상 점수 계산 (강의: score())
    raw_scores = anomaly_model.score(test)
    # scorer가 여러 개면 tuple/list, 하나면 단일 TimeSeries로 반환됨
    if isinstance(raw_scores, TimeSeries):
        raw_scores = [raw_scores]
    else:
        raw_scores = list(raw_scores)

    # detector_fit_mode="train"이면 train 구간 점수도 계산 (Detector를 train 분포로 학습)
    raw_scores_train = None
    if cfg.detector_fit_mode == "train":
        raw_scores_train = anomaly_model.score(train)
        if isinstance(raw_scores_train, TimeSeries):
            raw_scores_train = [raw_scores_train]
        else:
            raw_scores_train = list(raw_scores_train)

    # 5) 결과 정리: scorer별 점수(단변량/컴포넌트별) + Detector 이진화
    results: dict[str, dict] = {}
    binary_list = []
    for i, (name, sc) in enumerate(zip(scorer_names, raw_scores)):
        uni = _to_univariate(sc, "max")           # 평가/표시용 단변량 점수
        det = QuantileDetector(high_quantile=cfg.detector_quantile)
        if cfg.detector_fit_mode == "train" and raw_scores_train is not None:
            uni_train = _to_univariate(raw_scores_train[i], "max")
            det.fit(uni_train)
            binary = det.detect(uni)
        else:
            binary = det.fit_detect(uni)          # Detector → 이진 예측
        binary_list.append(binary)

        entry = {
            "score_uni": uni,
            "score_components": sc,                # 컴포넌트별 점수 (다변량)
            "binary": binary,
            "metrics": None,
            "curves": None,
        }

        # 6) 정답 라벨이 있으면 정량 평가 (강의: eval_metric_from_scores AUC-ROC)
        if test_labels is not None:
            entry["metrics"], entry["curves"] = _evaluate(uni, binary, test_labels)
        results[name] = entry

    # 7) Aggregator: 여러 scorer 이진 예측 → 통합
    #    "or"/"and"는 Darts 표준, "majority"는 본 파이프라인에서 확장한
    #    합의 수준 Aggregator(consensus_level=1.0이면 and와 동등, 작을수록 or에 근접).
    aggregated = None
    agg_info = {"method": cfg.aggregator.method, "consensus_level": cfg.aggregator.consensus_level,
               "n_voters": len(binary_list), "k": None}
    if len(binary_list) >= 1:
        # 윈도우 길이가 달라 길이가 다를 수 있으므로 교집합으로 정렬
        common_start = max(b.start_time() for b in binary_list)
        common_end = min(b.end_time() for b in binary_list)
        aligned = [b.slice(common_start, common_end) for b in binary_list]
        stacked = reduce(lambda x, y: x.stack(y), aligned)

        if len(binary_list) == 1:
            aggregated = aligned[0]  # 단일 scorer면 통합 불필요
        elif cfg.aggregator.method == "and":
            aggregated = AndAggregator().predict(stacked)
            agg_info["k"] = len(binary_list)
        elif cfg.aggregator.method == "or":
            aggregated = OrAggregator().predict(stacked)
            agg_info["k"] = 1
        else:  # "majority"
            mv = MajorityVoteAggregator(consensus_level=cfg.aggregator.consensus_level)
            aggregated = mv.predict(stacked)
            agg_info["k"] = MajorityVoteAggregator.votes_needed(len(binary_list), cfg.aggregator.consensus_level)

    # 8) 라벨-프리 진단 지표 (정답이 없어도 '적절성'을 판단할 수 있도록)
    diagnostics = _diagnostics(results, aggregated)

    meta = {
        "n_total": n,
        "n_train": len(train),
        "n_test": len(test),
        "freq": humanize_freq(str(series.freq)),
        "n_components": series.n_components,
        "value_cols": cfg.value_cols,
        "has_labels": test_labels is not None,
        "scorer_names": scorer_names,
        "aggregator": agg_info,
        "detector_fit_mode": cfg.detector_fit_mode,
    }

    return {
        "series": series,                 # 원본(스케일 전) 시계열
        "series_scaled": series_s,
        "test_index_range": (test.start_time(), test.end_time()),
        "labels": labels,
        "test_labels": test_labels,
        "results": results,               # scorer별 점수·이진·평가
        "aggregated": aggregated,         # 통합 이상 플래그
        "aggregator_info": agg_info,      # 합의 방식/비율/임계표수
        "diagnostics": diagnostics,
        "meta": meta,
        "clean_df": clean_df,
    }


# --------------------------------------------------------------------------- #
# 수동 보정 (클릭 기반 오탐 제거 / 이상 추가)
# --------------------------------------------------------------------------- #
def apply_manual_overrides(aggregated: TimeSeries, overrides: dict) -> TimeSeries:
    """
    수동 보정(override)을 통합 이상 플래그(aggregated)에 적용해 새 TimeSeries를 반환한다.

    Parameters
    ----------
    aggregated : TimeSeries
        out["aggregated"] (단변량, 값 0.0/1.0)
    overrides : dict[pd.Timestamp, int]
        {타임스탬프: 0 또는 1} — 0이면 그 시점을 "정상"으로 강제,
        1이면 그 시점을 "이상"으로 강제. aggregated.time_index에 없는
        키는 무시한다(에러를 내지 않는다).

    Returns
    -------
    TimeSeries
        overrides가 반영된 새 TimeSeries. 원본(aggregated)은 변경하지 않는다(불변 유지).
    """
    time_index = aggregated.time_index
    values = aggregated.values().ravel().copy()
    for ts, flag in overrides.items():
        if ts in time_index:
            pos = time_index.get_loc(ts)
            values[pos] = float(flag)
    return TimeSeries.from_times_and_values(time_index, values.reshape(-1, 1))


def override_diff_summary(aggregated: TimeSeries, overrides: dict) -> pd.DataFrame:
    """
    수동 보정 내역을 표로 만들어 반환한다(다운로드 탭/감사용).

    반환 컬럼: time, original_flag(int), overridden_flag(int), action(str)
        action은 "오탐 제거"(1->0) 또는 "이상 추가"(0->1) 중 하나의 한글 문자열.
    overrides가 비어 있으면 빈 DataFrame(같은 컬럼 구조)을 반환한다.
    """
    cols = ["time", "original_flag", "overridden_flag", "action"]
    if not overrides:
        return pd.DataFrame(columns=cols)

    time_index = aggregated.time_index
    values = aggregated.values().ravel()
    rows = []
    for ts, flag in overrides.items():
        if ts not in time_index:
            continue
        pos = time_index.get_loc(ts)
        original_flag = int(values[pos])
        overridden_flag = int(flag)
        action = "오탐 제거" if overridden_flag == 0 else "이상 추가"
        rows.append({
            "time": ts,
            "original_flag": original_flag,
            "overridden_flag": overridden_flag,
            "action": action,
        })
    return pd.DataFrame(rows, columns=cols)


# --------------------------------------------------------------------------- #
# 평가 (라벨 있는 경우)
# --------------------------------------------------------------------------- #
def _evaluate(score_uni: TimeSeries, binary: TimeSeries, labels: TimeSeries):
    """AUC-ROC / AUC-PR + Precision/Recall/F1 + Confusion matrix + 곡선."""
    auc_roc = float(eval_metric_from_scores(anomalies=labels, pred_scores=score_uni, metric="AUC_ROC"))
    auc_pr = float(eval_metric_from_scores(anomalies=labels, pred_scores=score_uni, metric="AUC_PR"))

    # 이진 예측과 라벨 정렬
    lab_al = labels.slice_intersect(binary)
    bp = binary.slice_intersect(lab_al)
    y_true = lab_al.values().ravel().astype(int)
    y_pred = bp.values().ravel().astype(int)

    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    # ROC / PR 곡선 (점수 기준 정렬)
    sc_al = score_uni.slice_intersect(lab_al)
    y_true_sc = labels.slice_intersect(sc_al).values().ravel().astype(int)
    y_score = sc_al.values().ravel()
    fpr, tpr, _ = roc_curve(y_true_sc, y_score)
    prec_c, rec_c, _ = precision_recall_curve(y_true_sc, y_score)

    metrics = {
        "AUC_ROC": auc_roc,
        "AUC_PR": auc_pr,
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "confusion_matrix": cm,
        "n_true_anomaly": int(y_true.sum()),
        "n_pred_anomaly": int(y_pred.sum()),
    }
    curves = {"roc": (fpr, tpr), "pr": (rec_c, prec_c)}
    return metrics, curves


# --------------------------------------------------------------------------- #
# 라벨-프리 진단 (정답이 없을 때 '적절성' 판단 근거)
# --------------------------------------------------------------------------- #
def _diagnostics(results: dict, aggregated: Optional[TimeSeries]) -> dict:
    """정답 라벨이 없어도 탐지 결과의 타당성을 가늠할 수 있는 지표들."""
    diag = {"per_scorer": {}}
    for name, e in results.items():
        b = e["binary"].values().ravel().astype(int)
        s = e["score_uni"].values().ravel()
        diag["per_scorer"][name] = {
            "anomaly_rate": float(b.mean()),
            "n_anomaly": int(b.sum()),
            "score_mean": float(np.mean(s)),
            "score_std": float(np.std(s)),
            "score_p95": float(np.percentile(s, 95)),
            "score_max": float(np.max(s)),
        }

    # Scorer 간 일치도 (여러 scorer가 동시에 이상이라 하면 신뢰도↑)
    names = list(results.keys())
    agreement = None
    if len(names) >= 2:
        agreement = np.zeros((len(names), len(names)))
        bins = {}
        for nm in names:
            b = results[nm]["binary"]
            bins[nm] = b
        for i, a in enumerate(names):
            for j, c in enumerate(names):
                ba = bins[a]
                bc = bins[c]
                cs = max(ba.start_time(), bc.start_time())
                ce = min(ba.end_time(), bc.end_time())
                va = ba.slice(cs, ce).values().ravel().astype(int)
                vc = bc.slice(cs, ce).values().ravel().astype(int)
                # Jaccard 유사도 (둘 다 이상=1 인 비율 / 적어도 하나가 1 인 비율)
                inter = np.logical_and(va, vc).sum()
                union = np.logical_or(va, vc).sum()
                agreement[i, j] = inter / union if union > 0 else 1.0
    diag["agreement_names"] = names
    diag["agreement_matrix"] = agreement

    if aggregated is not None:
        ab = aggregated.values().ravel().astype(int)
        diag["aggregated_anomaly_rate"] = float(ab.mean())
        diag["aggregated_n_anomaly"] = int(ab.sum())
    return diag


# --------------------------------------------------------------------------- #
# Threshold 민감도 (분위수를 바꾸면 탐지량이 어떻게 변하는지)
# --------------------------------------------------------------------------- #
def threshold_sweep(score_uni: TimeSeries, quantiles=None) -> pd.DataFrame:
    if quantiles is None:
        quantiles = np.round(np.arange(0.80, 0.995, 0.01), 3)
    s = score_uni.values().ravel()
    rows = []
    for q in quantiles:
        thr = np.quantile(s, q)
        rate = float((s >= thr).mean())
        rows.append({"quantile": float(q), "threshold": float(thr), "anomaly_rate": rate})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 정상성 검정 (ADF)
# --------------------------------------------------------------------------- #
def stationarity_report(clean_df: pd.DataFrame, value_cols: list[str]) -> pd.DataFrame:
    """
    각 값 컬럼에 대해 ADF(Augmented Dickey-Fuller) 정상성 검정을 수행해 표로 반환한다.

    반환 컬럼: column(str), adf_statistic(float), p_value(float),
              is_stationary(bool, p_value < 0.05), n_obs(int)
    데이터가 너무 짧거나(< 20) 검정이 실패하면 해당 행의 adf_statistic/p_value를
    NaN으로, is_stationary를 None으로 채우고 예외를 던지지 않는다.
    """
    from statsmodels.tsa.stattools import adfuller

    rows = []
    for col in value_cols:
        series = clean_df[col].dropna()
        n_obs = len(series)
        adf_statistic, p_value, is_stationary = np.nan, np.nan, None
        if n_obs >= 20:
            try:
                result = adfuller(series)
                adf_statistic = float(result[0])
                p_value = float(result[1])
                is_stationary = bool(p_value < 0.05)
            except Exception:
                adf_statistic, p_value, is_stationary = np.nan, np.nan, None
        rows.append({
            "column": col,
            "adf_statistic": adf_statistic,
            "p_value": p_value,
            "is_stationary": is_stationary,
            "n_obs": n_obs,
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# EDA — 결측치 현황 + 상관관계
# --------------------------------------------------------------------------- #
def missing_value_summary(raw_df: pd.DataFrame, value_cols: list[str]) -> pd.DataFrame:
    """
    보간 전 원본 데이터프레임을 받아 값 컬럼별 결측치 개수·비율을 표로 반환한다.

    반환 컬럼: column(str), n_missing(int), missing_rate(float, 0~1)
    raw_df가 비어 있거나(len 0) value_cols가 비어 있으면 빈 DataFrame을 반환한다.
    """
    n = len(raw_df)
    rows = []
    for c in value_cols:
        if c not in raw_df.columns:
            continue
        n_missing = int(raw_df[c].isna().sum())
        rows.append({
            "column": c,
            "n_missing": n_missing,
            "missing_rate": (n_missing / n) if n else 0.0,
        })
    return pd.DataFrame(rows, columns=["column", "n_missing", "missing_rate"])


def correlation_matrix(clean_df: pd.DataFrame, value_cols: list[str]) -> pd.DataFrame:
    """
    보간 후 데이터(out["clean_df"])를 받아 값 컬럼 간 Pearson 상관행렬을 반환한다.
    value_cols가 1개뿐이면 1x1 행렬(값 1.0)을 반환한다.
    """
    return clean_df[value_cols].corr()


def is_pyod_available() -> bool:
    return PYOD_AVAILABLE


def humanize_freq(freq_str: str) -> str:
    """Darts freq repr(예: '<30 * Minutes>', '<Hour>')을 'N분/N시간/N일' 형태로 변환."""
    import re

    s = str(freq_str).strip("<>")
    unit_map = {
        "second": "초", "seconds": "초",
        "minute": "분", "minutes": "분",
        "hour": "시간", "hours": "시간",
        "day": "일", "days": "일",
        "week": "주", "weeks": "주",
        "month": "개월", "months": "개월",
    }
    m = re.match(r"(\d+)\s*\*\s*(\w+)", s)
    if m:
        n, unit = m.groups()
        return f"{n}{unit_map.get(unit.lower(), unit)}"
    if s.lower() in unit_map:
        return f"1{unit_map[s.lower()]}"
    return s
