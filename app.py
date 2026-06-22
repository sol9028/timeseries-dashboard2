"""
app.py — 다변량 시계열 이상탐지 대시보드 (Streamlit)
=====================================================
임의의 다변량 시계열 CSV를 업로드하면 자동으로 이상탐지를 수행하고,
다양한 평가지표 시각화 대시보드로 탐지 결과의 적절성을 판단할 수 있다.

파이프라인은 강의 14 '시계열 이상탐지'(Darts 4모듈 구조)를 확장해 따른다.
    Forecasting + Scorer(Darts 기본 + PyOD 확장) -> 이상 점수
    -> Detector(이진화) -> Aggregator(Or/And/다수결 통합) -> 평가

실행:  streamlit run app.py
"""

import hashlib
import io

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from anomaly_pipeline import (
    PipelineConfig,
    ScorerConfig,
    AggregatorConfig,
    run_pipeline,
    threshold_sweep,
    apply_manual_overrides,
    override_diff_summary,
    stationarity_report,
    missing_value_summary,
    correlation_matrix,
    is_pyod_available,
)

st.set_page_config(page_title="시계열 이상탐지 대시보드", layout="wide", page_icon="🫐")

# 컬러 팔레트
CLR_ACCENT        = "#6464F0"
CLR_ACCENT_HOVER  = "#4F46C9"
CLR_ACCENT_ACTIVE = "#3B36A0"
CLR_ACCENT_LIGHT  = "#C7C5F5"
CLR_ANOMALY       = "#E4572E"
CLR_BLUE          = "#1D4ED8"
CLR_SLATE         = "#94A3B8"
CLR_OK            = "#2E9E5B"

st.markdown(f"""
    <style>
    div.stButton > button[kind="primary"] {{
        background-color: {CLR_ACCENT}; color: white; border: none; border-radius: 8px; font-weight: 600;
    }}
    div.stButton > button[kind="primary"]:hover {{ background-color: {CLR_ACCENT_HOVER}; }}
    div.stButton > button[kind="primary"]:active {{ background-color: {CLR_ACCENT_ACTIVE}; }}
    section[data-testid="stSidebar"] {{
        background-color: #F5F5FD;
        border-right: 2px solid {CLR_ACCENT_LIGHT};
    }}
    </style>""", unsafe_allow_html=True)

st.markdown(f"""
<div style="position: fixed; top: 60px; right: 20px; z-index: 999;
            background-color: #F5F5FD; border: 1px solid {CLR_ACCENT_LIGHT};
            border-radius: 8px; padding: 8px 14px; text-align: center;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
    <div style="font-size: 11px; color: #64748B;">Dashboard Developed by</div>
    <div style="font-size: 14px; font-weight: 700; color: {CLR_ACCENT};">C321028 박솔</div>
</div>
""", unsafe_allow_html=True)


def _hash_bytes(b: bytes) -> str:
    return hashlib.md5(b).hexdigest()


def _guess_time_col(df: pd.DataFrame) -> str:
    candidates = _time_col_candidates(df)
    if candidates:
        return candidates[0]
    return df.columns[0]


def _time_col_candidates(df: pd.DataFrame) -> list[str]:
    time_keywords = ("timestamp", "datetime", "date", "time")
    scored = []
    for c in df.columns:
        name_score = 1 if any(k in str(c).lower() for k in time_keywords) else 0
        is_numeric = pd.api.types.is_numeric_dtype(pd.to_numeric(df[c], errors="coerce"))
        if is_numeric and not name_score:
            parse_score = 0.0
        else:
            try:
                parsed = pd.to_datetime(df[c], errors="coerce")
                parse_score = float(parsed.notna().mean())
            except Exception:
                parse_score = 0.0
        if name_score or parse_score >= 0.8:
            scored.append((name_score, parse_score, c))
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [c for _, _, c in scored]


def _metric_card(title: str, value: str, caption: str, highlight: bool = False) -> None:
    bg = "#F5F5FD" if highlight else "#FFFFFF"
    border = f"2px solid {CLR_ACCENT}" if highlight else "1px solid #E2E8F0"
    value_color = CLR_ACCENT if highlight else "#0F172A"
    st.markdown(f"""
<div style="background:{bg}; border:{border}; border-radius:8px; padding:16px 18px; min-height:116px;">
  <div style="font-size:0.86rem; font-weight:700; color:#64748B; margin-bottom:8px;">{title}</div>
  <div style="font-size:1.85rem; line-height:1.15; font-weight:800; color:{value_color}; margin-bottom:10px;">{value}</div>
  <div style="font-size:0.82rem; color:#64748B;">{caption}</div>
</div>
""", unsafe_allow_html=True)


def _format_timedelta(delta: pd.Timedelta) -> str:
    seconds = int(delta.total_seconds())
    if seconds % 86400 == 0:
        return f"{seconds // 86400}일"
    if seconds % 3600 == 0:
        return f"{seconds // 3600}시간"
    if seconds % 60 == 0:
        return f"{seconds // 60}분"
    return f"{seconds}초"


def _time_gap_summary(raw_df: pd.DataFrame, time_col: str) -> dict:
    parsed = pd.to_datetime(raw_df[time_col], errors="coerce").dropna().drop_duplicates().sort_values()
    observed_index = pd.DatetimeIndex(parsed)
    if len(observed_index) < 2:
        return {"missing": None, "interval": None, "expected": len(observed_index), "observed": len(observed_index)}

    diffs = observed_index.to_series().diff().dropna()
    interval = diffs.mode().iloc[0] if not diffs.mode().empty else diffs.median()
    if pd.isna(interval) or interval <= pd.Timedelta(0):
        return {"missing": None, "interval": None, "expected": len(parsed), "observed": len(parsed)}

    expected_index = pd.date_range(observed_index[0], observed_index[-1], freq=interval)
    missing = max(0, len(expected_index.difference(observed_index)) if len(expected_index) else 0)
    return {
        "missing": int(missing),
        "interval": _format_timedelta(interval),
        "expected": int(len(expected_index)),
        "observed": int(len(parsed)),
    }


@st.cache_data(show_spinner=False)
def _cached_run(file_hash, csv_bytes, time_col, value_cols, label_col,
                train_ratio, lags, detector_quantile, detector_fit_mode,
                use_norm, use_kmeans, use_wasserstein, window, kmeans_k,
                use_lof, use_copod, use_ecod, lof_neighbors,
                agg_method, agg_consensus_level):
    df = pd.read_csv(io.BytesIO(csv_bytes))
    cfg = PipelineConfig(
        time_col=time_col,
        value_cols=list(value_cols),
        label_col=label_col,
        train_ratio=train_ratio,
        lags=lags,
        detector_quantile=detector_quantile,
        detector_fit_mode=detector_fit_mode,
        scorers=ScorerConfig(
            use_norm=use_norm, use_kmeans=use_kmeans, use_wasserstein=use_wasserstein,
            window=window, kmeans_k=kmeans_k,
            use_lof=use_lof, use_copod=use_copod, use_ecod=use_ecod,
            lof_neighbors=lof_neighbors,
        ),
        aggregator=AggregatorConfig(method=agg_method, consensus_level=agg_consensus_level),
    )
    return run_pipeline(df, cfg)


st.sidebar.title("🫐 분석 설정")
st.sidebar.divider()

st.sidebar.markdown("**🗂 데이터 파일**")
uploaded_file = st.sidebar.file_uploader(
    "CSV 업로드", type=["csv"], label_visibility="collapsed",
    help="시간 컬럼 1개 + 수치형 값 컬럼 여러 개(다변량). (선택) 정답 라벨 컬럼.")

preview_df = None
if uploaded_file is not None:
    try:
        csv_bytes_preview = uploaded_file.getvalue()
        preview_df = pd.read_csv(io.BytesIO(csv_bytes_preview))
    except Exception as e:
        st.sidebar.error(f"CSV를 읽을 수 없습니다: {e}")

if preview_df is not None:
    cols = list(preview_df.columns)
    time_candidates = _time_col_candidates(preview_df)
    if not time_candidates:
        time_candidates = cols
    guess_time = _guess_time_col(preview_df)

    st.sidebar.divider()
    st.sidebar.markdown("**🧭 컬럼 매핑**")
    time_col = st.sidebar.selectbox(
        "시간 컬럼", time_candidates, index=time_candidates.index(guess_time),
        help="시계열의 시간축으로 사용할 컬럼입니다.")

    numeric_cols = [c for c in cols if c != time_col and pd.api.types.is_numeric_dtype(
        pd.to_numeric(preview_df[c], errors="coerce"))]
    default_vals = [c for c in numeric_cols if c.lower() not in
                    ("is_anomaly", "anomaly", "label", "y", "target")]
    value_cols = st.sidebar.multiselect(
        "값 컬럼 (다변량)", numeric_cols, default=default_vals or numeric_cols,
        help="이상탐지에 사용할 수치형 변수입니다. 여러 개를 선택하면 다변량으로 분석합니다.")

    label_options = ["(없음)"] + [c for c in cols if c != time_col and c not in value_cols]
    label_choice = st.sidebar.selectbox(
        "정답 라벨 컬럼 (선택)", label_options,
        help="이상=1, 정상=0 컬럼이 있으면 AUC-ROC 등 정량 평가를 제공합니다.")
    label_col = None if label_choice == "(없음)" else label_choice

    st.sidebar.divider()
    st.sidebar.markdown("**⚙️ 파이프라인 파라미터**")
    train_ratio = st.sidebar.slider(
        "학습 구간 비율", 0.3, 0.8, 0.5, 0.05,
        help="전체 데이터 중 학습에 쓸 비율. 나머지는 테스트(평가)에 사용합니다.")
    lags = st.sidebar.number_input(
        "예측 모델 lag (과거 시점 수)", min_value=4, max_value=2000, value=48, step=4,
        help="강의 예시: 1주 = 7x24x2(30분 빈도). 데이터 빈도에 맞춰 설정.")
    detector_quantile = st.sidebar.slider(
        "Detector 임계 분위수", 0.80, 0.999, 0.97, 0.005,
        help="이상 점수가 이 분위수를 넘으면 이상으로 표시. 높일수록 보수적(탐지 적음).")
    detector_fit_mode_label = st.sidebar.radio(
        "Detector 학습 방식",
        ["테스트 구간 기준(기본, 민감도 높음)", "학습 구간 기준(보수적, 오탐 적음)"],
        index=0,
        help="기본값은 테스트 점수 분포로 임계값을 정해 민감하게 탐지합니다. "
            "학습 구간 기준은 모델이 학습한 정상 데이터의 점수 분포로 임계값을 정해 "
            "더 보수적이지만, '테스트 데이터를 보고 임계값을 정했다'는 비판에서 자유롭습니다.")
    detector_fit_mode = "test" if detector_fit_mode_label.startswith("테스트") else "train"

    st.sidebar.divider()
    st.sidebar.markdown("**📡 Scorer — Darts 기본**")
    use_norm = st.sidebar.checkbox(
        "NormScorer (예측오차 크기)", value=True,
        help="예측값과 실제값의 차이 크기를 이상 점수로 사용합니다.")
    use_kmeans = st.sidebar.checkbox(
        "KMeansScorer (오차 패턴 군집)", value=True,
        help="예측오차 패턴을 군집화해 낯선 패턴을 이상으로 봅니다.")
    if use_kmeans:
        kmeans_k = st.sidebar.number_input(
            "KMeans 군집 수 k", 2, 20, 2, 1,
            help="KMeansScorer가 오차 패턴을 나눌 군집 개수입니다.")
    else:
        kmeans_k = 2
    use_wasserstein = st.sidebar.checkbox(
        "WassersteinScorer (분포 변화)", value=True,
        help="윈도우 단위의 분포 변화가 클수록 높은 이상 점수를 줍니다.")

    with st.sidebar.expander("🔬 고급 설정 (Scorer 세부조정)", expanded=False):
        estimated_test_len = int(len(preview_df) * (1 - train_ratio))
        max_window = max(2, estimated_test_len - int(lags) - 1)
        max_window = max(2, min(500, max_window))
        safe_default_window = max(2, min(12, max_window))
        window = st.number_input(
            "Scorer 윈도우 (KMeans/Wasserstein/PyOD)",
            min_value=2,
            max_value=max_window,
            value=safe_default_window,
            step=1,
            help="Scorer가 한 번에 비교할 시점 수입니다. 데이터 길이보다 크면 오류가 나므로 자동으로 제한됩니다.")
        st.caption("`PyODScorer`로 PyOD 비지도 탐지기를 Darts Scorer로 감싼 것. "
                  "예측오차가 아니라 윈도우 내 분포 자체로 이상을 판단해 보완적.")
        if not is_pyod_available():
            st.warning("PyOD 라이브러리를 불러올 수 없어 LOF/COPOD/ECOD Scorer는 비활성화됩니다. "
                      "Darts 기본 Scorer(NormScorer/WassersteinScorer)는 정상 동작합니다.")
        use_lof = st.checkbox(
            "LOF (국소 밀도 기반)", value=False, disabled=not is_pyod_available(),
            help="주변 시점과 비교해 밀도가 낮은 구간을 이상으로 판단합니다.")
        lof_neighbors = st.number_input("LOF 이웃 수", 5, 100, 20, 5,
                                         disabled=not (use_lof and is_pyod_available()),
                                         help="LOF가 국소 밀도를 계산할 때 참고할 이웃 수입니다.")
        use_copod = st.checkbox(
            "COPOD (분포 기반, 빠름)", value=False, disabled=not is_pyod_available(),
            help="변수 분포의 꼬리 확률을 이용해 이상 가능성을 계산합니다.")
        use_ecod = st.checkbox(
            "ECOD (경험적 CDF, 해석 쉬움)", value=False, disabled=not is_pyod_available(),
            help="경험적 누적분포 기준으로 극단적인 값을 이상으로 판단합니다.")

    st.sidebar.divider()
    st.sidebar.markdown("**🧩 Aggregator — 결과 통합**")
    agg_method_label = st.sidebar.radio(
        "통합 방식",
        ["다수결(합의 수준)", "OR (하나라도 이상)", "AND (전부 동의)"],
        index=0,
        help="다수결은 Darts 표준 Or/And를 일반화한 합의 Aggregator. "
            "consensus_level=1.0이면 AND와 동일, 작을수록 OR에 가까워집니다.")
    _agg_map = {"다수결(합의 수준)": "majority", "OR (하나라도 이상)": "or", "AND (전부 동의)": "and"}
    agg_method = _agg_map[agg_method_label]
    agg_consensus_level = 0.5
    if agg_method == "majority":
        agg_consensus_level = st.sidebar.slider(
            "합의 수준 (consensus_level)", 0.1, 1.0, 0.5, 0.05,
            help="전체 Scorer 중 이 수준 이상이 동의해야 이상으로 확정. 0.5=과반.")
else:
    cols = value_cols = []
    time_col = label_col = None
    train_ratio, lags, detector_quantile = 0.5, 48, 0.97
    detector_fit_mode = "test"
    use_norm = use_wasserstein = True
    use_kmeans = use_lof = use_copod = use_ecod = False
    window, kmeans_k, lof_neighbors = 24, 2, 20
    agg_method, agg_consensus_level = "majority", 0.5


st.title("🫐 다변량 시계열 이상탐지 대시보드")
st.markdown("CSV 기반 시계열 데이터를 자동 분석하여 이상 탐지, 통합 판정, 성능 평가, 결과 다운로드까지 제공합니다!")
st.divider()


if uploaded_file is None:
    st.markdown(f"""
    <div style="text-align: center; padding: 60px 20px; background-color: #F5F5FD; border: 2px dashed {CLR_ACCENT_LIGHT}; border-radius: 12px; margin-top: 20px;">
        <h2 style="color: {CLR_ACCENT}; margin-bottom: 15px;">시계열 이상탐지 대시보드에 오신 것을 환영합니다!</h2>
        <p style="color: #64748B; font-size: 16px; line-height: 1.6;">
            왼쪽 사이드바에서 <b>다변량 시계열 CSV 파일</b>을 업로드하여 분석을 시작해 보세요.<br>
            전처리부터 이상탐지, 평가 대시보드까지의 파이프라인이 자동으로 실행됩니다!
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    st.markdown("## 프로젝트 주요 흐름")
    flow_cols = st.columns(3)
    flow_cards = [
        (
            "1. 데이터 준비",
            "CSV 업로드 -> 컬럼 선택 -> 결측치 보간 -> TimeSeries 변환 -> 스케일링",
            "시간, 값, 라벨 컬럼을 고르고 분석 가능한 모델 입력 형태로 정리합니다.",
        ),
        (
            "2. 이상탐지 및 통합 판정",
            "Darts/PyOD Scorer -> 이상 점수 계산 -> Detector 변환 -> Aggregator 통합",
            "여러 Scorer의 결과를 OR, AND, 다수결 방식 중 선택한 기준으로 합칩니다.",
        ),
        (
            "3. 평가 및 결과 활용",
            "성능 평가 -> 수동 보정 -> 최종 결과 확인 -> CSV 다운로드",
            "라벨 유무에 맞춰 결과를 평가하고, 필요한 보정 후 CSV로 내려받습니다.",
        ),
    ]
    for col, (title, flow, desc) in zip(flow_cols, flow_cards):
        with col:
            st.markdown(f"""
<div style="height:250px; min-height:250px; padding:24px 26px; border:1px solid #E2E8F0; border-radius:12px; background:#FFFFFF;">
  <div style="font-size:18px; font-weight:800; color:#0F172A; margin-bottom:18px;">{title}</div>
  <div style="font-size:16px; font-weight:800; line-height:1.65; color:{CLR_ACCENT}; word-break:keep-all; margin-bottom:18px;">{flow}</div>
  <div style="font-size:15px; line-height:1.65; color:#64748B; word-break:keep-all;">{desc}</div>
</div>
""", unsafe_allow_html=True)

    st.stop()

if preview_df is None or not value_cols:
    st.warning("값 컬럼을 1개 이상 선택하세요. (사이드바 '컬럼 매핑' 확인)")
    st.stop()


csv_bytes = uploaded_file.getvalue()
file_hash = _hash_bytes(csv_bytes)

if "manual_overrides" not in st.session_state:
    st.session_state.manual_overrides = {}

with st.spinner("이상탐지 파이프라인 실행 중..."):
    try:
        out = _cached_run(
            file_hash, csv_bytes, time_col, tuple(value_cols), label_col,
            train_ratio, int(lags), detector_quantile, detector_fit_mode,
            use_norm, use_kmeans, use_wasserstein, int(window), int(kmeans_k),
            use_lof, use_copod, use_ecod, int(lof_neighbors),
            agg_method, agg_consensus_level,
        )
    except Exception as e:
        st.error(f"파이프라인 오류: {e}")
        st.stop()

meta = out["meta"]
results = out["results"]
diag = out["diagnostics"]
scorer_names = meta["scorer_names"]

overrides_for_file = st.session_state.manual_overrides.setdefault(file_hash, {})
final_aggregated = out["aggregated"]
if final_aggregated is not None:
    final_aggregated = apply_manual_overrides(out["aggregated"], overrides_for_file)


with st.container(border=True):
    st.markdown("### 📋 1. 데이터 개요")

    agg_rate = diag.get("aggregated_anomaly_rate", 0.0)
    label_status = "있음" if meta["has_labels"] else "없음"
    label_caption = "성능 평가 가능" if meta["has_labels"] else "라벨-프리 진단 사용"

    row1 = st.columns(3)
    with row1[0]:
        _metric_card("관측치", f"{meta['n_total']:,}", "전체 시계열 데이터 수")
    with row1[1]:
        _metric_card("변수 수", f"{meta['n_components']:,}", "분석 대상 컬럼 수")
    with row1[2]:
        _metric_card("측정 간격", meta["freq"], "데이터 수집 주기")

    row2 = st.columns(3)
    with row2[0]:
        _metric_card("테스트 구간", f"{meta['n_test']:,}", "평가 대상 데이터 수")
    with row2[1]:
        _metric_card("통합 이상 비율", f"{agg_rate*100:.1f}%", "최종 Aggregator 기준", highlight=True)
    with row2[2]:
        _metric_card("라벨 여부", label_status, label_caption)

    t_prev, t_stat, t_missing, t_adf, t_viz = st.tabs(
        ["미리보기", "기초 통계", "결측치 현황", "정상성 검정 (ADF)", "시계열 시각화"])

    with t_prev:
        st.dataframe(out["clean_df"].head(20), use_container_width=True)

    with t_stat:
        st.dataframe(out["clean_df"][value_cols].describe().T, use_container_width=True)

    with t_missing:
        st.markdown("## 결측치 현황")
        st.caption("업로드된 원본 CSV를 기준으로 값 결측치와 timestamp 누락 여부를 확인합니다.")

        miss_df = missing_value_summary(preview_df, value_cols)
        raw_nan_total = int(miss_df["n_missing"].sum())
        gap = _time_gap_summary(preview_df, time_col)
        inserted_timestamps = max(
            0,
            len(out["series"].time_index) - len(
                pd.to_datetime(preview_df[time_col], errors="coerce").dropna().drop_duplicates()
            ),
        )
        gap_missing = 0 if gap["missing"] is None else int(gap["missing"])
        value_missing = "없음" if raw_nan_total == 0 else f"{raw_nan_total:,}건"
        time_missing = "계산 불가" if gap["missing"] is None else ("없음" if gap_missing == 0 else f"{gap_missing:,}개")
        observed_ts = f"{gap['observed']:,}개"
        expected_ts = "계산 불가" if gap["missing"] is None else f"{gap['expected']:,}개"
        needs_nan_interp = raw_nan_total > 0
        needs_time_fix = gap_missing > 0 or inserted_timestamps > 0

        st.markdown(f"""
### 원본 데이터 점검
- 값 결측치: {value_missing}
- timestamp 누락: {time_missing}
- 원본 timestamp: {observed_ts}
- 기대 timestamp: {expected_ts}

### 전처리 보정 여부
- NaN 보간 필요 여부: {"필요" if needs_nan_interp else "없음"}
- timestamp 보정 필요 여부: {"필요" if needs_time_fix else "없음"}
""")

        if not needs_nan_interp and not needs_time_fix:
            st.success("선택한 값 컬럼의 NaN 결측치와 timestamp 누락은 확인되지 않았습니다. 전처리 과정에서 추가 보간이 필요한 구간은 없습니다.")
        else:
            messages = []
            if needs_nan_interp:
                messages.append("선택한 값 컬럼에 NaN 결측치가 확인되었습니다. 전처리 과정에서 선형 보간을 적용하여 모델 입력 형태로 변환합니다.")
            if needs_time_fix:
                messages.append("일부 timestamp 구간이 누락되었습니다. 시계열 변환 과정에서 일정한 수집 간격에 맞춰 시간축을 보정합니다.")
            st.warning(" ".join(messages))

    with t_adf:
        st.caption(
            "ADF(Augmented Dickey-Fuller) 검정으로 각 변수의 정상성을 확인합니다. "
            "p-value < 0.05면 정상(stationary) 시계열로 판단합니다. "
            "비정상 시계열에 대한 예측 기반 Scorer(NormScorer 등)의 결과는 추세/계절성에 의한 "
            "잡음을 더 포함할 수 있으니 해석 시 참고하세요."
        )
        adf_df = stationarity_report(out["clean_df"], value_cols)
        cols_adf = st.columns(len(adf_df))
        for col_el, (_, row) in zip(cols_adf, adf_df.iterrows()):
            badge_color = CLR_OK if row["is_stationary"] else CLR_ANOMALY
            label = "정상" if row["is_stationary"] else "비정상"
            col_el.markdown(f"""
<div style="border:1px solid #E2E8F0; border-radius:10px; padding:0.8rem; text-align:center;">
  <div style="font-size:0.85rem; color:#64748B; margin-bottom:4px">{row['column']}</div>
  <div style="font-size:1.3rem; font-weight:700; color:{badge_color}">{label}</div>
  <div style="font-size:0.75rem; color:#94A3B8;">p={row['p_value']:.4f}</div>
</div>""", unsafe_allow_html=True)
        st.dataframe(adf_df, use_container_width=True, hide_index=True)

    with t_viz:
        st.caption(
            "일부 구간이 직선으로 이어지는 경우, 원본 데이터에 이미 보간된 값이 포함되어 있거나 "
            "시각화 과정에서 인접 시점이 선으로 연결된 결과일 수 있습니다. 결측 여부는 결측치 현황 탭의 "
            "원본 NaN 및 timestamp 누락 기준으로 확인하세요."
        )
        series = out["series"]
        ts_start, ts_end = out["test_index_range"]
        for col in value_cols:
            comp = series[col]
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=comp.time_index, y=comp.values().ravel(),
                mode="lines", name=col, line=dict(color=CLR_BLUE, width=1.3),
                fill="tozeroy", fillcolor="rgba(29,78,216,0.05)"))
            fig.add_vrect(x0=ts_start, x1=ts_end, fillcolor="gray",
                          opacity=0.08, line_width=0, annotation_text="테스트 구간",
                          annotation_position="top left")
            fig.update_layout(template="plotly_white", height=220,
                              margin=dict(l=10, r=10, t=30, b=10),
                              title=col, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        if len(value_cols) >= 2:
            st.markdown("#### 변수 간 상관관계")
            st.caption(
                "상관계수는 두 변수가 함께 움직이는 정도를 나타냅니다. "
                "정상적으로 강하게 연결된 변수 관계가 갑자기 달라지면 다변량 이상 신호로 볼 수 있습니다."
            )
            corr = correlation_matrix(out["clean_df"], value_cols)
            fig_corr = go.Figure(data=go.Heatmap(
                z=corr.values, x=corr.columns.tolist(), y=corr.columns.tolist(),
                text=corr.round(2).values, texttemplate="%{text}",
                colorscale="RdBu", zmid=0, zmin=-1, zmax=1))
            fig_corr.update_layout(template="plotly_white", height=320,
                                   margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig_corr, use_container_width=True)

st.divider()


with st.container(border=True):
    st.markdown("### 🔍 2. 이상탐지 결과")
    st.caption("Scorer별 이상 점수와 Detector가 변환한 이상 구간을 시각화합니다.")

    sel = st.selectbox(
        "Scorer 선택", scorer_names, key="scorer_sel",
        help="상세 차트에 표시할 이상 점수 계산 방식을 고릅니다.")
    st.caption("비교할 Scorer를 선택하면 원본 값, 이상 점수, 이진 탐지 결과를 함께 확인할 수 있습니다.")
    e = results[sel]
    score_uni = e["score_uni"]
    binary = e["binary"]

    show_col = value_cols[0]
    base = out["series"][show_col].slice_intersect(score_uni)

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        row_heights=[0.45, 0.3, 0.25], vertical_spacing=0.04,
                        subplot_titles=("", f"{sel} 이상 점수", "이상 탐지"))
    fig.add_trace(go.Scatter(x=base.time_index, y=base.values().ravel(),
                             mode="lines", line=dict(color=CLR_BLUE, width=1),
                             name=show_col), row=1, col=1)

    bvals = binary.slice_intersect(base).values().ravel().astype(int)
    btime = binary.slice_intersect(base).time_index
    anom_t = btime[bvals == 1]
    anom_y = base.slice_intersect(binary).values().ravel()[bvals == 1]
    fig.add_trace(go.Scatter(x=anom_t, y=anom_y, mode="markers",
                             marker=dict(color=CLR_ANOMALY, size=5),
                             name="탐지된 이상"), row=1, col=1)

    fig.add_trace(go.Scatter(x=score_uni.time_index, y=score_uni.values().ravel(),
                             mode="lines", line=dict(color="#8a6d3b", width=1),
                             name="이상 점수"), row=2, col=1)
    thr = np.quantile(score_uni.values().ravel(), detector_quantile)
    fig.add_trace(go.Scatter(
        x=score_uni.time_index,
        y=np.full(len(score_uni.time_index), thr),
        mode="lines",
        line=dict(color=CLR_ANOMALY, width=1, dash="dash"),
        name="이상 기준",
    ), row=2, col=1)

    fig.add_trace(go.Scatter(x=btime, y=bvals, mode="lines",
                             line=dict(color=CLR_ANOMALY, width=1.2, shape="hv"),
                             name="이상 여부"), row=3, col=1)

    if meta["has_labels"]:
        lab = out["test_labels"].slice_intersect(binary)
        fig.add_trace(go.Scatter(x=lab.time_index, y=lab.values().ravel(),
                                 mode="lines", line=dict(color=CLR_OK, width=1, dash="dot"),
                                 name="정답 라벨"), row=3, col=1)

    chart_title = f"{show_col} (원본/라벨)" if meta["has_labels"] else f"{show_col} (원본)"
    fig.update_layout(
        template="plotly_white",
        height=680,
        title=dict(
            text=chart_title,
            x=0.5,
            y=0.96,
            xanchor="center",
            yanchor="top",
            font=dict(size=16, color="#6B7280"),
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.08,
            xanchor="left",
            x=0,
        ),
        margin=dict(t=110, b=60, l=60, r=30),
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("📖 이 차트를 어떻게 해석하나요?"):
        st.markdown(f"""
- **1단(원본)**: 선택한 변수의 실제 값입니다. 빨간 점은 이 Scorer가 이상으로 표시한 시점입니다.
- **2단(이상 점수)**: 예측값과 실제값의 차이(또는 분포 거리)가 클수록 점수가 높습니다.
  점선은 현재 설정된 임계 분위수({detector_quantile:.2%})에 해당하는 점수입니다.
- **3단(이상 탐지)**: 점수가 임계값을 넘으면 1(이상), 아니면 0(정상)으로 이진화한 결과입니다.
- **판단 기준**: 점수가 짧고 뾰족하게 튀는 구간일수록 신뢰도가 높습니다. 점수가 계속
  임계값 근처에서 오르내리면 그 구간은 오탐 가능성이 있으니 아래 "수동 보정"에서
  직접 확인·정정하세요.
""")

    if out["aggregated"] is not None and len(scorer_names) >= 2:
        ai = out["aggregator_info"]
        method_label = {"or": "OrAggregator", "and": "AndAggregator",
                        "majority": "MajorityVoteAggregator"}[ai["method"]]
        st.markdown(f"#### 통합 이상 플래그 ({method_label})")
        if ai["method"] == "majority":
            st.caption(
                f"전체 {ai['n_voters']}개 Scorer 중 **{ai['k']}개 이상**(합의 수준 {ai['consensus_level']:.2f}) "
                f"동의해야 이상으로 확정합니다. (consensus_level=1.0 -> AND와 동일, 작을수록 OR에 근접)"
            )
        elif ai["method"] == "and":
            st.caption(f"전체 {ai['n_voters']}개 Scorer가 **모두** 동의해야 이상으로 확정합니다.")
        else:
            st.caption(f"{ai['n_voters']}개 Scorer 중 **하나라도** 이상이라 하면 이상으로 확정합니다.")
        agg = out["aggregated"]
        figa = go.Figure()
        figa.add_trace(go.Scatter(x=agg.time_index, y=agg.values().ravel(),
                                  mode="lines", line=dict(color=CLR_ANOMALY, width=1.2, shape="hv"),
                                  name="통합 이상"))
        figa.update_layout(template="plotly_white", height=180,
                           margin=dict(l=10, r=10, t=10, b=10),
                           yaxis=dict(tickvals=[0, 1], ticktext=["정상", "이상"]))
        st.plotly_chart(figa, use_container_width=True)

    if final_aggregated is not None:
        st.markdown("#### 수동 보정 (클릭으로 이상 플래그 직접 수정)")
        st.caption(
            "아래 차트에서 점을 클릭하거나 드래그로 선택한 뒤, 원하는 보정을 적용하세요. "
            "보정 결과는 자동 탐지와 구분되어 표시되고 다운로드 CSV에도 반영됩니다."
        )
        if st.session_state.get("_last_override_count"):
            st.caption(f"가장 최근 작업: {st.session_state['_last_override_count']}개 시점 보정 적용됨")
        corr_base = out["series"][show_col].slice_intersect(final_aggregated)
        corr_time = final_aggregated.slice_intersect(corr_base).time_index
        corr_flags = final_aggregated.slice_intersect(corr_base).values().ravel().astype(int)
        corr_vals = corr_base.slice_intersect(final_aggregated).values().ravel()

        removed_ts = {ts for ts, v in overrides_for_file.items() if v == 0}
        added_ts = {ts for ts, v in overrides_for_file.items() if v == 1}
        is_removed = np.array([t in removed_ts for t in corr_time])
        is_added = np.array([t in added_ts for t in corr_time])
        is_detected = (corr_flags == 1) & ~is_added

        figc = go.Figure()
        figc.add_trace(go.Scatter(
            x=corr_time, y=corr_vals, mode="lines",
            customdata=[str(t) for t in corr_time],
            line=dict(color=CLR_BLUE, width=1), name=show_col,
        ))
        figc.add_trace(go.Scatter(
            x=corr_time, y=corr_vals, mode="markers",
            customdata=[str(t) for t in corr_time],
            marker=dict(color="rgba(100, 100, 240, 0.22)", size=5),
            name="선택 가능 시점",
            showlegend=False,
            hovertemplate="%{x}<extra></extra>",
        ))
        figc.add_trace(go.Scatter(
            x=corr_time[is_detected], y=corr_vals[is_detected], mode="markers",
            customdata=[str(t) for t in corr_time[is_detected]],
            marker=dict(color=CLR_ANOMALY, size=6), name="탐지된 이상",
        ))
        figc.add_trace(go.Scatter(
            x=corr_time[is_removed], y=corr_vals[is_removed], mode="markers",
            customdata=[str(t) for t in corr_time[is_removed]],
            marker=dict(color=CLR_SLATE, size=7, symbol="x", opacity=0.6), name="이상 해제(수동)",
        ))
        figc.add_trace(go.Scatter(
            x=corr_time[is_added], y=corr_vals[is_added], mode="markers",
            customdata=[str(t) for t in corr_time[is_added]],
            marker=dict(color=CLR_OK, size=8, symbol="star"), name="이상 추가(수동)",
        ))
        figc.update_layout(template="plotly_white", height=320,
                           margin=dict(l=10, r=10, t=30, b=10),
                           legend=dict(orientation="h", y=1.12))
        selected_data = st.plotly_chart(
            figc, use_container_width=True, on_select="rerun", key="manual_correction_chart"
        )

        sel_timestamps = []
        if selected_data:
            sel_points = selected_data.get("selection", {}).get("points", [])
            for p in sel_points:
                ts_raw = p.get("customdata")
                if ts_raw is not None:
                    if isinstance(ts_raw, (list, tuple, np.ndarray)):
                        ts_raw = ts_raw[0] if len(ts_raw) else None
                if ts_raw is not None:
                    sel_timestamps.append(pd.Timestamp(ts_raw))
        sel_timestamps = sorted(set(sel_timestamps))

        cor1, cor2 = st.columns([2, 1])
        with cor1:
            action = st.radio(
                "선택한 시점을", ["이상 해제(정상으로)", "이상으로 추가"],
                horizontal=True, key="override_action",
                help="선택한 시점의 최종 통합 이상 플래그를 수동으로 바꿉니다.",
            )
        with cor2:
            st.caption(f"{len(sel_timestamps)}개 시점 선택됨" if sel_timestamps
                       else "차트에서 점을 선택하세요.")
        if st.button("선택 적용", type="primary", help="선택한 시점에 현재 보정 방식을 적용합니다."):
            if not sel_timestamps:
                st.warning("먼저 차트에서 점을 클릭하거나 드래그로 선택하세요.")
            else:
                new_flag = 0 if action == "이상 해제(정상으로)" else 1
                for ts in sel_timestamps:
                    overrides_for_file[ts] = new_flag
                st.session_state["_last_override_count"] = len(sel_timestamps)
                st.success(f"{len(sel_timestamps)}개 시점에 보정을 적용했습니다.")
                st.rerun()

        if overrides_for_file:
            st.caption(f"수동 보정 {len(overrides_for_file)}건 적용 중")
            st.dataframe(
                override_diff_summary(out["aggregated"], overrides_for_file),
                use_container_width=True, hide_index=True,
            )
        if st.button(
            "수동 보정 전체 초기화",
            disabled=not overrides_for_file,
            help="현재 파일에 저장된 수동 보정을 모두 지웁니다.",
        ):
            st.session_state.manual_overrides[file_hash] = {}
            st.session_state.pop("_last_override_count", None)
            st.rerun()

st.divider()


with st.container(border=True):
    st.markdown("### 📊 3. 평가 대시보드")

    if meta["has_labels"]:
        st.caption("정답 라벨이 있는 경우, Scorer별 이상탐지 성능을 정량 평가합니다.")
        st.caption(
            "AUC-ROC와 AUC-PR은 이상 점수의 분리 성능을 평가하고, "
            "Precision·Recall·F1은 Detector가 변환한 최종 이상 판정의 참고 지표로 제공합니다."
        )

        rows = []
        for nm, e in results.items():
            m = e["metrics"]
            rows.append({
                "Scorer": nm, "AUC-ROC": m["AUC_ROC"], "AUC-PR": m["AUC_PR"],
                "Precision": m["precision"], "Recall": m["recall"], "F1": m["f1"],
                "탐지수": m["n_pred_anomaly"], "실제이상": m["n_true_anomaly"],
            })
        mdf = pd.DataFrame(rows).set_index("Scorer")

        best_scorer = mdf["AUC-ROC"].idxmax()
        score_cols = st.columns(len(mdf))
        for col_el, (nm, row) in zip(score_cols, mdf.iterrows()):
            is_best = nm == best_scorer
            border = f"2px solid {CLR_ACCENT}" if is_best else "1px solid #E2E8F0"
            badge = "  ⭐" if is_best else ""
            col_el.markdown(f"""
<div style="border:{border}; border-radius:10px; padding:0.8rem; text-align:center;">
  <div style="font-size:0.85rem; color:#64748B; margin-bottom:4px">{nm}{badge}</div>
  <div style="font-size:1.5rem; font-weight:700; color:{CLR_ACCENT if is_best else '#1E293B'}">{row['AUC-ROC']:.3f}</div>
  <div style="font-size:0.75rem; color:#94A3B8; margin-bottom:4px">AUC-ROC</div>
  <div style="font-size:0.85rem">F1 {row['F1']:.3f} · AUC-PR {row['AUC-PR']:.3f}</div>
</div>""", unsafe_allow_html=True)

        st.markdown("###")
        t_table, t_roc, t_pr, t_cm = st.tabs(["지표 요약", "ROC 곡선", "PR 곡선", "혼동행렬"])

        with t_table:
            df_display = mdf.reset_index()
            df_display["Scorer"] = df_display["Scorer"].apply(
                lambda m: f"⭐ {m}" if m == best_scorer else m)
            st.dataframe(df_display.style.format({
                "AUC-ROC": "{:.3f}", "AUC-PR": "{:.3f}",
                "Precision": "{:.3f}", "Recall": "{:.3f}", "F1": "{:.3f}",
            }), use_container_width=True, hide_index=True)
            st.info(
                f"⭐ 분석 결과, **{best_scorer}**가 AUC-ROC {mdf.loc[best_scorer,'AUC-ROC']:.3f}로 "
                f"가장 우수한 이상/정상 분리 성능을 보였습니다. 이상은 보통 드물어 **AUC-PR**과 "
                f"**F1**이 실제 유용성을 더 잘 반영하니 함께 참고하세요."
            )

        with t_roc:
            st.caption("FPR(오탐률) 대비 TPR(탐지율)의 변화를 보여줍니다. 곡선이 왼쪽 위에 붙을수록 좋습니다.")
            figroc = go.Figure()
            for nm, e in results.items():
                fpr, tpr = e["curves"]["roc"]
                figroc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=nm))
            figroc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                        line=dict(dash="dash", color=CLR_SLATE), name="랜덤"))
            figroc.update_layout(template="plotly_white", title="ROC 곡선", height=360,
                                 xaxis_title="FPR", yaxis_title="TPR",
                                 margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(figroc, use_container_width=True)

        with t_pr:
            st.caption("Recall(탐지율) 대비 Precision(정밀도)의 변화를 보여줍니다. 이상이 드문 데이터에선 ROC보다 더 신뢰할 수 있는 지표입니다.")
            figpr = go.Figure()
            for nm, e in results.items():
                rec, prec = e["curves"]["pr"]
                figpr.add_trace(go.Scatter(x=rec, y=prec, mode="lines", name=nm))
            figpr.update_layout(template="plotly_white", title="Precision-Recall 곡선", height=360,
                                xaxis_title="Recall", yaxis_title="Precision",
                                margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(figpr, use_container_width=True)

        with t_cm:
            st.caption("실제/예측 조합별 시점 수입니다. 우측 하단(실제 이상·예측 이상)이 클수록 탐지를 잘한 것입니다.")
            sel_cm = st.selectbox(
                "혼동행렬 Scorer", list(results.keys()), key="cm_sel",
                help="혼동행렬로 확인할 Scorer를 선택합니다.")
            cm = results[sel_cm]["metrics"]["confusion_matrix"]
            figcm = go.Figure(data=go.Heatmap(
                z=cm, x=["예측: 정상", "예측: 이상"], y=["실제: 정상", "실제: 이상"],
                text=cm, texttemplate="%{text}", colorscale="Blues", showscale=False))
            figcm.update_layout(template="plotly_white", title=f"혼동행렬 — {sel_cm}", height=360,
                                margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(figcm, use_container_width=True)

    else:
        st.caption("정답 라벨이 없어 **라벨-프리 진단 지표**로 탐지 결과의 적절성을 판단합니다.")

        prows = []
        for nm, d in diag["per_scorer"].items():
            prows.append({
                "Scorer": nm, "이상비율": d["anomaly_rate"], "탐지수": d["n_anomaly"],
                "점수평균": d["score_mean"], "점수표준편차": d["score_std"],
                "점수p95": d["score_p95"], "점수최대": d["score_max"],
            })
        pdf = pd.DataFrame(prows)

        diag_cols = st.columns(len(pdf))
        for col_el, (_, row) in zip(diag_cols, pdf.iterrows()):
            col_el.markdown(f"""
<div style="border:1px solid #E2E8F0; border-radius:10px; padding:0.8rem; text-align:center;">
  <div style="font-size:0.85rem; color:#64748B; margin-bottom:4px">{row['Scorer']}</div>
  <div style="font-size:1.5rem; font-weight:700; color:#1E293B">{row['이상비율']*100:.1f}%</div>
  <div style="font-size:0.75rem; color:#94A3B8;">이상 비율 ({row['탐지수']:.0f}건)</div>
</div>""", unsafe_allow_html=True)

        st.markdown("###")
        t_table2, t_dist, t_sweep, t_agree = st.tabs(
            ["진단 요약", "점수 분포", "임계 민감도", "Scorer 일치도"])

        with t_table2:
            st.dataframe(pdf.style.format({
                "이상비율": "{:.3f}", "점수평균": "{:.3f}", "점수표준편차": "{:.3f}",
                "점수p95": "{:.3f}", "점수최대": "{:.3f}"}), use_container_width=True, hide_index=True)
            st.info(
                "**판단 가이드** · 이상비율이 비현실적으로 높으면(예: >10%) 임계 분위수를 올리세요. "
                "여러 Scorer의 일치도가 높은 구간일수록 실제 이상일 가능성이 큽니다."
            )

        with t_dist:
            st.caption("Scorer 점수가 어느 범위에 몰려 있는지와 현재 임계값이 어디에 놓이는지 보여줍니다.")
            sel_d = st.selectbox(
                "점수 분포 Scorer", list(results.keys()), key="dist_sel",
                help="점수 분포를 확인할 Scorer를 선택합니다.")
            s = results[sel_d]["score_uni"].values().ravel()
            thr = np.quantile(s, detector_quantile)
            figh = go.Figure()
            figh.add_trace(go.Histogram(x=s, nbinsx=60, marker_color=CLR_ACCENT, opacity=0.8))
            figh.add_vline(x=thr, line=dict(color=CLR_ANOMALY, dash="dash"),
                           annotation_text=f"임계({detector_quantile})")
            figh.update_layout(template="plotly_white", title="이상 점수 분포", height=360,
                               margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(figh, use_container_width=True)

        with t_sweep:
            st.caption("임계 분위수를 바꿀 때 이상으로 표시되는 시점 비율이 얼마나 달라지는지 보여줍니다.")
            sel_s = st.selectbox(
                "임계 민감도 Scorer", list(results.keys()), key="sweep_sel",
                help="임계값 변화에 따른 탐지율을 확인할 Scorer를 선택합니다.")
            sw = threshold_sweep(results[sel_s]["score_uni"])
            figs = go.Figure()
            figs.add_trace(go.Scatter(x=sw["quantile"], y=sw["anomaly_rate"],
                                      mode="lines+markers", line=dict(color=CLR_ANOMALY)))
            figs.add_vline(x=detector_quantile, line=dict(color=CLR_ACCENT, dash="dash"))
            figs.update_layout(template="plotly_white", title="임계 분위수 민감도 (탐지율 변화)", height=360,
                               xaxis_title="분위수", yaxis_title="이상 비율",
                               margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(figs, use_container_width=True)

        with t_agree:
            if diag["agreement_matrix"] is not None:
                st.caption("값이 높을수록 서로 다른 Scorer가 같은 시점을 이상으로 판단 -> 탐지 신뢰도가 높습니다.")
                names = diag["agreement_names"]
                am = diag["agreement_matrix"]
                figj = go.Figure(data=go.Heatmap(
                    z=am, x=names, y=names, text=np.round(am, 2),
                    texttemplate="%{text}", colorscale="Purples", zmin=0, zmax=1))
                figj.update_layout(template="plotly_white", height=360,
                                   margin=dict(l=10, r=10, t=10, b=10))
                st.plotly_chart(figj, use_container_width=True)
            else:
                st.info("Scorer가 1개뿐이라 일치도를 계산할 수 없습니다. 사이드바에서 Scorer를 추가해 보세요.")

st.divider()


with st.container(border=True):
    st.markdown("### ⬇️ 4. 결과 다운로드")
    st.caption("탐지 결과를 CSV로 내려받아 외부에서 검토하거나 보고에 활용할 수 있습니다.")

    frames = {}
    for nm, e in results.items():
        s = e["score_uni"]
        b = e["binary"]
        frames[f"score_{nm}"] = pd.Series(s.values().ravel(), index=s.time_index)
        frames[f"flag_{nm}"] = pd.Series(b.values().ravel().astype(int), index=b.time_index)
    res_df = pd.DataFrame(frames)
    if final_aggregated is not None:
        agg = final_aggregated
        res_df["flag_aggregated"] = pd.Series(
            agg.values().ravel().astype(int), index=agg.time_index)
        res_df["flag_manually_overridden"] = [
            int(ts in overrides_for_file) for ts in res_df.index
        ]
    if meta["has_labels"]:
        lab = out["test_labels"]
        res_df["label"] = pd.Series(lab.values().ravel().astype(int), index=lab.time_index)
    res_df.index.name = "time"
    res_df = res_df.sort_index()

    dl1, dl2 = st.columns(2)
    dl1.metric("총 시점", f"{len(res_df):,}")
    agg_col = "flag_aggregated" if "flag_aggregated" in res_df.columns else f"flag_{scorer_names[0]}"
    dl2.metric("이상 판정 시점", f"{int(res_df[agg_col].sum()):,}")

    st.caption("미리보기 (상위 20행):")
    st.dataframe(res_df.head(20), use_container_width=True)

    csv_out = res_df.to_csv().encode("utf-8-sig")
    st.download_button("결과 CSV 다운로드", csv_out,
                       file_name="anomaly_results.csv", mime="text/csv", type="primary",
                       help="Scorer별 점수, 플래그, 수동 보정 결과를 CSV로 저장합니다.")

st.divider()
