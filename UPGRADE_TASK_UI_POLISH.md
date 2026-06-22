# 작업 지시서 — UI 가시성/일관성 정리 (9개 항목)

`app.py`만 수정한다. 다른 파일은 건드리지 않는다. 단, 5번 항목(freq 표시 가공)은
`anomaly_pipeline.py`에 헬퍼 함수를 추가해야 한다. 각 작업 후 `python3 -m py_compile`로
문법 검사를 한다.

이 지시서는 화면 캡처를 보고 작성되었다. **현재 코드에 "클릭 보정"(수동 보정) 기능이
이미 구현되어 있다는 전제**로 작성했다 — 만약 아직 구현 전이라면 4번 항목은 그 기능을
구현할 때 함께 반영한다.

---

## 1. 위성 이모지(🛰️) → 블루베리 이모지(🫐)로 전체 교체

`app.py`에서 `🛰️`가 나오는 모든 위치를 `🫐`로 바꾼다 (최소 3곳: `st.set_page_config`의
`page_icon`, `st.sidebar.title`, 메인 `st.title`). 문자열 전체를 `grep -n "🛰️"`로 찾아
빠짐없이 교체한다.

---

## 2. "1. 데이터 개요" 메트릭 카드 가시성 개선

### 원인

`meta["freq"]`가 Darts `TimeSeries.freq`의 raw repr(예: `<30 * Minutes>`)을 그대로
표시해 `st.metric`의 좁은 컬럼 폭에서 잘려 보인다(`<30 * ...`).

### 2-1. `anomaly_pipeline.py`에 변환 함수 추가

파일 하단에 추가한다:

```python
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
```

`run_pipeline`의 `meta` 딕셔너리에서 `"freq": str(series.freq)`를
`"freq": humanize_freq(str(series.freq))`로 바꾼다.

### 2-2. `app.py` 수정 — 메트릭 카드 레이아웃

`from anomaly_pipeline import (...)`에 위 함수는 import할 필요 없다(이미 `meta["freq"]`에
가공되어 들어온다). 대신 카드 레이아웃 자체를 5컬럼 `st.columns(5)`에서 **2행 구조**로
바꿔 폭을 넉넉하게 한다:

```python
m1, m2, m3 = st.columns(3)
m1.metric("관측치", f"{meta['n_total']:,}")
m2.metric("변수 수", meta["n_components"])
m3.metric("빈도", meta["freq"])
m4, m5 = st.columns(2)
m4.metric("테스트 구간", f"{meta['n_test']:,}")
if meta["has_labels"]:
    best = max(results.items(), key=lambda kv: kv[1]["metrics"]["AUC_ROC"])
    m5.metric("최고 AUC-ROC", f"{best[1]['metrics']['AUC_ROC']:.3f}", best[0])
else:
    agg_rate = diag.get("aggregated_anomaly_rate", 0.0)
    m5.metric("통합 이상 비율", f"{agg_rate*100:.1f}%")
```

---

## 3. Scorer 3단 차트에 해석 가이드 추가

"섹션 2. 이상탐지 결과"의 메인 3단 패널 차트(원본/이상 점수/이상 탐지, `make_subplots`로
그리는 부분) **바로 다음**, 통합 이상 플래그 섹션 이전에 해석 가이드를 추가한다:

```python
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
```

---

## 4. 수동 보정 "선택 적용" 버튼에 결과 피드백 추가

현재 코드에서 "선택 적용" 버튼(`st.button("✨ 선택 적용", ...)` 또는 동일 기능 버튼)을
누른 직후 `st.rerun()`만 호출되고 화면이 그냥 새로고침되어, 사용자가 보정이 실제로
적용됐는지 확인할 길이 없다. 버튼 클릭 처리 블록에 명시적 피드백을 추가한다:

```python
if st.button("✨ 선택 적용", type="primary"):
    if not selected_timestamps:  # 선택된 시점이 없는 경우의 변수명은 실제 코드에 맞춘다
        st.warning("먼저 차트에서 점을 클릭하거나 드래그로 선택하세요.")
    else:
        # ... 기존 override 적용 로직 ...
        st.session_state["_last_override_count"] = len(selected_timestamps)
        st.success(f"✅ {len(selected_timestamps)}개 시점에 보정을 적용했습니다.")
        st.rerun()
```

`st.rerun()`은 즉시 페이지를 새로고침하므로 `st.success` 메시지가 보일 틈 없이 사라질 수
있다. 이를 보완하기 위해, rerun 이후에도 보정이 적용됐음을 알 수 있게 차트 위쪽에
상시 caption을 추가한다(이미 보정 내역 표가 있다면 그 위에 추가):

```python
if st.session_state.get("_last_override_count"):
    st.caption(f"가장 최근 작업: {st.session_state['_last_override_count']}개 시점 보정 적용됨")
```

---

## 5. 모든 시각화/설정 항목에 `help` 툴팁 추가

사이드바와 본문에서 **`help` 파라미터가 없는** 위젯·시각화 제목에 짧은 설명을 추가한다.
이미 `help`가 있는 위젯(예: lag, detector_quantile 슬라이더)은 건드리지 않는다.
다음 위치를 확인해 빠진 곳에 추가한다:

- 사이드바 `train_ratio` 슬라이더 → `help="전체 데이터 중 학습에 쓸 비율. 나머지는 테스트(평가)에 사용합니다."`
- "Scorer 선택" radio/checkbox 그룹 제목 옆 → `st.markdown` 대신 `st.caption`에 한 줄 설명이 이미 있다면 유지, 없으면 추가
- 평가 대시보드의 ROC/PR 곡선, 혼동행렬 탭 제목 → 탭 안에 `st.caption`으로 한 줄 설명 추가:
  - ROC 곡선: `"FPR(오탐률) 대비 TPR(탐지율)의 변화를 보여줍니다. 곡선이 왼쪽 위에 붙을수록 좋습니다."`
  - PR 곡선: `"Recall(탐지율) 대비 Precision(정밀도)의 변화를 보여줍니다. 이상이 드문 데이터에선 ROC보다 더 신뢰할 수 있는 지표입니다."`
  - 혼동행렬: `"실제/예측 조합별 시점 수입니다. 우측 하단(실제 이상·예측 이상)이 클수록 탐지를 잘한 것입니다."`
- 라벨-프리 진단의 "점수 분포", "임계 민감도", "Scorer 일치도" 탭에 이미 caption이 있다면
  유지하고, 없는 곳만 위와 같은 형식으로 추가한다.

`?` 아이콘 자체는 Streamlit이 `help` 파라미터를 넘기면 자동으로 위젯 옆에 그려주므로,
별도 아이콘을 수동으로 그릴 필요는 없다. `st.subheader`/`st.markdown` 제목처럼 `help`를
받지 않는 요소는 옆에 `st.caption`을 추가하는 방식으로 대체한다.

---

## 6. 파이프라인 설명 중복 제거

현재 화면에 파이프라인 설명(`Forecasting+Scorer→점수→Detector→Aggregator→평가` 흐름)이
**두 곳**에 나온다 — 메인 타이틀 아래 부제(`st.markdown("Darts ForecastingAnomalyModel...")`)와,
빈 상태의 "📖 파이프라인 / CSV 형식 자세히 보기" expander 안. 부제 줄은 한 줄 요약이라
유지하고, expander 안의 "파이프라인 (강의 14 기반 + 확장)" 5단계 설명은 **빈 상태에서만
보이므로 그대로 둔다** — 둘은 보이는 시점이 다르므로 사실 중복이 아니다.

다만 화면 캡처를 보면 데이터 업로드 후에도 같은 설명이 다시 나오는 지점이 있다면(섹션
어딘가에 파이프라인 5단계가 재차 설명되는 부분), 그 두 번째 인스턴스를 제거하고 짧은
한 줄 캡션으로 축소한다. 정확한 중복 위치는 실제 코드를 보고 판단하되, **부제 한 줄 +
빈 상태 expander 하나, 총 두 곳까지만 허용**하고 그 이상 반복되는 설명은 제거한다.

---

## 7. 이모지 정리 — 필수만 남기고 나머지 제거

다음 기준으로 정리한다.

**유지(섹션 식별용 핵심 이모지, 그대로 둔다)**:
- 4개 섹션 제목의 이모지: 📋(데이터 개요), 🔍(이상탐지 결과), 📊(평가 대시보드), ⬇️(결과 다운로드)
- 사이드바 카테고리 제목의 이모지 중 시각적 구분이 필요한 것만 1개씩(예: 데이터 파일 📁,
  컬럼 매핑 🧭, Aggregator 🧩) — 같은 레벨에서 2개 이상 겹치지 않게 한다

**제거 대상**:
- 위젯 라벨 안의 장식용 이모지(예: "🕒 시간 컬럼", "📊 값 컬럼", "🎯 정답 라벨 컬럼"에서
  이모지를 떼고 텍스트만 남긴다 — "시간 컬럼", "값 컬럼 (다변량)", "정답 라벨 컬럼 (선택)")
- `st.caption`/`st.markdown` 본문 문장 중간에 들어간 이모지(✨, ⭐ 등 강조용 제외)
- 같은 화면에서 3개 이상 중복되는 이모지(예: 🧩가 사이드바와 본문에 둘 다 있다면 본문만 유지)

`⭐`(Scorer 카드의 "최고 성능" 강조 배지)는 정보 전달 기능이 있으므로 유지한다.

---

## 8. 개발자 배지 텍스트 변경

"Dashboard Developed by" 아래 표시되는 텍스트를 찾아 변경한다.

```python
<div style="font-size: 14px; font-weight: 700; color: {CLR_ACCENT};">시계열 이상탐지 프로젝트</div>
```

위 줄에서 `시계열 이상탐지 프로젝트`를 `C321028 박솔`로 교체한다.

---

## 9. "샘플 데이터가 없으면..." 캡션 전체 삭제

사이드바에서 다음 줄을 찾아 완전히 삭제한다(코드 자체를 지운다, 빈 문자열로 바꾸는 게 아니다):

```python
st.sidebar.caption("샘플 데이터가 없으면 `python make_multivariate_sample.py`로 생성하세요.")
```

---

## 완료 기준

- `python3 -m py_compile anomaly_pipeline.py app.py` 통과
- 페이지 아이콘·사이드바·메인 타이틀에 🛰️ 대신 🫐가 보임
- `nyc_taxi.csv` 업로드 시 "빈도" 카드에 `<30 * ...` 대신 `30분`이 보임
- 메인 3단 차트 아래 "이 차트를 어떻게 해석하나요?" expander가 보임
- 수동 보정 "선택 적용" 클릭 후 적용된 시점 수가 caption으로 보임
- ROC/PR/혼동행렬 탭에 한 줄 설명이 보임
- 화면 어디에도 "시계열 이상탐지 프로젝트" 텍스트가 없고 "C321028 박솔"로 대체됨
- "샘플 데이터가 없으면..." 캡션이 사이드바에서 완전히 사라짐
- 위젯 라벨에서 장식용 이모지가 빠지고 텍스트만 남음(섹션 제목 이모지는 유지)

## 변경하지 말 것

- `aggregators.py`는 건드리지 않는다.
- 색상 변수(`CLR_ACCENT` 등)와 평가지표 계산 로직은 변경하지 않는다.
- 4개 섹션 제목의 이모지(📋🔍📊⬇️)는 제거하지 않는다 — 섹션 구분의 핵심 식별자다.
