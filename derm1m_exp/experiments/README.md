# 5가지 피부과 진단 방법 비교 실험

GPT-4o 모델을 사용하여 5가지 진단 방법을 체계적으로 비교합니다.

## 비교 방법

| # | 방법 | 설명 |
|---|------|------|
| 1 | Baseline + Labels | 질병 라벨 목록 제공 |
| 2 | Baseline + No Labels | 라벨 없이 자유 진단 |
| 3 | Baseline + Hierarchical | 온톨로지 트리 구조 제공 |
| 4 | DermatologyAgent | 온톨로지 기반 계층적 탐색 |
| 5 | ReActAgent | ReAct 패턴 추론 에이전트 |

## 설치

```bash
# 상위 requirements 설치
pip install -r ../requirements.txt

# 또는 최소 의존성만
pip install openai pandas numpy tqdm Pillow
```

## API 키 설정

### 방법 1: .env 파일 사용 (권장)

`baseline/.env` 파일에 API 키가 설정되어 있으면 자동으로 로드됩니다.

```bash
# baseline/.env
OPENAI_API_KEY=sk-your-api-key
```

### 방법 2: 환경변수 설정

```bash
export OPENAI_API_KEY=sk-your-api-key
```

### 방법 3: 명령줄 인자

```bash
python run_comparison_experiment.py --api_key sk-your-api-key ...
```

## 사용법

### 테스트 모드 (5개 샘플) - 가장 간단한 실행

```bash
# .env 파일이 있으면 API 키와 데이터셋 경로 자동 설정
python run_comparison_experiment.py --test_mode --num_samples 5
```

### 전체 실행 (100개 샘플)

```bash
python run_comparison_experiment.py
```

### 특정 방법만 실행

```bash
python run_comparison_experiment.py \
    --methods baseline_labels,react_agent \
    --test_mode
```

### 모든 옵션 지정

```bash
python run_comparison_experiment.py \
    --input_csv /path/to/Derm1M_v2_pretrain_ontology_sampled_100.csv \
    --output_dir ./outputs \
    --api_key $OPENAI_API_KEY \
    --test_mode \
    --num_samples 5 \
    --verbose
```

## 결과 파일

```
outputs/{timestamp}/
├── experiment_config.json          # 실험 설정
├── logs/
│   └── experiment.log              # 실험 로그
├── predictions/
│   ├── 1_baseline_labels.csv       # 방법 1 예측
│   ├── 2_baseline_no_labels.csv    # 방법 2 예측
│   ├── 3_baseline_hierarchical.csv # 방법 3 예측
│   ├── 4_dermatology_agent.csv     # 방법 4 예측
│   └── 5_react_agent.csv           # 방법 5 예측
└── evaluation/
    ├── metrics_summary.csv         # 메트릭 요약
    ├── per_sample_comparison.csv   # 샘플별 비교
    └── detailed_analysis.csv       # 상세 분석
```

## 평가 메트릭

- **Exact Match**: 정확히 일치
- **Partial Match**: 부분 일치 비율
- **Hierarchical F1**: 계층적 유사도 기반 F1
- **Avg Distance**: 평균 트리 거리 (낮을수록 좋음)
- **Partial Credit**: 공통 조상까지의 점수
- **Level Accuracy**: 레벨별 정확도 (L1~L5)

## 파일 구조

```
experiments/
├── run_comparison_experiment.py  # 메인 실험 스크립트
├── experiment_utils.py           # 유틸리티 함수
├── hierarchical_baseline.py      # 온톨로지 프롬프트 래퍼
├── vlm_wrapper.py                # GPT-4o 래퍼
├── requirements.txt              # 의존성
└── README.md                     # 이 파일
```
