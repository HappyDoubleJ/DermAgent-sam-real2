# DermAgent 코드베이스 완전 분석 문서

## 목차
1. [프로젝트 개요](#1-프로젝트-개요)
2. [디렉터리 구조](#2-디렉터리-구조)
3. [핵심 컴포넌트 상세 분석](#3-핵심-컴포넌트-상세-분석)
4. [데이터 구조](#4-데이터-구조)
5. [온톨로지 시스템](#5-온톨로지-시스템)
6. [에이전트 시스템](#6-에이전트-시스템)
7. [SAM 통합 시스템](#7-sam-통합-시스템)
8. [평가 시스템](#8-평가-시스템)
9. [실험 프레임워크](#9-실험-프레임워크)
10. [사용 방법](#10-사용-방법)
11. [증상 분석 스키마 추가 방안](#11-증상-분석-스키마-추가-방안)
12. [최종 테스트 로직 설명](#12-최종-테스트-로직-설명)

---

## 1. 프로젝트 개요

### 1.1 목적
DermAgent는 **피부 질환 자동 진단 시스템**으로, Vision-Language Models(VLM)과 온톨로지 기반 계층적 탐색을 결합하여 피부 이미지를 분석하고 질환을 진단합니다.

### 1.2 핵심 특징
- **온톨로지 기반 계층적 진단**: 950+ 피부 질환을 트리 구조로 분류
- **다양한 진단 방법 지원**: Baseline, ReAct Agent, DermatologyAgent, SAM 통합 에이전트
- **SAM(Segment Anything Model) 통합**: 병변 영역 자동 세그멘테이션
- **계층적 평가 메트릭**: Exact Match, Hierarchical F1, Partial Credit 등

### 1.3 지원 VLM 모델
- GPT-4o (OpenAI)
- Qwen2-VL (Alibaba)
- InternVL2 (OpenGVLab)

---

## 2. 디렉터리 구조

```
DermAgent-sam-real2/
├── dataset/
│   ├── Derm1M/
│   │   ├── ontology.json                    # 피부 질환 온톨로지 트리
│   │   ├── Derm1M_v2_pretrain_ontology_sampled_100.csv
│   │   ├── Derm1M_v2_pretrain_ontology_sampled_500.csv
│   │   └── Derm1M_v2_pretrain_ontology_sampled_1000.csv
│   ├── random100.csv                        # 테스트용 100개 샘플
│   └── derm1m.py
├── derm1m_exp/
│   ├── DermAgent/
│   │   └── agent/
│   │       ├── dermatology_agent.py         # 핵심 진단 에이전트
│   │       ├── dermatology_agent_sam.py     # SAM 통합 에이전트
│   │       ├── react_agent.py               # ReAct 패턴 에이전트
│   │       ├── pipeline.py                  # 통합 파이프라인
│   │       ├── run_agent.py                 # 에이전트 실행 스크립트
│   │       └── compare_agents.py            # 에이전트 비교
│   ├── eval/
│   │   ├── evaluation_metrics.py            # 평가 메트릭
│   │   └── ontology_utils.py                # 온톨로지 유틸리티
│   ├── baseline/
│   │   ├── baseline.py                      # 기본 베이스라인
│   │   ├── baseline_sam.py                  # SAM 베이스라인
│   │   └── model.py                         # VLM 모델 래퍼
│   ├── experiments/
│   │   ├── run_comparison_experiment.py     # 실험 메인 스크립트
│   │   ├── experiment_utils.py              # 실험 유틸리티
│   │   ├── hierarchical_baseline.py         # 계층적 베이스라인
│   │   ├── sam_wrapper.py                   # SAM 래퍼
│   │   ├── vlm_wrapper.py                   # VLM 래퍼
│   │   └── outputs/                         # 실험 결과
│   └── SA-project-SAM/                      # SAM 모듈 (submodule)
└── project_path.py                          # 프로젝트 경로 설정
```

---

## 3. 핵심 컴포넌트 상세 분석

### 3.1 진단 방법 8가지 비교

| # | 방법명 | 설명 | 특징 |
|---|--------|------|------|
| 1 | Baseline + Labels | 질병 라벨 목록을 프롬프트에 제공 | 가장 단순한 방식 |
| 2 | Baseline + No Labels | 라벨 없이 자유 진단 | Open-ended 진단 |
| 3 | Baseline + Hierarchical | 온톨로지 트리 구조 전체 제공 | 구조 정보 활용 |
| 4 | DermatologyAgent | 온톨로지 기반 계층적 탐색 에이전트 | 단계별 탐색, Backtracking |
| 5 | ReActAgent | ReAct 패턴 기반 추론 에이전트 | Reasoning + Acting 반복 |
| 6 | SAM Baseline | SAM 세그멘테이션 + VLM 진단 | 병변 영역 분리 |
| 7 | SAM + LLM-Guided | LLM 가이드 세그멘테이션 + VLM | 지능적 병변 탐지 |
| 8 | DermatologyAgentSAM | SAM + 온톨로지 기반 에이전트 | 통합 접근 |

---

## 4. 데이터 구조

### 4.1 random100.csv 컬럼 설명

```
filename              : 이미지 파일 경로 (예: youtube/xxx.jpg, pubmed/xxx.png)
caption               : 상세 이미지 설명 (LLM이 분석하기 좋은 정보 포함)
truncated_caption     : 축약된 캡션
source                : 데이터 출처 (youtube, pubmed, textbook, public 등)
source_type           : 소스 유형 (edu, knowledge, forum)
disease_label         : 정답 질환 라벨 (Ground Truth)
hierarchical_disease_label : 계층적 질환 라벨 (쉼표로 구분된 경로)
skin_concept          : 피부 개념 (red, patch, macule 등)
body_location         : 신체 위치 (face, trunk, arm 등)
symptoms              : 증상 정보 (대부분 "No symptom information")
age                   : 나이 정보
gender                : 성별 정보
```

### 4.2 주요 발견사항
- **symptoms 컬럼**: 대부분 "No symptom information"으로 비어있음
- **caption 컬럼**: 증상, 외관, 진단에 대한 상세 설명 포함
- **skin_concept 컬럼**: 시각적 특징 (red, patch, papule 등)
- **hierarchical_disease_label**: 온톨로지 경로 (예: "inflammatory, infectious, fungal, Tinea corporis")

### 4.3 데이터 예시

```csv
filename: youtube/nSw7XdNR0bg_frame_35881_3.jpg
caption: "This image depicts Veckus syndrome, a predominantly male disease characterized
         by clinical diagnoses such as relapsing polychondritis, neutrophilic dermatoses,
         and forms of vasculitis. It causes subtle chondritis in the ear and nose..."
disease_label: vasculitis
hierarchical_disease_label: vasculitis
skin_concept: No visual concepts
body_location: ear, nose
symptoms: No symptom information
```

---

## 5. 온톨로지 시스템

### 5.1 OntologyTree 클래스 (`ontology_utils.py`)

피부 질환을 계층적 트리 구조로 관리하는 핵심 클래스입니다.

```python
class OntologyTree:
    """온톨로지 트리 구조 관리"""

    def __init__(self, ontology_path: Optional[str] = None):
        # ontology.json 자동 탐색 또는 명시적 경로 사용

    # 핵심 메서드들
    def is_valid_node(self, node: str) -> bool:
        """노드가 온톨로지에 존재하는지 확인"""

    def get_path_to_root(self, node: str) -> List[str]:
        """노드에서 루트까지의 경로 반환
        예: "Tinea corporis" -> ["Tinea corporis", "fungal", "infectious", "inflammatory", "root"]
        """

    def get_hierarchical_distance(self, node1: str, node2: str) -> int:
        """두 노드 간의 계층적 거리 계산"""

    def get_lca(self, node1: str, node2: str) -> Optional[str]:
        """Lowest Common Ancestor (최소 공통 조상) 찾기"""

    def get_children(self, node: str) -> List[str]:
        """노드의 직계 자식 반환"""

    def get_all_descendants(self, node: str) -> Set[str]:
        """노드의 모든 자손 반환"""

    def get_canonical_name(self, node: str) -> Optional[str]:
        """정규화된 노드 이름 반환 (대소문자/공백 무시)"""
```

### 5.2 온톨로지 구조 예시

```
root
├── inflammatory
│   ├── infectious
│   │   ├── bacterial
│   │   │   ├── pyoderma
│   │   │   │   └── abscess
│   │   │   └── ...
│   │   ├── fungal
│   │   │   ├── Tinea corporis
│   │   │   ├── Tinea pedis
│   │   │   └── Candidiasis
│   │   └── viral
│   │       └── ...
│   ├── non-infectious
│   │   ├── eczema
│   │   │   ├── contact dermatitis
│   │   │   │   └── allergic contact dermatitis
│   │   │   └── ...
│   │   └── psoriasis
│   └── ...
├── proliferations
│   ├── benign
│   └── malignant
│       └── melanoma
└── ...
```

---

## 6. 에이전트 시스템

### 6.1 DermatologyAgent (`dermatology_agent.py`)

온톨로지 기반 계층적 탐색을 수행하는 핵심 에이전트입니다.

#### 6.1.1 진단 단계 (DiagnosisStep)

```python
class DiagnosisStep(Enum):
    INITIAL_ASSESSMENT = "initial_assessment"      # Step 1: 특징 추출
    CATEGORY_CLASSIFICATION = "category_classification"  # Step 2: 대분류 선택
    SUBCATEGORY_CLASSIFICATION = "subcategory_classification"  # Step 3: 중/소분류
    DIFFERENTIAL_DIAGNOSIS = "differential_diagnosis"  # Step 4: 감별 진단
    FINAL_DIAGNOSIS = "final_diagnosis"            # Step 5: 최종 진단
```

#### 6.1.2 관찰 결과 데이터 클래스

```python
@dataclass
class ObservationResult:
    morphology: List[str]     # 형태: papule, plaque, vesicle 등
    color: List[str]          # 색상: red, brown, white 등
    distribution: List[str]   # 분포: localized, generalized 등
    location: str             # 위치: face, trunk 등
    surface: List[str]        # 표면: scaly, smooth 등
    symptoms: List[str]       # 증상: itching, burning 등 ★ 현재 미사용
    raw_description: str      # 원본 VLM 응답
```

#### 6.1.3 진단 플로우

```
1. step_initial_assessment()
   └─ VLM으로 이미지 분석 → morphology, color, distribution, surface, location 추출

2. step_category_classification()
   └─ 루트 카테고리 선택 (inflammatory, proliferations, ...)

3. step_subcategory_classification() [반복]
   └─ 하위 카테고리 탐색 (최대 3단계 깊이)
   └─ 필요시 Backtracking 수행

4. step_differential_diagnosis()
   └─ DifferentialDiagnosisTool로 후보 질환 비교

5. step_final_diagnosis()
   └─ 최종 진단 결정 및 신뢰도 계산
```

#### 6.1.4 Backtracking 메커니즘

신뢰도가 낮으면 이전 단계로 돌아가 다른 경로를 탐색합니다.

```python
def _should_backtrack(self, state: DiagnosisState) -> bool:
    """
    Backtracking 필요 여부 판단
    - 모든 후보의 신뢰도 < MIN_DIFFERENTIAL_CONFIDENCE (0.3)
    - backtrack_count < max_backtracks (3)
    """

def _backtrack_to_subcategory(self, state: DiagnosisState) -> bool:
    """하위 카테고리 레벨로 돌아가서 대체 경로 시도"""

def _backtrack_to_category(self, state: DiagnosisState) -> bool:
    """카테고리 레벨로 돌아가서 다른 루트 카테고리 시도"""
```

### 6.2 ReActDermatologyAgent (`react_agent.py`)

ReAct (Reasoning + Acting) 패턴을 적용한 에이전트입니다.

#### 6.2.1 행동 유형

```python
class ActionType(Enum):
    OBSERVE = "observe"              # 이미지 관찰
    ANALYZE_FEATURES = "analyze"     # 특징 분석
    NAVIGATE_ONTOLOGY = "navigate"   # 온톨로지 탐색
    COMPARE_DISEASES = "compare"     # 질환 비교
    VERIFY = "verify"                # 진단 검증
    CONCLUDE = "conclude"            # 결론 도출
```

#### 6.2.2 사고 단계 데이터 클래스

```python
@dataclass
class ThoughtStep:
    step_num: int
    thought: str       # 현재 생각 (Reasoning)
    action: ActionType # 수행할 행동 (Acting)
    action_input: Dict # 행동 입력 파라미터
    observation: str   # 행동 결과 (Observation)
```

#### 6.2.3 도구 클래스

- **OntologyNavigatorTool**: 온톨로지 탐색 (get_children, get_path, search, validate)
- **DiseaseComparatorTool**: VLM 기반 질환 비교

---

## 7. SAM 통합 시스템

### 7.1 DermatologyAgentSAM (`dermatology_agent_sam.py`)

SAM(Segment Anything Model)을 통합한 진단 에이전트입니다.

#### 7.1.1 세그멘테이션 결과 데이터 클래스

```python
@dataclass
class SegmentationResult:
    mask: Optional[np.ndarray]        # 세그멘테이션 마스크
    score: float                      # 세그멘테이션 품질 점수
    method: str                       # 사용된 방법 (center, lesion_features, llm_guided)
    overlay_image: Optional[np.ndarray]  # 오버레이 이미지
    cropped_image: Optional[np.ndarray]  # 크롭된 이미지
    lesion_count: int                 # 감지된 병변 수
    bbox: Optional[Tuple[int, int, int, int]]  # 바운딩 박스
```

#### 7.1.2 진단 파이프라인 (6단계)

```
Step 0: step_segmentation()
   └─ SAM으로 병변 영역 세그멘테이션
   └─ 전략: center, lesion_features, both, llm_guided

Step 1: step_initial_assessment()
   └─ 세그멘테이션 오버레이를 사용한 VLM 분석

Step 2-5: 기존 DermatologyAgent와 동일
   └─ 세그멘테이션 결과를 활용한 계층적 탐색
```

#### 7.1.3 세그멘테이션 전략

| 전략 | 설명 |
|------|------|
| `center` | 이미지 중앙 기준 세그멘테이션 |
| `lesion_features` | 병변 특징 기반 세그멘테이션 |
| `both` | 두 방법 중 점수가 높은 것 선택 |
| `llm_guided` | LLM이 병변 위치를 가이드 |

### 7.2 SAMBaselineWrapper (`sam_wrapper.py`)

실험 프레임워크에서 SAM을 쉽게 사용할 수 있도록 래핑합니다.

```python
class SAMBaselineWrapper:
    def __init__(
        self,
        vlm_model,
        segmenter_type: str = "sam",         # sam, sam2, medsam2
        segmentation_strategy: str = "center",
        checkpoint_dir: Optional[str] = None,
        verbose: bool = False
    )

    def analyze_image(self, image_path: str) -> Dict:
        """단일 이미지 분석 - 세그멘테이션 + VLM 진단"""
```

---

## 8. 평가 시스템

### 8.1 HierarchicalEvaluator (`evaluation_metrics.py`)

온톨로지 트리 구조를 활용한 평가 메트릭을 제공합니다.

#### 8.1.1 평가 메트릭 종류

| 메트릭 | 설명 | 계산 방식 |
|--------|------|-----------|
| **Exact Match** | 정확히 일치 | GT ∩ Pred ≠ ∅ |
| **Partial Match** | 부분 일치 비율 | \|GT ∩ Pred\| / \|GT\| |
| **Hierarchical F1** | 계층적 F1 점수 | Jaccard 유사도 기반 |
| **Avg Distance** | 평균 계층적 거리 | LCA까지의 거리 합 |
| **Partial Credit** | 부분 점수 | 공통 경로 깊이 비례 |
| **Level Accuracy** | 레벨별 정확도 | 각 깊이에서의 정확도 |
| **Top-K Accuracy** | Top-K 정확도 | 상위 K개 예측 중 정답 포함 여부 |

#### 8.1.2 계층적 유사도 계산

```python
def hierarchical_similarity(self, label1: str, label2: str) -> float:
    """
    Jaccard 유사도 기반 (root 제외):
    Similarity = |Ancestors(A) ∩ Ancestors(B)| / |Ancestors(A) ∪ Ancestors(B)|

    예: "Tinea corporis" vs "Tinea pedis"
        - Ancestors(Tinea corporis) = {Tinea corporis, fungal, infectious, inflammatory}
        - Ancestors(Tinea pedis) = {Tinea pedis, fungal, infectious, inflammatory}
        - 교집합 = {fungal, infectious, inflammatory}
        - 합집합 = {Tinea corporis, Tinea pedis, fungal, infectious, inflammatory}
        - 유사도 = 3/5 = 0.6
    """
```

#### 8.1.3 사용 예시

```python
evaluator = HierarchicalEvaluator()

# 단일 샘플 평가
result = evaluator.evaluate_single(
    gt_labels=["Tinea corporis"],
    pred_labels=["Tinea pedis"]
)
# {'exact_match': 0.0, 'hierarchical_f1': 0.6, 'avg_min_distance': 2.0, ...}

# 배치 평가 (Top-K 포함)
batch_result = evaluator.evaluate_batch_with_top_k(
    ground_truths=[["Tinea corporis"], ["Psoriasis"]],
    predictions=[["Tinea pedis", "Candidiasis"], ["Psoriasis"]],
    k_values=[1, 3, 5]
)
```

---

## 9. 실험 프레임워크

### 9.1 run_comparison_experiment.py

8가지 진단 방법을 비교하는 메인 실험 스크립트입니다.

#### 9.1.1 실행 방법

```bash
# .env 파일 사용 (API 키 자동 로드)
python run_comparison_experiment.py \
    --input_csv /path/to/random100.csv \
    --output_dir ./outputs \
    --test_mode --num_samples 5

# 특정 방법만 실행
python run_comparison_experiment.py \
    --methods baseline_labels,react_agent \
    --input_csv /path/to/data.csv
```

#### 9.1.2 출력 디렉터리 구조

```
outputs/20251204_101531/
├── experiment_config.json       # 실험 설정
├── logs/
│   └── experiment.log          # 실험 로그
├── predictions/
│   ├── 1_baseline_labels.csv   # 각 방법별 예측 결과
│   ├── 2_baseline_no_labels.csv
│   ├── 3_baseline_hierarchical.csv
│   ├── 4_dermatology_agent.csv
│   └── 5_react_agent.csv
└── evaluation/
    ├── metrics_summary.csv     # 메트릭 요약
    ├── detailed_analysis.csv   # 상세 분석
    ├── per_sample_comparison.csv
    └── method_differences.json
```

### 9.2 experiment_utils.py

실험에 필요한 유틸리티 함수들을 제공합니다.

#### 9.2.1 주요 데이터 클래스

```python
@dataclass
class AgentTrace:
    """에이전트 전체 추론 트레이스"""
    sample_id: int
    filename: str
    agent_type: str
    steps: List[AgentStep]           # 추론 단계들
    observations: Dict[str, Any]     # 관찰 결과
    ontology_path: List[str]         # 탐색 경로
    candidates_considered: List[str] # 고려한 후보들
    primary_diagnosis: str           # 최종 진단
    confidence: float                # 신뢰도

@dataclass
class MethodResult:
    """단일 방법의 단일 샘플 결과"""
    sample_id: int
    filename: str
    ground_truth: str
    hierarchical_gt: str
    prediction: str
    confidence: float
    reasoning: str
    all_predictions: List[str]       # Top-K용 전체 예측
    agent_trace: Optional[AgentTrace]

@dataclass
class MethodEvaluation:
    """단일 방법의 전체 평가 결과"""
    method_name: str
    exact_match: float
    partial_match: float
    hierarchical_f1: float
    avg_distance: float
    partial_credit: float
    level_accuracy: Dict[int, float]
    top_k_accuracy: Dict[int, float]
```

---

## 10. 사용 방법

### 10.1 환경 설정

```bash
# 의존성 설치
pip install -r derm1m_exp/requirements.txt

# OpenAI API 키 설정
export OPENAI_API_KEY="your-api-key"

# 또는 .env 파일 생성
echo "OPENAI_API_KEY=your-api-key" > derm1m_exp/experiments/.env
```

### 10.2 단일 이미지 진단

```python
from derm1m_exp.DermAgent.agent.dermatology_agent import DermatologyAgent
from derm1m_exp.experiments.vlm_wrapper import GPT4oWrapper

# VLM 초기화
vlm = GPT4oWrapper(api_key="your-api-key", use_labels_prompt=False)

# 에이전트 초기화
agent = DermatologyAgent(vlm_model=vlm, verbose=True)

# 진단 수행
result = agent.diagnose("/path/to/skin_image.jpg")
print(f"진단: {result['final_diagnosis']}")
print(f"신뢰도: {result['confidence']}")
```

### 10.3 배치 실험 실행

```bash
cd derm1m_exp/experiments

python run_comparison_experiment.py \
    --input_csv ../dataset/random100.csv \
    --output_dir ./outputs \
    --methods baseline_labels,baseline_no_labels,dermatology_agent,react_agent \
    --num_samples 10 \
    --verbose
```

### 10.4 결과 평가

```python
from derm1m_exp.eval.evaluation_metrics import HierarchicalEvaluator

evaluator = HierarchicalEvaluator()

# CSV에서 결과 로드 후 평가
result = evaluator.evaluate_batch_with_top_k(
    ground_truths=gts,
    predictions=preds,
    k_values=[1, 3, 5]
)

evaluator.print_evaluation_report(result)
```

---

## 11. 증상 분석 스키마 추가 방안

### 11.1 현재 상황 분석

#### 11.1.1 random100.csv의 symptoms 컬럼 현황
- 대부분의 레코드가 **"No symptom information"**으로 되어 있음
- 실제 증상 정보가 있는 레코드: 극소수
- 예시: `tender` (abscess의 경우)

#### 11.1.2 대안적 증상 정보 소스
| 컬럼명 | 내용 | 증상 정보 유무 |
|--------|------|----------------|
| `symptoms` | 명시적 증상 | 대부분 비어있음 |
| `caption` | 상세 설명 | ★ 증상 정보 풍부 |
| `truncated_caption` | 축약된 설명 | 증상 정보 포함 |
| `skin_concept` | 시각적 특징 | 간접적 증상 힌트 |

#### 11.1.3 caption에서 추출 가능한 증상 정보 예시

```
caption: "...a skin disease characterized by red, itchy patches on the skin..."
         → 추출 가능: itchy (가려움)

caption: "...an abscess, characterized by yellow-green color and severe tenderness..."
         → 추출 가능: tenderness (압통)

caption: "...characterized by bothersome appearance, itching, burning..."
         → 추출 가능: itching (가려움), burning (화끈거림)
```

### 11.2 증상 분석 구조화 스키마 제안

#### 11.2.1 피부 질환 도메인 특화 증상 스키마

```python
from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum

class SymptomSeverity(Enum):
    NONE = "none"
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"

class SymptomDuration(Enum):
    ACUTE = "acute"           # < 2주
    SUBACUTE = "subacute"     # 2주 ~ 6주
    CHRONIC = "chronic"       # > 6주
    UNKNOWN = "unknown"

@dataclass
class DermatologySymptomSchema:
    """피부 질환 도메인 특화 증상 스키마"""

    # === 주관적 증상 (Subjective Symptoms) ===
    pruritus: bool = False              # 가려움 (Itching)
    pruritus_severity: SymptomSeverity = SymptomSeverity.NONE

    pain: bool = False                  # 통증
    pain_type: str = ""                 # burning, stinging, throbbing, aching
    pain_severity: SymptomSeverity = SymptomSeverity.NONE

    tenderness: bool = False            # 압통 (만지면 아픔)
    burning: bool = False               # 화끈거림
    tingling: bool = False              # 저림/따끔거림

    # === 객관적 증상 (Objective Signs) ===
    erythema: bool = False              # 홍반 (발적)
    swelling: bool = False              # 부종
    warmth: bool = False                # 열감
    discharge: bool = False             # 분비물
    discharge_type: str = ""            # serous, purulent, hemorrhagic

    scaling: bool = False               # 각질/비늘
    crusting: bool = False              # 딱지
    bleeding: bool = False              # 출혈

    # === 전신 증상 (Systemic Symptoms) ===
    fever: bool = False                 # 발열
    malaise: bool = False               # 권태감
    lymphadenopathy: bool = False       # 림프절 비대

    # === 시간적 특성 ===
    duration: SymptomDuration = SymptomDuration.UNKNOWN
    onset: str = ""                     # sudden, gradual
    progression: str = ""               # stable, worsening, improving
    recurrent: bool = False             # 재발성

    # === 악화/완화 요인 ===
    aggravating_factors: List[str] = field(default_factory=list)
    # 예: sun_exposure, heat, sweating, stress, specific_foods

    relieving_factors: List[str] = field(default_factory=list)
    # 예: cold, moisturizer, rest

    # === 원본 텍스트 ===
    raw_symptom_text: str = ""
    extraction_source: str = ""         # caption, symptoms, user_input
    extraction_confidence: float = 0.0

    def to_dict(self) -> dict:
        return {
            "subjective": {
                "pruritus": {"present": self.pruritus, "severity": self.pruritus_severity.value},
                "pain": {"present": self.pain, "type": self.pain_type, "severity": self.pain_severity.value},
                "tenderness": self.tenderness,
                "burning": self.burning,
                "tingling": self.tingling,
            },
            "objective": {
                "erythema": self.erythema,
                "swelling": self.swelling,
                "warmth": self.warmth,
                "discharge": {"present": self.discharge, "type": self.discharge_type},
                "scaling": self.scaling,
                "crusting": self.crusting,
                "bleeding": self.bleeding,
            },
            "systemic": {
                "fever": self.fever,
                "malaise": self.malaise,
                "lymphadenopathy": self.lymphadenopathy,
            },
            "temporal": {
                "duration": self.duration.value,
                "onset": self.onset,
                "progression": self.progression,
                "recurrent": self.recurrent,
            },
            "modifying_factors": {
                "aggravating": self.aggravating_factors,
                "relieving": self.relieving_factors,
            },
            "metadata": {
                "raw_text": self.raw_symptom_text,
                "source": self.extraction_source,
                "confidence": self.extraction_confidence,
            }
        }
```

### 11.3 증상 추출 모듈 구현

```python
class SymptomExtractor:
    """caption/truncated_caption에서 증상 정보를 추출하는 모듈"""

    SYMPTOM_KEYWORDS = {
        # 주관적 증상
        "pruritus": ["itchy", "itching", "pruritus", "itch"],
        "pain": ["pain", "painful", "hurts", "aching", "sore"],
        "burning": ["burning", "burn", "hot sensation"],
        "tenderness": ["tender", "tenderness", "sensitive to touch"],
        "tingling": ["tingling", "prickling", "pins and needles"],

        # 객관적 증상
        "erythema": ["red", "redness", "erythema", "erythematous", "inflamed"],
        "swelling": ["swollen", "swelling", "edema", "puffy"],
        "scaling": ["scaly", "scaling", "scale", "flaky", "flaking"],
        "crusting": ["crusted", "crusting", "crust", "scab"],
        "discharge": ["discharge", "oozing", "weeping", "pus", "purulent"],

        # 전신 증상
        "fever": ["fever", "febrile", "high temperature"],

        # 심각도
        "severe": ["severe", "intense", "extreme", "unbearable"],
        "mild": ["mild", "slight", "minor"],
    }

    def __init__(self, vlm_model=None):
        self.vlm = vlm_model

    def extract_from_text(self, text: str) -> DermatologySymptomSchema:
        """텍스트에서 증상 정보 추출 (키워드 기반)"""
        schema = DermatologySymptomSchema()
        text_lower = text.lower()

        # 키워드 매칭
        for symptom, keywords in self.SYMPTOM_KEYWORDS.items():
            for keyword in keywords:
                if keyword in text_lower:
                    setattr(schema, symptom, True)
                    break

        schema.raw_symptom_text = text
        schema.extraction_source = "keyword_matching"
        schema.extraction_confidence = 0.6

        return schema

    def extract_with_vlm(self, text: str, image_path: str = None) -> DermatologySymptomSchema:
        """VLM을 사용한 고급 증상 추출"""
        if self.vlm is None:
            return self.extract_from_text(text)

        prompt = f"""Analyze the following dermatological description and extract symptoms.

Description: {text}

Extract symptoms and respond in JSON format:
{{
    "subjective_symptoms": {{
        "pruritus": {{"present": true/false, "severity": "none/mild/moderate/severe"}},
        "pain": {{"present": true/false, "type": "burning/stinging/throbbing/aching", "severity": "..."}},
        "tenderness": true/false,
        "burning": true/false,
        "tingling": true/false
    }},
    "objective_signs": {{
        "erythema": true/false,
        "swelling": true/false,
        "warmth": true/false,
        "scaling": true/false,
        "crusting": true/false,
        "discharge": {{"present": true/false, "type": "serous/purulent/hemorrhagic"}}
    }},
    "duration": "acute/subacute/chronic/unknown",
    "onset": "sudden/gradual",
    "aggravating_factors": ["factor1", "factor2"],
    "relieving_factors": ["factor1", "factor2"]
}}

Provide ONLY the JSON output."""

        if image_path:
            response = self.vlm.chat_img(prompt, [image_path])
        else:
            # 텍스트만 있는 경우
            response = self.vlm.chat_img(prompt, [])

        # JSON 파싱 후 스키마로 변환
        schema = self._parse_vlm_response(response)
        schema.raw_symptom_text = text
        schema.extraction_source = "vlm"
        schema.extraction_confidence = 0.85

        return schema

    def _parse_vlm_response(self, response: str) -> DermatologySymptomSchema:
        """VLM 응답을 스키마로 변환"""
        import json
        import re

        schema = DermatologySymptomSchema()

        try:
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                data = json.loads(json_match.group())

                # Subjective
                subj = data.get("subjective_symptoms", {})
                if subj.get("pruritus", {}).get("present"):
                    schema.pruritus = True
                    severity_str = subj.get("pruritus", {}).get("severity", "none")
                    schema.pruritus_severity = SymptomSeverity[severity_str.upper()]

                if subj.get("pain", {}).get("present"):
                    schema.pain = True
                    schema.pain_type = subj.get("pain", {}).get("type", "")

                schema.tenderness = subj.get("tenderness", False)
                schema.burning = subj.get("burning", False)
                schema.tingling = subj.get("tingling", False)

                # Objective
                obj = data.get("objective_signs", {})
                schema.erythema = obj.get("erythema", False)
                schema.swelling = obj.get("swelling", False)
                schema.warmth = obj.get("warmth", False)
                schema.scaling = obj.get("scaling", False)
                schema.crusting = obj.get("crusting", False)
                if obj.get("discharge", {}).get("present"):
                    schema.discharge = True
                    schema.discharge_type = obj.get("discharge", {}).get("type", "")

                # Temporal
                duration_str = data.get("duration", "unknown")
                schema.duration = SymptomDuration[duration_str.upper()]
                schema.onset = data.get("onset", "")

                # Modifying factors
                schema.aggravating_factors = data.get("aggravating_factors", [])
                schema.relieving_factors = data.get("relieving_factors", [])

        except Exception:
            pass

        return schema
```

### 11.4 caption/truncated_caption 사용 테스트 방안

random100.csv의 `symptoms` 컬럼 대신 `caption` 또는 `truncated_caption`을 사용하는 것은 **매우 합리적인 선택**입니다.

#### 11.4.1 장점
1. **풍부한 정보**: caption에는 시각적 특징, 증상, 진단 단서가 모두 포함
2. **실제 임상 시나리오 시뮬레이션**: 의사가 환자로부터 받는 정보와 유사
3. **다양한 증상 표현**: 자연어로 다양하게 표현된 증상 정보

#### 11.4.2 단점
1. **노이즈 가능성**: 진단명이나 불필요한 정보도 포함될 수 있음
2. **일관성 부족**: 각 caption의 형식과 상세도가 다름
3. **Ground Truth 오염**: caption에 진단명이 포함되어 있어 테스트 편향 가능

#### 11.4.3 권장 테스트 설정

```python
# 테스트 시 caption에서 진단명을 제거하고 사용
def sanitize_caption_for_symptom_test(caption: str, disease_label: str) -> str:
    """caption에서 진단명을 제거하여 편향 방지"""
    # 진단명 변형들 제거
    sanitized = caption

    # disease_label 및 변형 제거
    patterns = [
        disease_label,
        disease_label.lower(),
        disease_label.replace(" ", "_"),
        disease_label.replace("_", " "),
    ]

    for pattern in patterns:
        sanitized = sanitized.replace(pattern, "[CONDITION]")

    return sanitized
```

---

## 12. 최종 테스트 로직 설명

### 12.1 증상 통합 진단 파이프라인 (제안)

증상 분석 스키마를 통합한 새로운 진단 파이프라인입니다.

```
┌─────────────────────────────────────────────────────────────────┐
│                    입력 (Input)                                  │
├─────────────────────────────────────────────────────────────────┤
│  • 이미지 경로 (image_path)                                      │
│  • 텍스트 정보 (caption 또는 user_reported_symptoms)             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              Step 0: 증상 정보 추출 (NEW)                        │
├─────────────────────────────────────────────────────────────────┤
│  SymptomExtractor.extract_with_vlm(caption_text, image_path)    │
│  → DermatologySymptomSchema 생성                                 │
│  → 구조화된 증상 정보: pruritus, pain, erythema, scaling...     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              Step 1: 이미지 분석 (기존 + 증상 통합)              │
├─────────────────────────────────────────────────────────────────┤
│  • VLM으로 시각적 특징 추출 (morphology, color, distribution)   │
│  • 증상 스키마 정보를 프롬프트에 통합                            │
│  • ObservationResult.symptoms에 구조화된 증상 저장               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              Step 2: 증상 기반 카테고리 필터링 (NEW)             │
├─────────────────────────────────────────────────────────────────┤
│  증상-질환 연관 규칙 적용:                                       │
│  • pruritus + scaling → eczema, psoriasis 우선 탐색             │
│  • tenderness + discharge → infectious 우선 탐색                │
│  • 기존 VLM 기반 분류와 결합                                     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              Step 3-5: 계층적 탐색 (기존)                        │
├─────────────────────────────────────────────────────────────────┤
│  • 하위 카테고리 분류                                            │
│  • 감별 진단 (증상 정보 활용)                                    │
│  • 최종 진단 및 신뢰도 계산                                      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    출력 (Output)                                 │
├─────────────────────────────────────────────────────────────────┤
│  {                                                               │
│    "primary_diagnosis": "allergic contact dermatitis",          │
│    "confidence": 0.85,                                           │
│    "differential_diagnoses": ["eczema", "psoriasis"],           │
│    "symptoms_analysis": {                                        │
│      "extracted_symptoms": {...},  // 구조화된 증상              │
│      "symptom_diagnosis_correlation": 0.78  // 증상-진단 상관    │
│    },                                                            │
│    "reasoning": "..."                                            │
│  }                                                               │
└─────────────────────────────────────────────────────────────────┘
```

### 12.2 증상 통합 DermatologyAgent 코드 수정 방안

```python
# dermatology_agent.py 수정안

class DermatologyAgentWithSymptoms(DermatologyAgent):
    """증상 분석을 통합한 확장 에이전트"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.symptom_extractor = SymptomExtractor(vlm_model=self.vlm)

        # 증상-질환 카테고리 연관 규칙
        self.symptom_category_hints = {
            "pruritus": ["inflammatory", "non-infectious", "eczema", "allergic"],
            "scaling": ["inflammatory", "non-infectious", "psoriasis", "eczema"],
            "tenderness": ["inflammatory", "infectious", "bacterial"],
            "discharge": ["inflammatory", "infectious", "bacterial", "pyoderma"],
            "fever": ["inflammatory", "infectious"],
        }

    def diagnose_with_symptoms(
        self,
        image_path: str,
        symptom_text: str = "",  # caption 또는 사용자 입력 증상
        max_depth: int = 4
    ) -> dict:
        """증상 정보를 포함한 진단"""

        state = DiagnosisState()

        # Step 0: 증상 추출 (NEW)
        if symptom_text:
            symptom_schema = self.symptom_extractor.extract_with_vlm(
                symptom_text, image_path
            )
            state.symptom_analysis = symptom_schema

            # 증상 기반 카테고리 힌트 생성
            category_hints = self._get_category_hints_from_symptoms(symptom_schema)
            state.category_hints = category_hints

        # Step 1: 초기 평가 (증상 정보 통합)
        state = self.step_initial_assessment_with_symptoms(image_path, state)

        # Step 2-5: 기존 로직 (카테고리 힌트 활용)
        state = self.step_category_classification(image_path, state)
        state = self.step_subcategory_classification(image_path, state, max_depth)
        state = self.step_differential_diagnosis(image_path, state)
        state = self.step_final_diagnosis(image_path, state)

        return self._build_result_with_symptoms(state)

    def _get_category_hints_from_symptoms(
        self,
        symptom_schema: DermatologySymptomSchema
    ) -> List[str]:
        """증상에서 카테고리 힌트 추출"""
        hints = []

        if symptom_schema.pruritus:
            hints.extend(self.symptom_category_hints.get("pruritus", []))
        if symptom_schema.scaling:
            hints.extend(self.symptom_category_hints.get("scaling", []))
        if symptom_schema.tenderness:
            hints.extend(self.symptom_category_hints.get("tenderness", []))
        if symptom_schema.discharge:
            hints.extend(self.symptom_category_hints.get("discharge", []))
        if symptom_schema.fever:
            hints.extend(self.symptom_category_hints.get("fever", []))

        # 빈도 기반 정렬
        from collections import Counter
        hint_counts = Counter(hints)
        return [h for h, _ in hint_counts.most_common()]

    def step_initial_assessment_with_symptoms(
        self,
        image_path: str,
        state: DiagnosisState
    ) -> DiagnosisState:
        """증상 정보를 포함한 초기 평가"""

        # 기존 프롬프트에 증상 정보 추가
        symptom_info = ""
        if hasattr(state, 'symptom_analysis') and state.symptom_analysis:
            schema = state.symptom_analysis
            symptom_parts = []

            if schema.pruritus:
                symptom_parts.append(f"pruritus ({schema.pruritus_severity.value})")
            if schema.pain:
                symptom_parts.append(f"pain ({schema.pain_type})")
            if schema.tenderness:
                symptom_parts.append("tenderness")
            if schema.burning:
                symptom_parts.append("burning sensation")
            if schema.scaling:
                symptom_parts.append("scaling")
            if schema.erythema:
                symptom_parts.append("erythema")

            if symptom_parts:
                symptom_info = f"\n\nReported symptoms: {', '.join(symptom_parts)}"

        # 수정된 프롬프트
        prompt = self.prompts["initial_assessment"] + symptom_info + """

Additionally, consider the reported symptoms when analyzing the image.
Correlate visual findings with the reported symptoms."""

        response, success = self._call_vlm(prompt, image_path, state, "initial_assessment")
        # ... 나머지 기존 로직

        return state
```

### 12.3 테스트 스크립트 예시

```python
#!/usr/bin/env python3
"""
증상 분석 통합 테스트 스크립트

random100.csv의 caption을 증상 정보로 사용하여 테스트
"""

import pandas as pd
from pathlib import Path
from derm1m_exp.experiments.vlm_wrapper import GPT4oWrapper
# 새로 추가할 모듈들
from symptom_schema import DermatologySymptomSchema, SymptomExtractor
from dermatology_agent_with_symptoms import DermatologyAgentWithSymptoms

def run_symptom_integration_test(
    csv_path: str,
    image_base_dir: str,
    num_samples: int = 10,
    use_truncated_caption: bool = False
):
    """증상 통합 테스트 실행"""

    # 데이터 로드
    df = pd.read_csv(csv_path)
    df = df.head(num_samples)

    # VLM 및 에이전트 초기화
    vlm = GPT4oWrapper(api_key=os.environ["OPENAI_API_KEY"], use_labels_prompt=False)
    agent = DermatologyAgentWithSymptoms(vlm_model=vlm, verbose=True)

    results = []

    for idx, row in df.iterrows():
        image_path = str(Path(image_base_dir) / row['filename'])
        gt_label = row['disease_label']

        # caption 또는 truncated_caption 사용
        if use_truncated_caption:
            symptom_text = row.get('truncated_caption', '')
        else:
            symptom_text = row.get('caption', '')

        # 진단명 제거 (편향 방지)
        symptom_text = sanitize_caption_for_symptom_test(symptom_text, gt_label)

        print(f"\n[{idx}] Processing: {row['filename']}")
        print(f"    Symptom text (first 100 chars): {symptom_text[:100]}...")

        # 진단 수행
        result = agent.diagnose_with_symptoms(
            image_path=image_path,
            symptom_text=symptom_text,
            max_depth=4
        )

        result['ground_truth'] = gt_label
        result['sample_id'] = idx
        result['filename'] = row['filename']
        results.append(result)

        print(f"    GT: {gt_label}")
        print(f"    Pred: {result['primary_diagnosis']}")
        print(f"    Symptoms extracted: {result.get('symptoms_analysis', {})}")

    return results

def compare_with_and_without_symptoms(csv_path: str, image_base_dir: str):
    """증상 정보 유무에 따른 성능 비교"""

    print("=" * 60)
    print("실험 1: 증상 정보 없이 진단")
    print("=" * 60)
    results_without_symptoms = run_baseline_diagnosis(csv_path, image_base_dir)

    print("\n" + "=" * 60)
    print("실험 2: caption 기반 증상 정보 포함 진단")
    print("=" * 60)
    results_with_symptoms = run_symptom_integration_test(
        csv_path, image_base_dir, use_truncated_caption=False
    )

    print("\n" + "=" * 60)
    print("실험 3: truncated_caption 기반 증상 정보 포함 진단")
    print("=" * 60)
    results_with_truncated = run_symptom_integration_test(
        csv_path, image_base_dir, use_truncated_caption=True
    )

    # 결과 비교
    evaluator = HierarchicalEvaluator()

    print("\n" + "=" * 60)
    print("결과 비교")
    print("=" * 60)

    for name, results in [
        ("Without Symptoms", results_without_symptoms),
        ("With Caption", results_with_symptoms),
        ("With Truncated Caption", results_with_truncated)
    ]:
        gts = [[r['ground_truth']] for r in results]
        preds = [[r['primary_diagnosis']] for r in results]
        eval_result = evaluator.evaluate_batch(gts, preds)

        print(f"\n{name}:")
        print(f"  Exact Match: {eval_result.exact_match:.4f}")
        print(f"  Hierarchical F1: {eval_result.hierarchical_f1:.4f}")
        print(f"  Partial Credit: {eval_result.avg_partial_credit:.4f}")

if __name__ == "__main__":
    compare_with_and_without_symptoms(
        csv_path="dataset/random100.csv",
        image_base_dir="/path/to/images"
    )
```

### 12.4 예상 결과 및 평가 항목

| 평가 항목 | 설명 |
|-----------|------|
| **증상 추출 정확도** | caption에서 얼마나 정확하게 증상을 추출했는지 |
| **증상-진단 상관관계** | 추출된 증상이 최종 진단과 얼마나 일치하는지 |
| **성능 향상 정도** | 증상 정보 추가로 인한 정확도 향상 |
| **잘못된 증상의 영향** | 부정확한 증상 정보가 진단에 미치는 부정적 영향 |

### 12.5 결론 및 권장사항

1. **caption 사용 권장**: `symptoms` 컬럼 대신 `caption` 사용이 더 풍부한 정보 제공
2. **편향 방지 필수**: caption에서 진단명을 제거하여 테스트
3. **단계적 접근**:
   - Phase 1: 키워드 기반 증상 추출 테스트
   - Phase 2: VLM 기반 고급 증상 추출
   - Phase 3: 증상-진단 연관 규칙 최적화
4. **평가 메트릭 추가**: 증상 추출 정확도, 증상-진단 상관관계 평가

---

## 부록

### A. 파일별 주요 클래스/함수 요약

| 파일 | 주요 클래스/함수 |
|------|------------------|
| `dermatology_agent.py` | `DermatologyAgent`, `DiagnosisState`, `ObservationResult` |
| `react_agent.py` | `ReActDermatologyAgent`, `ThoughtStep`, `DiagnosisResult` |
| `dermatology_agent_sam.py` | `DermatologyAgentSAM`, `SegmentationResult` |
| `ontology_utils.py` | `OntologyTree` |
| `evaluation_metrics.py` | `HierarchicalEvaluator`, `EvaluationResult` |
| `experiment_utils.py` | `AgentTrace`, `MethodResult`, `MethodEvaluation` |
| `run_comparison_experiment.py` | `run_*()` 함수들, CLI 진입점 |
| `vlm_wrapper.py` | `GPT4oWrapper` |
| `sam_wrapper.py` | `SAMBaselineWrapper` |

### B. 환경 변수

```bash
OPENAI_API_KEY       # OpenAI API 키 (필수)
CUDA_VISIBLE_DEVICES # GPU 지정 (로컬 모델 사용 시)
```

### C. 참고 자료

- Derm1M Dataset: 대규모 피부 질환 이미지 데이터셋
- SAM (Segment Anything Model): Meta의 범용 세그멘테이션 모델
- ReAct Pattern: Reasoning + Acting 반복 패턴

---

**문서 버전**: 1.0
**작성일**: 2025-12-05
**작성자**: Claude (DermAgent 코드베이스 분석)
