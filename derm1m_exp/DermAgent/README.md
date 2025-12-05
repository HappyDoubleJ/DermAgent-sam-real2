# DermAgent - 완전 가이드

> Derm1M 데이터셋의 계층적 질병 분류 온톨로지를 활용한 피부과 진단 에이전트 및 평가 시스템

**마지막 업데이트**: 2025-11-26 | **상태**: ✅ 모든 모듈 정상 작동 확인

---

## 📑 목차

1. [프로젝트 개요](#프로젝트-개요)
2. [디렉터리 구조](#디렉터리-구조)
3. [빠른 시작](#빠른-시작)
4. [주요 기능](#주요-기능)
5. [평가 시스템](#평가-시스템)
6. [진단 에이전트](#진단-에이전트)
7. [사용 가이드](#사용-가이드)
8. [온톨로지 구조](#온톨로지-구조)
9. [문제 해결](#문제-해결)
10. [변경 이력](#변경-이력)

---

## 프로젝트 개요

### 🌟 주요 특징

- **계층적 온톨로지**: 369개 피부질환을 5단계 계층 구조로 조직화
- **자동 경로 감지**: `ontology.json` 파일을 자동으로 찾아 로드
- **계층적 평가**: 단순 정확도가 아닌 질병 분류의 계층 구조를 고려한 평가
- **다양한 메트릭**: Exact Match, Hierarchical F1, Partial Credit, Level-wise Accuracy
- **AI 진단 에이전트**: ReAct 패턴 기반 체계적 진단 프레임워크
- **VLM 통합**: GPT-4o, Qwen-VL, InternVL 지원
- **유연한 사용**: 자동 경로, project_path, 직접 경로 지정 등 3가지 방법

### ⚙️ 요구사항

```
Python 3.7+
numpy
torch                  # 에이전트 사용 시
transformers           # 에이전트 사용 시
openai                 # GPT-4o 사용 시
qwen-vl-utils          # Qwen-VL 사용 시
tqdm                   # 진행 표시용
```

---

## 디렉터리 구조

```
/home/work/wonjun/DermAgent/
├── project_path.py                           # 프로젝트 경로 설정
├── dataset/
│   └── Derm1M/
│       ├── ontology.json                     # 온톨로지 파일 (369개 질환)
│       └── random_samples_100/
│           └── sampled_data.csv
│
└── derm1m_exp/
    └── DermAgent/
        ├── COMPLETE_GUIDE.md                 # 이 파일 - 통합 가이드
        ├── README.md                         # 프로젝트 개요
        ├── USAGE_GUIDE.md                    # 평가 시스템 가이드
        ├── AGENT_GUIDE.md                    # 에이전트 가이드
        ├── STRUCTURE.md                      # 구조 및 실행 방법
        ├── 경로수정_완료.md                  # 경로 수정 내역
        │
        ├── eval/                             # 평가 모듈 (40KB)
        │   ├── ontology_utils.py             # 온톨로지 트리 관리
        │   ├── evaluation_metrics.py         # 계층적 평가 메트릭
        │   └── example_usage.py              # 사용 예제
        │
        └── agent/                            # 진단 에이전트 (93KB)
            ├── dermatology_agent.py          # 기본 진단 에이전트
            ├── react_agent.py                # ReAct 패턴 에이전트
            ├── pipeline.py                   # 통합 파이프라인
            └── run_agent.py                  # 에이전트 실행 스크립트
```

### 파일 크기 및 통계

| 모듈 | 파일 수 | 총 크기 | 주요 기능 |
|------|---------|---------|-----------|
| **eval/** | 3 | 40KB | 온톨로지 관리, 평가 메트릭 |
| **agent/** | 4 | 93KB | 진단 에이전트, 파이프라인 |
| **문서** | 6 | 45KB | 가이드 및 문서화 |

---

## 빠른 시작

### 1️⃣ 평가 시스템 예제

```bash
cd /home/work/wonjun/DermAgent/derm1m_exp/DermAgent/eval
python example_usage.py
```

**출력 예시:**
```
✓ Ontology loaded from: /home/work/wonjun/DermAgent/dataset/Derm1M/ontology.json

총 노드 수: 369
최대 깊이: 5
리프 노드: 317

Hierarchical F1: 0.8000
Partial Credit: 0.7500
```

### 2️⃣ 에이전트 실행

```bash
cd /home/work/wonjun/DermAgent/derm1m_exp/DermAgent/agent
OPENAI_API_KEY=sk-xxx \
python run_agent.py \
    --input_csv /path/to/sampled_data.csv \
    --image_dir /path/to/images \
    --output results.json \
    --model gpt \
    --verbose
```

첫 실행 시 온톨로지가 자동으로 탐색되고, 결과가 `results.json`에 저장됩니다.

### 3️⃣ Python 코드에서 사용

```python
import sys
sys.path.append('/home/work/wonjun/DermAgent/derm1m_exp/DermAgent/eval')

from ontology_utils import OntologyTree
from evaluation_metrics import HierarchicalEvaluator

# 자동 경로로 초기화
tree = OntologyTree()
evaluator = HierarchicalEvaluator()

# 단일 샘플 평가
result = evaluator.evaluate_single(
    gt_labels=["Tinea corporis"],
    pred_labels=["Tinea pedis"]
)

print(f"Hierarchical F1: {result['hierarchical_f1']:.4f}")
print(f"Partial Credit: {result['partial_credit']:.4f}")
```

---

## 주요 기능

### 1. 온톨로지 트리 관리 (`eval/ontology_utils.py`)

**369개 피부질환의 계층적 구조를 관리합니다.**

```python
from ontology_utils import OntologyTree

tree = OntologyTree()  # 자동 경로 감지

# 경로 추출
path = tree.get_path_to_root("Tinea corporis")
# ['Tinea corporis', 'fungal', 'infectious', 'inflammatory', 'root']

# 거리 계산
distance = tree.get_hierarchical_distance("Tinea corporis", "Tinea pedis")
# 2 (같은 fungal 카테고리)

distance = tree.get_hierarchical_distance("Tinea corporis", "Psoriasis")
# 5 (다른 브랜치)

# 공통 조상 찾기
lca = tree.get_lca("Tinea corporis", "Tinea pedis")
# 'fungal'

# 자식 노드 탐색
children = tree.get_children("fungal")
# ['Kerion', 'Tinea corporis', 'Tinea pedis', 'Candidiasis', ...]

# 형제 노드 찾기
siblings = tree.get_siblings("Tinea corporis")
# ['Kerion', 'Tinea pedis', 'Candidiasis', ...]

# 라벨 정규화
canonical = tree.get_canonical_name("tinea corporis")  # "Tinea corporis"
canonical = tree.get_canonical_name("PSORIASIS")       # "Psoriasis"

# 유효성 검사
valid_labels = tree.filter_valid_labels([
    "Tinea corporis",
    "invalid_disease",
    "Psoriasis"
])
# ['Tinea corporis', 'Psoriasis']
```

### 2. 계층적 평가 메트릭 (`eval/evaluation_metrics.py`)

**온톨로지를 고려한 공정한 평가를 제공합니다.**

```python
from evaluation_metrics import HierarchicalEvaluator

evaluator = HierarchicalEvaluator()  # 자동 경로 감지

# 단일 샘플 평가
result = evaluator.evaluate_single(
    gt_labels=["Tinea corporis"],
    pred_labels=["Tinea pedis"]
)

print(result)
# {
#   'valid': True,
#   'exact_match': 0.0,
#   'hierarchical_f1': 0.8000,
#   'hierarchical_precision': 0.8000,
#   'hierarchical_recall': 0.8000,
#   'avg_min_distance': 2.0,
#   'partial_credit': 0.7500,
#   'level_matches': [1, 1, 1, 0, 0]
# }

# 배치 평가
ground_truths = [
    ["Tinea corporis"],
    ["Psoriasis"],
    ["Eczema"],
]

predictions = [
    ["Tinea pedis"],
    ["Psoriasis"],
    ["Atopic dermatitis"],
]

result = evaluator.evaluate_batch(ground_truths, predictions)
evaluator.print_evaluation_report(result)
```

**평가 리포트 예시:**
```
============================================================
HIERARCHICAL EVALUATION REPORT
============================================================

[Sample Statistics]
  Total samples: 3
  Valid samples: 3
  Skipped samples: 0

[Basic Metrics]
  Exact Match Accuracy: 0.3333
  Partial Match Ratio: 0.3333

[Hierarchical Metrics]
  Hierarchical Precision: 0.7333
  Hierarchical Recall: 0.7333
  Hierarchical F1: 0.7333
  Avg Min Distance: 2.0000

[Partial Credit]
  Avg Partial Credit Score: 0.5833
  Avg Ancestor Match Score: 0.4200

[Level-wise Accuracy]
  Level 1: 1.0000    ← 대분류는 거의 맞춤
  Level 2: 1.0000
  Level 3: 0.6667
  Level 4: 0.0000    ← 구체 질환은 어려움
  Level 5: 0.0000
============================================================
```

### 3. 진단 에이전트 (`agent/dermatology_agent.py`)

**온톨로지 기반 계층적 탐색과 도구 기반 추론을 활용합니다.**

```python
import sys
sys.path.append('/home/work/wonjun/DermAgent/derm1m_exp/DermAgent/agent')
sys.path.append('/home/work/wonjun/DermAgent/derm1m_exp/DermAgent/eval')

from dermatology_agent import DermatologyAgent
from pipeline import GPT4oVLM

vlm = GPT4oVLM(api_key="YOUR_OPENAI_KEY")  # 또는 OPENAI_API_KEY 환경 변수 사용
agent = DermatologyAgent(
    ontology_path=None,  # 자동 탐색
    vlm_model=vlm,
    verbose=True
)

# 진단 실행
result = agent.diagnose("/path/to/image.jpg", max_depth=4)

print(result)
# {
#   "image_path": "/path/to/image.jpg",
#   "final_diagnosis": ["Tinea corporis"],
#   "diagnosis_path": ["inflammatory", "infectious", "fungal", "Tinea corporis"],
#   "confidence_scores": {
#     "inflammatory": 0.85,
#     "infectious": 0.75,
#     "fungal": 0.80,
#     "Tinea corporis": 0.70
#   },
#   "observations": {
#     "morphology": ["annular", "scaly"],
#     "color": ["red"],
#     "location": "trunk"
#   }
# }
```

**에이전트 도구:**
- `OntologyNavigator`: 온톨로지 트리 탐색
- `DifferentialDiagnosisTool`: 후보 질환 비교 및 점수화

### 4. ReAct 에이전트 (`agent/react_agent.py`)

**Reasoning + Acting 패턴으로 체계적인 진단을 수행합니다.**

```python
from react_agent import ReActDermatologyAgent
from pipeline import GPT4oVLM

# ReAct 에이전트 생성
vlm = GPT4oVLM(api_key="YOUR_OPENAI_KEY")
agent = ReActDermatologyAgent(
    ontology_path=None,
    vlm_model=vlm,
    max_steps=8,
    verbose=True
)

# 진단 실행
result = agent.diagnose("/path/to/image.jpg")

print(result.primary_diagnosis)      # "Tinea corporis"
print(result.differential_diagnoses) # ["Psoriasis", "Eczema"]
print(result.confidence)             # 0.75
print(result.ontology_path)          # ['Tinea corporis', 'fungal', ...]
```

**추론 과정:**
```
Step 1: OBSERVE → 이미지에서 임상 특징 추출
Step 2: NAVIGATE → 온톨로지 대분류 식별 (inflammatory)
Step 3: NAVIGATE → 소분류 좁히기 (infectious → fungal)
Step 4: COMPARE → 후보 비교 (Tinea corporis vs Tinea pedis)
Step 5: VERIFY → 진단 일관성 검증
Step 6: CONCLUDE → 최종 진단 및 신뢰도
```

### 5. 통합 파이프라인 (`agent/pipeline.py`)

**ReAct 에이전트 + 계층적 평가를 통합한 완전한 파이프라인입니다.**

```bash
# GPT-4o 사용
python pipeline.py \
    --input data.csv \
    --output results.json \
    --image_dir /path/to/images \
    --model gpt \
    --api_key YOUR_API_KEY

# Qwen-VL 사용 (GPU)
CUDA_VISIBLE_DEVICES=0,1 python pipeline.py \
    --input data.csv \
    --output results.json \
    --model qwen \
    --model_path Qwen/Qwen2-VL-7B-Instruct

# InternVL 사용 (GPU)
python pipeline.py \
    --input data.csv \
    --output results.json \
    --model internvl \
    --model_path OpenGVLab/InternVL2-8B
```

**입력 CSV 형식:**
```csv
filename,disease_label
image001.jpg,Tinea corporis
image002.jpg,"Psoriasis, Eczema"
image003.jpg,Atopic dermatitis
```

---

## 평가 시스템

### 평가 메트릭 상세

#### 1. **Exact Match**
- 정확히 일치하면 1, 아니면 0
- 가장 엄격한 평가 기준

#### 2. **Hierarchical Distance**
- 두 노드 간의 트리 상 거리
- 예: `Tinea corporis ↔ Tinea pedis = 2` (같은 fungal 카테고리)
- 예: `Tinea corporis ↔ Psoriasis = 5` (다른 브랜치)

#### 3. **Hierarchical F1**
- 계층적 유사도 기반 Precision/Recall/F1
- **권장 값**: 0.8 이상

공식:
```
Similarity(A, B) = |Ancestors(A) ∩ Ancestors(B)| / |Ancestors(A) ∪ Ancestors(B)|
Precision = Avg(max similarity for each prediction)
Recall = Avg(max similarity for each ground truth)
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

#### 4. **Partial Credit Score**
- 공통 조상까지의 경로 비율
- Level 4까지 맞춰야 하는데 Level 3까지만 맞으면 0.75점
- **권장 값**: 0.7 이상

공식:
```
Partial Credit = (공통 조상 깊이) / (GT 깊이)
```

#### 5. **Level-wise Accuracy**
- 각 레벨에서의 정확도
- Level 1 (대분류): inflammatory vs proliferations
- Level 4-5 (구체 질환): Tinea corporis

**권장 기준:**

| 메트릭 | 권장 값 | 설명 |
|--------|---------|------|
| Exact Match | 높을수록 좋음 | 정확한 질환명 일치 |
| Hierarchical F1 | 0.8+ | 계층적 유사도 |
| Avg Min Distance | 낮을수록 좋음 | 평균 트리 거리 (0이 이상적) |
| Partial Credit | 0.7+ | 부분 점수 |
| Level 1 Accuracy | 0.9+ | 대분류 정확도 |
| Level 4 Accuracy | 0.4+ | 구체 질환 정확도 (도전적) |

---

## 진단 에이전트

### 에이전트 아키텍처

```
┌─────────────────────────────────────────┐
│         DermatologyAgent                │
│                                         │
│  ┌──────────────┐  ┌──────────────┐   │
│  │  VLM Model   │  │  Ontology    │   │
│  │  (GPT/Qwen)  │  │  Tree        │   │
│  └──────────────┘  └──────────────┘   │
│         │                  │           │
│         ▼                  ▼           │
│  ┌──────────────────────────────┐     │
│  │      Agent Tools             │     │
│  │  - OntologyNavigator         │     │
│  │  - DifferentialDiagnosisTool │     │
│  └──────────────────────────────┘     │
│                                         │
│  Diagnosis Pipeline:                   │
│  1. Initial Assessment                 │
│  2. Category Classification (L1)       │
│  3. Subcategory Classification (L2-3)  │
│  4. Differential Diagnosis             │
│  5. Final Diagnosis                    │
└─────────────────────────────────────────┘
```

### VLM 모델 지원

| 모델 | 제공자 | 사용 방법 |
|------|--------|-----------|
| **GPT-4o** | OpenAI | `--model gpt --api_key KEY` |
| **Qwen-VL** | Alibaba | `--model qwen --model_path PATH` |
| **InternVL** | OpenGVLab | `--model internvl --model_path PATH` |

---

## 사용 가이드

### 경로 설정 (자동 감지)

모든 모듈은 **자동 경로 감지**를 지원합니다:

**우선순위:**
1. `project_path.py` 사용 (우선순위 1)
2. 상대 경로로 찾기 (우선순위 2)
3. 현재 디렉토리 기준 (우선순위 3)

```python
# 방법 1: 자동 경로 (권장)
tree = OntologyTree()
evaluator = HierarchicalEvaluator()
agent = DermatologyAgent()

# 방법 2: project_path 사용
import sys
sys.path.append('/path/to/DermAgent')
import project_path
tree = OntologyTree(project_path.ONTOLOGY_PATH)

# 방법 3: 직접 경로 지정
tree = OntologyTree("/explicit/path/to/ontology.json")
```

### 실행 예제

#### 1. 평가 시스템 데모

```bash
cd /home/work/wonjun/DermAgent/derm1m_exp/DermAgent/eval

# 전체 예제 실행
python example_usage.py

# 온톨로지 유틸리티만
python ontology_utils.py

# 평가 메트릭만
python evaluation_metrics.py
```

#### 2. 에이전트 실행

```bash
cd /home/work/wonjun/DermAgent/derm1m_exp/DermAgent/agent

# 기본 에이전트
python dermatology_agent.py

# ReAct 에이전트
python react_agent.py

# CSV 데이터로 실행 (GPT-4o)
python run_agent.py \
    --input_csv /path/to/data.csv \
    --image_dir /path/to/images \
    --output results.json \
    --model gpt \
    --api_key YOUR_API_KEY \
    --verbose
```

#### 3. 파이프라인 실행

```bash
cd /home/work/wonjun/DermAgent/derm1m_exp/DermAgent/agent

# 실제 데이터 처리
python pipeline.py \
    --input /path/to/data.csv \
    --output results.json \
    --image_dir /path/to/images \
    --model gpt \
    --api_key YOUR_API_KEY \
    --max_steps 8 \
    --verbose
```

### Python API 사용

#### 평가 시스템

```python
import sys
sys.path.append('/home/work/wonjun/DermAgent/derm1m_exp/DermAgent/eval')

from ontology_utils import OntologyTree
from evaluation_metrics import HierarchicalEvaluator

# 초기화
tree = OntologyTree()
evaluator = HierarchicalEvaluator()

# 배치 평가
ground_truths = [["Tinea corporis"], ["Psoriasis"], ["Eczema"]]
predictions = [["Tinea pedis"], ["Psoriasis"], ["Atopic dermatitis"]]

result = evaluator.evaluate_batch(ground_truths, predictions)
evaluator.print_evaluation_report(result)
```

#### 에이전트

```python
import sys
sys.path.append('/home/work/wonjun/DermAgent/derm1m_exp/DermAgent/agent')
sys.path.append('/home/work/wonjun/DermAgent/derm1m_exp/DermAgent/eval')

from dermatology_agent import DermatologyAgent
from pipeline import GPT4oVLM

# 에이전트 생성
vlm = GPT4oVLM(api_key="YOUR_OPENAI_KEY")
agent = DermatologyAgent(
    ontology_path=None,
    vlm_model=vlm,
    verbose=True
)

# 진단 실행
result = agent.diagnose("/path/to/image.jpg", max_depth=4)
print(result["final_diagnosis"])
print(result["diagnosis_path"])
```

---

## 온톨로지 구조

### 전체 구조

```
root
├── Hair diseases (모발 질환)
│   ├── Alopecia areata
│   └── ...
│
├── Nail diseases (조갑 질환)
│   ├── Onychomycosis
│   └── ...
│
├── inflammatory (염증성)                    ← Level 1
│   ├── infectious (감염성)                  ← Level 2
│   │   ├── bacterial (세균성)              ← Level 3
│   │   │   ├── Cellulitis                  ← Level 4
│   │   │   ├── Impetigo
│   │   │   └── Folliculitis
│   │   │
│   │   ├── fungal (진균성)                 ← Level 3
│   │   │   ├── Tinea corporis             ← Level 4
│   │   │   ├── Tinea pedis
│   │   │   ├── Tinea capitis
│   │   │   └── Candidiasis
│   │   │
│   │   ├── viral (바이러스성)              ← Level 3
│   │   │   ├── Herpes simplex
│   │   │   └── Molluscum contagiosum
│   │   │
│   │   └── parasitic (기생충성)            ← Level 3
│   │       └── Scabies
│   │
│   └── non-infectious (비감염성)            ← Level 2
│       ├── Eczema                          ← Level 3
│       │   ├── Atopic dermatitis           ← Level 4
│       │   └── Contact dermatitis
│       │
│       ├── Psoriasis                       ← Level 3
│       │   ├── Plaque psoriasis            ← Level 4
│       │   └── Guttate psoriasis
│       │
│       └── ...
│
├── proliferations (증식성)                  ← Level 1
│   ├── benign (양성)                       ← Level 2
│   │   ├── melanocytic                     ← Level 3
│   │   │   └── Nevus                       ← Level 4
│   │   │
│   │   └── non-melanocytic                 ← Level 3
│   │       └── Seborrheic keratosis        ← Level 4
│   │
│   └── malignant (악성)                    ← Level 2
│       ├── Melanoma                        ← Level 3
│       ├── Basal cell carcinoma
│       └── Squamous cell carcinoma
│
└── ...
```

### 통계

- **총 노드**: 369개
- **최대 깊이**: 5 레벨
- **리프 노드**: 317개 (실제 질환)
- **주요 카테고리**: inflammatory, proliferations, hair diseases, nail diseases 등

### 예시: Tinea corporis의 전체 경로

```
Level 0: root
Level 1: inflammatory
Level 2: infectious
Level 3: fungal
Level 4: Tinea corporis
```

---

## 문제 해결

### ImportError 발생 시

**증상:**
```python
ModuleNotFoundError: No module named 'ontology_utils'
```

**해결 방법:**
```python
import sys
sys.path.append('/home/work/wonjun/DermAgent/derm1m_exp/DermAgent/eval')
sys.path.append('/home/work/wonjun/DermAgent/derm1m_exp/DermAgent/agent')
```

### ontology.json을 찾을 수 없다는 오류

**증상:**
```
FileNotFoundError: ontology.json 파일을 찾을 수 없습니다.
```

**해결 방법:**

1. 프로젝트 구조 확인:
```bash
ls /home/work/wonjun/DermAgent/dataset/Derm1M/ontology.json
```

2. 명시적으로 경로 지정:
```python
tree = OntologyTree("/home/work/wonjun/DermAgent/dataset/Derm1M/ontology.json")
```

3. project_path 확인:
```python
import sys
sys.path.append('/home/work/wonjun/DermAgent')
import project_path
print(project_path.ONTOLOGY_PATH)
```

### CUDA 메모리 부족

**증상:**
```
RuntimeError: CUDA out of memory
```

**해결 방법:**
```bash
# 멀티 GPU 사용
CUDA_VISIBLE_DEVICES=0,1 python pipeline.py ...

# 또는 배치 크기 줄이기
python pipeline.py --limit 10 ...
```

### VLM 응답 파싱 실패

**증상:**
```
Warning: Failed to parse VLM response
```

**해결 방법:**
1. verbose=True로 원시 응답 확인:
```python
agent = DermatologyAgent(verbose=True)
```

2. 프롬프트 수정하여 JSON 형식 강제

### 현재 디렉터리 확인

```bash
pwd
# 출력: /home/work/wonjun/DermAgent/derm1m_exp/DermAgent/agent
# 또는: /home/work/wonjun/DermAgent/derm1m_exp/DermAgent/eval
```

---

## 변경 이력

### 2025-11-26: 경로 수정 및 구조 개선

#### ✅ 주요 변경사항

1. **디렉터리 구조 재정리**
   - 파일들을 `agent/`와 `eval/` 폴더로 분리
   - 모듈화된 구조로 개선

2. **Import 경로 수정**
   - `agent/dermatology_agent.py`: eval 폴더 경로 추가
   - `agent/react_agent.py`: eval 폴더 경로 추가
   - `agent/pipeline.py`: eval 및 agent 폴더 경로 추가
   - `agent/run_agent.py`: 경로 설정 수정

3. **문법 오류 수정**
   - `agent/react_agent.py`: 들여쓰기 오류 수정
   - `agent/react_agent.py`: try-except 블록 완성

4. **자동 경로 감지 유지**
   - `ontology_path=None`이면 자동으로 찾기
   - project_path.py → 상대 경로 → 현재 디렉토리 순으로 탐색

5. **문서화 개선**
   - COMPLETE_GUIDE.md 추가 (통합 가이드)
   - STRUCTURE.md 추가 (구조 및 실행 방법)
   - 경로수정_완료.md 추가 (수정 내역)

#### 🧪 테스트 결과

모든 파일이 정상 작동함을 확인:

| 파일 | 상태 | 테스트 명령 |
|------|------|-------------|
| eval/example_usage.py | ✅ | `python eval/example_usage.py` |
| eval/ontology_utils.py | ✅ | `python eval/ontology_utils.py` |
| eval/evaluation_metrics.py | ✅ | `python eval/evaluation_metrics.py` |
| agent/dermatology_agent.py | ✅ | `python agent/dermatology_agent.py` |
| agent/react_agent.py | ✅ | `python agent/react_agent.py` |
| agent/run_agent.py | ✅ | `python agent/run_agent.py --help` |
| agent/pipeline.py | ✅ | `python agent/pipeline.py --help` |

---

## 학습 경로 추천

### 초급: 평가 시스템 이해

1. `eval/example_usage.py` 실행
2. README.md 읽기
3. USAGE_GUIDE.md 참고
4. 단일 샘플 평가 실습
5. 배치 평가 실습

### 중급: 에이전트 기본 이해

1. `agent/dermatology_agent.py` 데모 실행
2. 코드 읽고 구조 이해
3. 실제 VLM 연결 후 진단 테스트
4. 도구 시스템 이해 (OntologyNavigator, DifferentialDiagnosisTool)

### 고급: ReAct 에이전트

1. `agent/react_agent.py` 데모 실행
2. AGENT_GUIDE.md 읽기
3. ReAct 패턴 이해
4. 실제 VLM과 통합 테스트

### 실전: 파이프라인 사용

1. `agent/run_agent.py` 또는 `agent/pipeline.py`로 실제 데이터 처리
2. CSV 데이터 준비
3. 결과 분석 및 평가
4. 하이퍼파라미터 튜닝

---

## 추가 리소스

### 관련 파일

- **프로젝트 루트 경로**: `/home/work/wonjun/DermAgent/project_path.py`
- **온톨로지 파일**: `/home/work/wonjun/DermAgent/dataset/Derm1M/ontology.json`
- **샘플 데이터**: `/home/work/wonjun/DermAgent/dataset/Derm1M/random_samples_100/`

### 문서

- **README.md**: 프로젝트 개요 및 빠른 시작
- **USAGE_GUIDE.md**: 평가 시스템 사용법 상세 설명
- **AGENT_GUIDE.md**: 에이전트 프레임워크 완전 가이드
- **STRUCTURE.md**: 디렉터리 구조 및 실행 방법
- **경로수정_완료.md**: 경로 수정 내역
- **COMPLETE_GUIDE.md**: 이 파일 - 통합 완전 가이드

---

## 라이센스 및 인용

이 프로젝트는 Derm1M 데이터셋의 평가를 위해 개발되었습니다.

---

**프로젝트 상태**: ✅ 프로덕션 준비 완료
**마지막 업데이트**: 2025-11-26
**테스트 환경**: Python 3.7+, Linux
**메인테이너**: DermAgent Team
