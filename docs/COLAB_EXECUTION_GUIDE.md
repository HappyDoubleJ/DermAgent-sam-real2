# DermAgent 증상 통합 실험 - Google Colab 실행 가이드

## 목차
1. [환경 설정](#1-환경-설정)
2. [데이터 준비](#2-데이터-준비)
3. [실험 실행](#3-실험-실행)
4. [결과 해석](#4-결과-해석)
5. [출력 파일 설명](#5-출력-파일-설명)
6. [문제 해결](#6-문제-해결)

---

## 1. 환경 설정

### 1.1 Colab 셀 1: 기본 설정 및 저장소 클론

```python
# GPU 런타임 확인 (선택사항 - SAM 사용 시 필요)
!nvidia-smi

# 저장소 클론
!git clone https://github.com/HappyDoubleJ/DermAgent-sam-real2.git
%cd DermAgent-sam-real2

# wonjun 브랜치로 전환
!git fetch origin wonjun
!git checkout wonjun
```

### 1.2 Colab 셀 2: 의존성 설치

```python
# 기본 의존성 설치
!pip install openai pandas numpy tqdm pillow

# 평가 메트릭용
!pip install scikit-learn

# SAM 설치 (선택사항 - SAM 사용 시)
!pip install git+https://github.com/facebookresearch/segment-anything.git

# SAM 체크포인트 다운로드 (선택사항)
!mkdir -p derm1m_exp/SA-project-SAM/checkpoints
!wget -q https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth \
    -O derm1m_exp/SA-project-SAM/checkpoints/sam_vit_h_4b8939.pth
```

### 1.3 Colab 셀 3: API 키 설정

```python
import os
from google.colab import userdata

# 방법 1: Colab Secrets 사용 (권장)
try:
    os.environ["OPENAI_API_KEY"] = userdata.get('OPENAI_API_KEY')
    print("✓ API 키가 Colab Secrets에서 로드되었습니다.")
except:
    pass

# 방법 2: 직접 입력
# os.environ["OPENAI_API_KEY"] = "sk-your-api-key-here"

# 확인
if "OPENAI_API_KEY" in os.environ:
    print(f"✓ API 키 설정 완료 (길이: {len(os.environ['OPENAI_API_KEY'])})")
else:
    print("✗ API 키가 설정되지 않았습니다!")
```

---

## 2. 데이터 준비

### 2.1 Colab 셀 4: 데이터셋 업로드

```python
# 방법 1: Google Drive 마운트
from google.colab import drive
drive.mount('/content/drive')

# 데이터 경로 설정 (Google Drive에 데이터가 있는 경우)
DATA_DIR = "/content/drive/MyDrive/DermAgent_Data"
IMAGE_DIR = f"{DATA_DIR}/images"
CSV_PATH = f"{DATA_DIR}/random100.csv"

# 방법 2: 직접 업로드
# from google.colab import files
# uploaded = files.upload()

# 방법 3: 저장소 내 샘플 데이터 사용
CSV_PATH = "/content/DermAgent-sam-real2/dataset/random100.csv"
```

### 2.2 Colab 셀 5: 데이터 확인

```python
import pandas as pd

# CSV 로드 및 확인
df = pd.read_csv(CSV_PATH)
print(f"총 샘플 수: {len(df)}")
print(f"\n컬럼 목록:")
print(df.columns.tolist())

print(f"\n첫 3개 샘플:")
print(df[['filename', 'disease_label', 'symptoms']].head(3))

print(f"\nsymptoms 컬럼 분포:")
print(df['symptoms'].value_counts().head())

print(f"\ncaption 예시:")
print(df['caption'].iloc[0][:200] + "...")
```

---

## 3. 실험 실행

### 3.1 Colab 셀 6: 빠른 테스트 (5개 샘플)

```bash
%%bash
cd /content/DermAgent-sam-real2/derm1m_exp/experiments

python run_symptom_integration_experiment.py \
    --input_csv /content/DermAgent-sam-real2/dataset/random100.csv \
    --image_dir /content/drive/MyDrive/DermAgent_Data/images \
    --output_dir /content/outputs \
    --num_samples 5 \
    --sam_strategies none,center \
    --verbose
```

### 3.2 Colab 셀 7: 전체 실험 (SAM 없이)

```bash
%%bash
cd /content/DermAgent-sam-real2/derm1m_exp/experiments

# SAM 없이 증상 분석만 테스트
python run_symptom_integration_experiment.py \
    --input_csv /content/DermAgent-sam-real2/dataset/random100.csv \
    --image_dir /content/drive/MyDrive/DermAgent_Data/images \
    --output_dir /content/outputs \
    --num_samples 20 \
    --sam_strategies none \
    --verbose
```

### 3.3 Colab 셀 8: 전체 실험 (모든 SAM 전략)

```bash
%%bash
cd /content/DermAgent-sam-real2/derm1m_exp/experiments

# 모든 SAM 전략 + 증상 유무 비교
python run_symptom_integration_experiment.py \
    --input_csv /content/DermAgent-sam-real2/dataset/random100.csv \
    --image_dir /content/drive/MyDrive/DermAgent_Data/images \
    --output_dir /content/outputs \
    --num_samples 50 \
    --sam_strategies none,center,lesion_features,both \
    --verbose
```

### 3.4 Colab 셀 9: 커스텀 실험 (Python으로 세밀한 제어)

```python
import sys
sys.path.insert(0, '/content/DermAgent-sam-real2/derm1m_exp/experiments')
sys.path.insert(0, '/content/DermAgent-sam-real2/derm1m_exp/eval')

import pandas as pd
from vlm_wrapper import GPT4oWrapper
from run_symptom_integration_experiment import SymptomIntegrationExperiment

# VLM 초기화
vlm = GPT4oWrapper(
    api_key=os.environ["OPENAI_API_KEY"],
    use_labels_prompt=False
)

# 실험 초기화
experiment = SymptomIntegrationExperiment(
    vlm_model=vlm,
    verbose=True
)

# 데이터 로드
df = pd.read_csv(CSV_PATH)

# 실험 1: 증상 없이 기본 진단
results_baseline = experiment.run_experiment_batch(
    df=df,
    image_base_dir=IMAGE_DIR,
    experiment_name="baseline_no_symptom",
    sam_strategy="none",
    use_symptoms=False,
    num_samples=10
)

# 실험 2: caption에서 증상 추출하여 진단
results_with_caption = experiment.run_experiment_batch(
    df=df,
    image_base_dir=IMAGE_DIR,
    experiment_name="with_caption_symptoms",
    sam_strategy="none",
    use_symptoms=True,
    symptom_source="caption",
    num_samples=10
)

# 실험 3: truncated_caption 사용
results_with_truncated = experiment.run_experiment_batch(
    df=df,
    image_base_dir=IMAGE_DIR,
    experiment_name="with_truncated_symptoms",
    sam_strategy="none",
    use_symptoms=True,
    symptom_source="truncated_caption",
    num_samples=10
)

# 요약 출력
experiment.print_summary()

# 결과 저장
saved_files = experiment.save_results("/content/outputs")
print("\n저장된 파일:")
for name, path in saved_files.items():
    print(f"  {name}: {path}")
```

---

## 4. 결과 해석

### 4.1 Colab 셀 10: 결과 로드 및 시각화

```python
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob

# 가장 최근 결과 파일 찾기
output_dir = "/content/outputs"
summary_files = sorted(glob(f"{output_dir}/experiment_summary_*.csv"))
latest_summary = summary_files[-1] if summary_files else None

if latest_summary:
    summary_df = pd.read_csv(latest_summary)
    print("실험 요약:")
    print(summary_df.to_string())

    # 시각화
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Exact Match 비교
    ax1 = axes[0]
    summary_df.plot.bar(x='experiment_name', y='exact_match_accuracy', ax=ax1, legend=False)
    ax1.set_title('Exact Match Accuracy by Experiment')
    ax1.set_ylabel('Accuracy')
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')

    # Hierarchical F1 비교
    ax2 = axes[1]
    summary_df.plot.bar(x='experiment_name', y='hierarchical_f1', ax=ax2, legend=False, color='orange')
    ax2.set_title('Hierarchical F1 by Experiment')
    ax2.set_ylabel('F1 Score')
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(f"{output_dir}/comparison_chart.png", dpi=150, bbox_inches='tight')
    plt.show()
else:
    print("결과 파일을 찾을 수 없습니다.")
```

### 4.2 Colab 셀 11: SAM 전략별 비교

```python
# SAM 전략별 비교 파일 로드
sam_comparison_files = sorted(glob(f"{output_dir}/sam_strategy_comparison_*.csv"))
if sam_comparison_files:
    sam_df = pd.read_csv(sam_comparison_files[-1])
    print("SAM 전략별 비교:")
    print(sam_df.to_string())

    # SAM 전략별 성능 비교 차트
    if len(sam_df) > 0:
        fig, ax = plt.subplots(figsize=(12, 6))

        x = range(len(sam_df))
        width = 0.35

        bars1 = ax.bar([i - width/2 for i in x], sam_df['Exact Match (%)'].str.rstrip('%').astype(float),
                       width, label='Exact Match (%)', color='steelblue')
        bars2 = ax.bar([i + width/2 for i in x], sam_df['Hierarchical F1'].astype(float) * 100,
                       width, label='Hierarchical F1 (%)', color='coral')

        ax.set_xlabel('Experiment Configuration')
        ax.set_ylabel('Score (%)')
        ax.set_title('SAM Strategy Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels([f"{row['SAM Strategy']}\n{'w/ symptoms' if row['Use Symptoms'] else 'no symptoms'}"
                           for _, row in sam_df.iterrows()], rotation=45, ha='right')
        ax.legend()

        plt.tight_layout()
        plt.savefig(f"{output_dir}/sam_comparison_chart.png", dpi=150, bbox_inches='tight')
        plt.show()
```

### 4.3 Colab 셀 12: 상세 결과 분석

```python
# 상세 결과 로드
detailed_files = sorted(glob(f"{output_dir}/detailed_results_*.csv"))
if detailed_files:
    detailed_df = pd.read_csv(detailed_files[-1])

    print(f"총 실험 결과: {len(detailed_df)}개")
    print(f"\n실험별 분포:")
    print(detailed_df['experiment_name'].value_counts())

    # 정확도 높은/낮은 샘플 분석
    print("\n=== Exact Match 성공 샘플 ===")
    success_samples = detailed_df[detailed_df['exact_match'] == True]
    if len(success_samples) > 0:
        print(success_samples[['filename', 'ground_truth', 'prediction', 'experiment_name']].head(10))

    print("\n=== Exact Match 실패 샘플 ===")
    failed_samples = detailed_df[detailed_df['exact_match'] == False]
    if len(failed_samples) > 0:
        print(failed_samples[['filename', 'ground_truth', 'prediction', 'experiment_name']].head(10))
```

---

## 5. 출력 파일 설명

### 5.1 생성되는 CSV 파일

| 파일명 | 설명 | 용도 |
|--------|------|------|
| `detailed_results_YYYYMMDD_HHMMSS.csv` | 모든 샘플의 상세 결과 | 개별 샘플 분석 |
| `experiment_summary_YYYYMMDD_HHMMSS.csv` | 실험별 평균 메트릭 | 실험 비교 |
| `sam_strategy_comparison_YYYYMMDD_HHMMSS.csv` | SAM 전략별 성능 비교 | SAM 전략 선택 |
| `full_comparison_YYYYMMDD_HHMMSS.csv` | 샘플별 모든 실험 결과 병합 | LLM 문서화용 |
| `experiment_config_YYYYMMDD_HHMMSS.json` | 실험 설정 정보 | 재현성 |

### 5.2 상세 결과 CSV 컬럼 설명

```
sample_id              : 샘플 번호
filename               : 이미지 파일명
ground_truth           : 정답 라벨
hierarchical_gt        : 계층적 정답 라벨
prediction             : 예측 결과
confidence             : 예측 신뢰도
differential_diagnoses : 감별 진단 목록
experiment_name        : 실험 이름
sam_strategy           : SAM 전략 (none/center/lesion_features/both)
use_symptoms           : 증상 사용 여부
symptom_source         : 증상 소스 (caption/truncated_caption)
extracted_symptoms     : 추출된 증상 요약
symptom_extraction_confidence : 증상 추출 신뢰도
segmentation_score     : 세그멘테이션 품질 점수
segmentation_method    : 세그멘테이션 방법
exact_match            : 정확 일치 여부 (True/False)
hierarchical_distance  : 계층적 거리 (낮을수록 좋음)
hierarchical_f1        : 계층적 F1 점수
partial_credit         : 부분 점수
raw_response           : VLM 원본 응답
reasoning              : 진단 근거
```

### 5.3 실험 요약 CSV 컬럼 설명

```
experiment_name          : 실험 이름
sam_strategy             : SAM 전략
use_symptoms             : 증상 사용 여부
symptom_source           : 증상 소스
total_samples            : 총 샘플 수
valid_samples            : 유효 샘플 수
exact_match_accuracy     : Exact Match 정확도
avg_hierarchical_distance: 평균 계층적 거리
hierarchical_f1          : 평균 Hierarchical F1
partial_credit           : 평균 Partial Credit
avg_segmentation_score   : 평균 세그멘테이션 점수
avg_confidence           : 평균 신뢰도
```

---

## 6. 문제 해결

### 6.1 일반적인 오류 및 해결책

#### API 키 오류
```
오류: OPENAI_API_KEY가 필요합니다.

해결:
1. Colab Secrets에 OPENAI_API_KEY 추가
2. 또는 코드에서 직접 설정:
   os.environ["OPENAI_API_KEY"] = "sk-..."
```

#### 이미지 경로 오류
```
오류: 이미지 없음: youtube/xxx.jpg

해결:
1. IMAGE_DIR 경로 확인
2. 파일 존재 확인:
   !ls -la {IMAGE_DIR}/youtube/ | head
```

#### SAM 모듈 오류
```
오류: SAM 모듈 로드 실패

해결:
1. SAM 없이 실행 (--sam_strategies none)
2. SAM 재설치:
   !pip uninstall segment-anything -y
   !pip install git+https://github.com/facebookresearch/segment-anything.git
```

#### 메모리 부족
```
오류: CUDA out of memory

해결:
1. 배치 크기 줄이기:
   --num_samples 5
2. SAM 비활성화:
   --sam_strategies none
3. 런타임 재시작 후 다시 시도
```

### 6.2 결과 파일 다운로드

```python
# Colab에서 결과 다운로드
from google.colab import files
import zipfile
import os

# 결과 압축
output_dir = "/content/outputs"
zip_path = "/content/experiment_results.zip"

with zipfile.ZipFile(zip_path, 'w') as zipf:
    for root, dirs, filenames in os.walk(output_dir):
        for filename in filenames:
            file_path = os.path.join(root, filename)
            arcname = os.path.relpath(file_path, output_dir)
            zipf.write(file_path, arcname)

# 다운로드
files.download(zip_path)
```

### 6.3 Google Drive에 결과 저장

```python
# 결과를 Google Drive에 저장
import shutil

drive_output = "/content/drive/MyDrive/DermAgent_Results"
os.makedirs(drive_output, exist_ok=True)

# 결과 파일 복사
for file in glob(f"{output_dir}/*"):
    shutil.copy(file, drive_output)

print(f"결과가 {drive_output}에 저장되었습니다.")
```

---

## 부록: 빠른 시작 (한 번에 복사-붙여넣기)

```python
# === Colab 전체 실행 스크립트 ===

# 1. 설정
!git clone https://github.com/HappyDoubleJ/DermAgent-sam-real2.git 2>/dev/null || echo "이미 클론됨"
%cd /content/DermAgent-sam-real2
!git fetch origin wonjun && git checkout wonjun
!pip install -q openai pandas numpy tqdm pillow

# 2. API 키 설정 (수동으로 수정 필요)
import os
os.environ["OPENAI_API_KEY"] = "sk-your-api-key-here"  # <- 여기에 API 키 입력

# 3. Google Drive 마운트
from google.colab import drive
drive.mount('/content/drive')

# 4. 실험 실행 (10개 샘플, SAM 없이)
!cd /content/DermAgent-sam-real2/derm1m_exp/experiments && \
    python run_symptom_integration_experiment.py \
    --input_csv /content/DermAgent-sam-real2/dataset/random100.csv \
    --image_dir /content/drive/MyDrive/DermAgent_Data/images \
    --output_dir /content/outputs \
    --num_samples 10 \
    --sam_strategies none \
    --verbose

# 5. 결과 확인
import pandas as pd
from glob import glob
summary = sorted(glob("/content/outputs/experiment_summary_*.csv"))[-1]
print(pd.read_csv(summary).to_string())
```

---

**문서 버전**: 1.0
**작성일**: 2025-12-05
