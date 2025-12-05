#!/usr/bin/env python3
"""
8가지 피부과 진단 방법 비교 실험 스크립트

비교 대상:
1. Baseline + Labels: 질병 라벨 목록 제공
2. Baseline + No Labels: 라벨 없이 자유 진단
3. Baseline + Hierarchical: 온톨로지 트리 구조 제공
4. DermatologyAgent: 온톨로지 기반 계층적 탐색 에이전트
5. ReActAgent: ReAct 패턴 기반 추론 에이전트
6. SAM Baseline: SAM 세그멘테이션 + VLM 진단
7. SAM + LLM-Guided: LLM 가이드 세그멘테이션 + VLM 진단
8. DermatologyAgentSAM: SAM 세그멘테이션 + 온톨로지 기반 에이전트

사용법:
    # .env 파일 사용 (API 키 자동 로드)
    python run_comparison_experiment.py \
        --input_csv /home/heodnjswns/DermAgent/dataset/Derm1M/Derm1M_v2_pretrain_ontology_sampled_100.csv \
        --output_dir ./outputs \
        --test_mode --num_samples 5

    # API 키 직접 지정
    python run_comparison_experiment.py \
        --input_csv /path/to/data.csv \
        --output_dir ./outputs \
        --api_key $OPENAI_API_KEY

    # 특정 방법만 실행
    python run_comparison_experiment.py \
        --methods baseline_labels,react_agent \
        ...
"""

import os
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List

# 경로 설정
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPT_DIR.parent / "eval"))
sys.path.insert(0, str(SCRIPT_DIR.parent / "baseline"))
sys.path.insert(0, str(SCRIPT_DIR.parent / "DermAgent" / "agent"))

# .env 파일 자동 로드
def load_env_file():
    """
    .env 파일에서 환경 변수 로드
    우선순위: 현재 디렉터리 > baseline 디렉터리 > 프로젝트 루트
    """
    env_paths = [
        SCRIPT_DIR / ".env",                    # experiments/.env
        SCRIPT_DIR.parent / "baseline" / ".env", # baseline/.env
        SCRIPT_DIR.parent / ".env",              # derm1m_exp/.env
        PROJECT_ROOT / ".env",                   # 프로젝트 루트/.env
    ]

    for env_path in env_paths:
        if env_path.exists():
            print(f"[INFO] .env 파일 로드: {env_path}")
            with open(env_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        key = key.strip()
                        value = value.strip()
                        # 이미 설정된 환경 변수는 덮어쓰지 않음
                        if key not in os.environ:
                            os.environ[key] = value
            return True
    return False

# .env 파일 로드 실행
load_env_file()

# 로컬 모듈
from experiment_utils import (
    setup_logging,
    create_output_directory,
    save_predictions_csv,
    save_metrics_summary_csv,
    save_per_sample_comparison_csv,
    save_detailed_analysis_csv,
    save_experiment_config,
    save_all_agent_traces,
    load_dataset,
    print_metrics_summary,
    analyze_method_differences,
    MethodResult,
    MethodEvaluation,
    ExperimentConfig,
    AgentTrace,
    AgentStep
)
from hierarchical_baseline import HierarchicalBaselineModel

# 평가 모듈
from ontology_utils import OntologyTree
from evaluation_metrics import HierarchicalEvaluator

# VLM 모델 (독립 래퍼 사용)
try:
    from vlm_wrapper import GPT4oWrapper as GPT4o
except ImportError:
    # 폴백: 기존 model.py 시도
    try:
        from model import GPT4o
    except ImportError:
        print("[경고] VLM 모델 임포트 실패")

# 에이전트
try:
    from dermatology_agent import DermatologyAgent
except ImportError:
    print("[경고] DermatologyAgent 임포트 실패")

try:
    from react_agent import ReActDermatologyAgent
except ImportError:
    print("[경고] ReActDermatologyAgent 임포트 실패")

# SAM 통합 에이전트
SAM_AGENT_AVAILABLE = False
try:
    from dermatology_agent_sam import DermatologyAgentSAM
    SAM_AGENT_AVAILABLE = True
except ImportError:
    print("[경고] DermatologyAgentSAM 임포트 실패")

# SAM Wrapper
SAM_AVAILABLE = False
try:
    from sam_wrapper import SAMBaselineWrapper
    SAM_AVAILABLE = SAMBaselineWrapper.is_available()
except ImportError:
    print("[경고] SAM Wrapper 임포트 실패")


# ============ 방법별 진단 함수 ============

def run_baseline_with_labels(
    vlm,
    df,
    image_base_dir: str,
    logger,
    verbose: bool = True
) -> List[MethodResult]:
    """
    방법 1: Baseline + Labels
    질병 라벨 목록을 프롬프트에 제공하는 기본 베이스라인
    """
    logger.info("=" * 60)
    logger.info("방법 1: Baseline + Labels 실행 시작")
    logger.info("=" * 60)

    results = []

    # 프롬프트 (use_labels_prompt=True 방식과 동일)
    script_dir = SCRIPT_DIR.parent / "baseline"
    disease_labels_path = script_dir / "extracted_node_names.txt"

    if disease_labels_path.exists():
        with open(disease_labels_path, "r") as f:
            disease_labels = [
                line.strip().split("→")[1] if "→" in line else line.strip()
                for line in f.readlines() if line.strip()
            ]
        disease_labels_str = ", ".join(disease_labels)
    else:
        # 온톨로지에서 직접 추출
        tree = OntologyTree()
        disease_labels_str = ", ".join(sorted(tree.valid_nodes))

    system_prompt = f"""You are a dermatology expert. Analyze this skin image.
When identifying skin conditions, the disease_label should be one of the following: {disease_labels_str}

IMPORTANT: If no clear skin lesion is visible, the image does not show identifiable human skin, or you cannot make a confident diagnosis, set disease_label to "no definitive diagnosis".

Provide your response in JSON format:
{{
    "disease_label": "diagnosis from the list above (or 'no definitive diagnosis' if uncertain)",
    "body_location": "anatomical location",
    "caption": "detailed description",
    "confidence": 0.0-1.0
}}"""

    try:
        from tqdm import tqdm
        iterator = tqdm(df.iterrows(), total=len(df), desc="Baseline+Labels")
    except ImportError:
        iterator = df.iterrows()

    for idx, row in iterator:
        filename = row['filename']
        gt_label = str(row.get('disease_label', ''))
        hierarchical_gt = str(row.get('hierarchical_disease_label', ''))

        image_path = os.path.join(image_base_dir, filename)

        try:
            response = vlm.chat_img(system_prompt, [image_path], max_tokens=512)
            parsed = _parse_json_response(response)
            pred_label = parsed.get('disease_label', '')
            confidence = float(parsed.get('confidence', 0.5))
            reasoning = parsed.get('caption', '')

            # 빈 값이면 "no definitive diagnosis"로 설정
            if not pred_label or pred_label.strip() == "":
                pred_label = "no definitive diagnosis"
        except Exception as e:
            logger.error(f"  [오류] {filename}: {str(e)}")
            pred_label = "no definitive diagnosis"
            confidence = 0.0
            reasoning = str(e)
            response = ""

        result = MethodResult(
            sample_id=idx,
            filename=filename,
            ground_truth=gt_label,
            hierarchical_gt=hierarchical_gt,
            prediction=pred_label,
            confidence=confidence,
            reasoning=reasoning,
            raw_response=response
        )
        results.append(result)

        if verbose:
            logger.info(f"  [{idx}] GT: {gt_label} | Pred: {pred_label}")

    logger.info(f"방법 1 완료: {len(results)}개 샘플 처리")
    return results


def run_baseline_without_labels(
    vlm,
    df,
    image_base_dir: str,
    logger,
    verbose: bool = True
) -> List[MethodResult]:
    """
    방법 2: Baseline + No Labels
    라벨 목록 없이 자유롭게 진단
    """
    logger.info("=" * 60)
    logger.info("방법 2: Baseline + No Labels 실행 시작")
    logger.info("=" * 60)

    results = []

    system_prompt = """You are a dermatology expert. Analyze this skin image.
Provide a diagnosis based on the visible skin findings.
Do NOT assume any predefined disease label list.

IMPORTANT: If no clear skin lesion is visible, the image does not show identifiable human skin, or you cannot make a confident diagnosis, set disease_label to "no definitive diagnosis".

Provide your response in JSON format:
{
    "disease_label": "your diagnosis (or 'no definitive diagnosis' if uncertain)",
    "body_location": "anatomical location",
    "caption": "detailed description of findings",
    "confidence": 0.0-1.0,
    "differential_diagnoses": ["other possible diagnoses"]
}"""

    try:
        from tqdm import tqdm
        iterator = tqdm(df.iterrows(), total=len(df), desc="Baseline-Labels")
    except ImportError:
        iterator = df.iterrows()

    for idx, row in iterator:
        filename = row['filename']
        gt_label = str(row.get('disease_label', ''))
        hierarchical_gt = str(row.get('hierarchical_disease_label', ''))

        image_path = os.path.join(image_base_dir, filename)

        try:
            response = vlm.chat_img(system_prompt, [image_path], max_tokens=512)
            parsed = _parse_json_response(response)
            pred_label = parsed.get('disease_label', '')
            confidence = float(parsed.get('confidence', 0.5))
            reasoning = parsed.get('caption', '')

            # 빈 값이면 "no definitive diagnosis"로 설정
            if not pred_label or pred_label.strip() == "":
                pred_label = "no definitive diagnosis"
        except Exception as e:
            logger.error(f"  [오류] {filename}: {str(e)}")
            pred_label = "no definitive diagnosis"
            confidence = 0.0
            reasoning = str(e)
            response = ""

        result = MethodResult(
            sample_id=idx,
            filename=filename,
            ground_truth=gt_label,
            hierarchical_gt=hierarchical_gt,
            prediction=pred_label,
            confidence=confidence,
            reasoning=reasoning,
            raw_response=response
        )
        results.append(result)

        if verbose:
            logger.info(f"  [{idx}] GT: {gt_label} | Pred: {pred_label}")

    logger.info(f"방법 2 완료: {len(results)}개 샘플 처리")
    return results


def run_baseline_hierarchical(
    vlm,
    df,
    image_base_dir: str,
    logger,
    verbose: bool = True
) -> List[MethodResult]:
    """
    방법 3: Baseline + Hierarchical
    전체 온톨로지 트리 구조를 프롬프트에 제공
    """
    logger.info("=" * 60)
    logger.info("방법 3: Baseline + Hierarchical 실행 시작")
    logger.info("=" * 60)

    results = []

    # HierarchicalBaselineModel 사용
    hier_model = HierarchicalBaselineModel(vlm, verbose=False)

    try:
        from tqdm import tqdm
        iterator = tqdm(df.iterrows(), total=len(df), desc="Baseline+Hier")
    except ImportError:
        iterator = df.iterrows()

    for idx, row in iterator:
        filename = row['filename']
        gt_label = str(row.get('disease_label', ''))
        hierarchical_gt = str(row.get('hierarchical_disease_label', ''))

        image_path = os.path.join(image_base_dir, filename)

        try:
            response_dict = hier_model.analyze_image(image_path)
            pred_label = response_dict.get('disease_label', '')
            confidence = float(response_dict.get('confidence', 0.5))
            reasoning = response_dict.get('reasoning', '')
            raw_response = response_dict.get('raw_response', '')

            # 빈 값이면 "no definitive diagnosis"로 설정
            if not pred_label or pred_label.strip() == "":
                pred_label = "no definitive diagnosis"
        except Exception as e:
            logger.error(f"  [오류] {filename}: {str(e)}")
            pred_label = "no definitive diagnosis"
            confidence = 0.0
            reasoning = str(e)
            raw_response = ""

        result = MethodResult(
            sample_id=idx,
            filename=filename,
            ground_truth=gt_label,
            hierarchical_gt=hierarchical_gt,
            prediction=pred_label,
            confidence=confidence,
            reasoning=reasoning,
            raw_response=raw_response
        )
        results.append(result)

        if verbose:
            logger.info(f"  [{idx}] GT: {gt_label} | Pred: {pred_label}")

    logger.info(f"방법 3 완료: {len(results)}개 샘플 처리")
    return results


def run_dermatology_agent(
    vlm,
    df,
    image_base_dir: str,
    logger,
    verbose: bool = True
) -> List[MethodResult]:
    """
    방법 4: DermatologyAgent
    온톨로지 기반 계층적 탐색 에이전트
    """
    logger.info("=" * 60)
    logger.info("방법 4: DermatologyAgent 실행 시작")
    logger.info("=" * 60)

    results = []

    # 에이전트 초기화
    try:
        agent = DermatologyAgent(vlm_model=vlm, verbose=False)
    except Exception as e:
        logger.error(f"DermatologyAgent 초기화 실패: {e}")
        # 빈 결과 반환
        for idx, row in df.iterrows():
            results.append(MethodResult(
                sample_id=idx,
                filename=row['filename'],
                ground_truth=str(row.get('disease_label', '')),
                hierarchical_gt=str(row.get('hierarchical_disease_label', '')),
                prediction="Error",
                confidence=0.0,
                reasoning=f"Agent initialization failed: {e}"
            ))
        return results

    try:
        from tqdm import tqdm
        iterator = tqdm(df.iterrows(), total=len(df), desc="DermAgent")
    except ImportError:
        iterator = df.iterrows()

    for idx, row in iterator:
        filename = row['filename']
        gt_label = str(row.get('disease_label', ''))
        hierarchical_gt = str(row.get('hierarchical_disease_label', ''))

        image_path = os.path.join(image_base_dir, filename)

        agent_trace = None  # 에이전트 트레이스

        try:
            diagnosis_result = agent.diagnose(image_path)

            all_predictions = []  # Top-K용 전체 예측 리스트

            if hasattr(diagnosis_result, 'final_diagnosis'):
                pred_label = diagnosis_result.final_diagnosis
                confidence = getattr(diagnosis_result, 'confidence', 0.5)
                reasoning = getattr(diagnosis_result, 'reasoning', '')

                # final_diagnosis가 리스트면 모든 예측 저장
                if isinstance(pred_label, list):
                    all_predictions = [str(p) for p in pred_label if p]
                    pred_label = pred_label[0] if pred_label else ""
                else:
                    all_predictions = [str(pred_label)] if pred_label else []
            elif isinstance(diagnosis_result, dict):
                pred_label = diagnosis_result.get('final_diagnosis', '')
                confidence = diagnosis_result.get('confidence', 0.5)
                reasoning = diagnosis_result.get('reasoning', '')

                if isinstance(pred_label, list):
                    all_predictions = [str(p) for p in pred_label if p]
                    pred_label = pred_label[0] if pred_label else ""
                else:
                    all_predictions = [str(pred_label)] if pred_label else []
            else:
                pred_label = str(diagnosis_result)
                confidence = 0.5
                reasoning = ""
                all_predictions = [pred_label] if pred_label else []

            # 빈 값이면 "no definitive diagnosis"로 설정
            if not pred_label or (isinstance(pred_label, str) and pred_label.strip() == ""):
                pred_label = "no definitive diagnosis"
                if not all_predictions:
                    all_predictions = ["no definitive diagnosis"]

            # AgentTrace 생성 (DermatologyAgent)
            agent_trace = _extract_dermatology_agent_trace(
                idx, filename, image_path, diagnosis_result, pred_label, confidence
            )

        except Exception as e:
            logger.error(f"  [오류] {filename}: {str(e)}")
            pred_label = "no definitive diagnosis"
            confidence = 0.0
            reasoning = str(e)
            all_predictions = ["no definitive diagnosis"]
            # 오류 시에도 트레이스 생성
            agent_trace = AgentTrace(
                sample_id=idx,
                filename=filename,
                agent_type="dermatology_agent",
                image_path=image_path,
                primary_diagnosis=pred_label,
                confidence=confidence,
                errors=[str(e)]
            )

        result = MethodResult(
            sample_id=idx,
            filename=filename,
            ground_truth=gt_label,
            hierarchical_gt=hierarchical_gt,
            prediction=pred_label,
            confidence=float(confidence),
            reasoning=str(reasoning),
            all_predictions=all_predictions,
            agent_trace=agent_trace
        )
        results.append(result)

        if verbose:
            logger.info(f"  [{idx}] GT: {gt_label} | Pred: {pred_label}")

    logger.info(f"방법 4 완료: {len(results)}개 샘플 처리")
    return results


def _extract_dermatology_agent_trace(
    sample_id: int,
    filename: str,
    image_path: str,
    diagnosis_result,
    pred_label: str,
    confidence: float
) -> AgentTrace:
    """DermatologyAgent 결과에서 AgentTrace 추출"""
    trace = AgentTrace(
        sample_id=sample_id,
        filename=filename,
        agent_type="dermatology_agent",
        image_path=image_path,
        primary_diagnosis=pred_label,
        confidence=confidence
    )

    # dict인 경우 상세 정보 추출
    if isinstance(diagnosis_result, dict):
        # 관찰 결과
        if 'observations' in diagnosis_result:
            trace.observations = diagnosis_result['observations']

        # 온톨로지 경로
        if 'diagnosis_path' in diagnosis_result:
            trace.ontology_path = diagnosis_result['diagnosis_path']

        # 후보군
        if 'candidates_considered' in diagnosis_result:
            trace.candidates_considered = diagnosis_result['candidates_considered']

        # 후보 점수
        if 'confidence_scores' in diagnosis_result:
            trace.candidate_scores = diagnosis_result['confidence_scores']

        # 감별 진단
        if 'final_diagnosis' in diagnosis_result:
            final_diag = diagnosis_result['final_diagnosis']
            if isinstance(final_diag, list) and len(final_diag) > 1:
                trace.differential_diagnoses = final_diag[1:]

        # 추론 과정
        if 'reasoning_history' in diagnosis_result:
            for i, step_data in enumerate(diagnosis_result['reasoning_history']):
                step = AgentStep(
                    step_num=i + 1,
                    thought=str(step_data.get('step', '')),
                    action=step_data.get('step', ''),
                    action_input={},
                    observation=str(step_data.get('reasoning', ''))[:500]
                )
                trace.steps.append(step)
            trace.total_steps = len(trace.steps)

        # VLM 호출 수
        if 'vlm_calls' in diagnosis_result:
            trace.total_vlm_calls = diagnosis_result['vlm_calls']

        # 경고/오류
        if 'warnings' in diagnosis_result:
            trace.warnings = diagnosis_result['warnings']
        if 'errors' in diagnosis_result:
            trace.errors = diagnosis_result['errors']

        # 최종 추론
        if 'reasoning_history' in diagnosis_result and diagnosis_result['reasoning_history']:
            last_step = diagnosis_result['reasoning_history'][-1]
            trace.final_reasoning = str(last_step.get('reasoning', ''))

    return trace


def run_react_agent(
    vlm,
    df,
    image_base_dir: str,
    logger,
    verbose: bool = True,
    max_steps: int = 8
) -> List[MethodResult]:
    """
    방법 5: ReActAgent
    ReAct 패턴 기반 추론 에이전트
    """
    logger.info("=" * 60)
    logger.info("방법 5: ReActAgent 실행 시작")
    logger.info("=" * 60)

    results = []

    # 에이전트 초기화
    try:
        agent = ReActDermatologyAgent(vlm_model=vlm, max_steps=max_steps, verbose=False)
    except Exception as e:
        logger.error(f"ReActAgent 초기화 실패: {e}")
        # 빈 결과 반환
        for idx, row in df.iterrows():
            results.append(MethodResult(
                sample_id=idx,
                filename=row['filename'],
                ground_truth=str(row.get('disease_label', '')),
                hierarchical_gt=str(row.get('hierarchical_disease_label', '')),
                prediction="Error",
                confidence=0.0,
                reasoning=f"Agent initialization failed: {e}"
            ))
        return results

    try:
        from tqdm import tqdm
        iterator = tqdm(df.iterrows(), total=len(df), desc="ReActAgent")
    except ImportError:
        iterator = df.iterrows()

    for idx, row in iterator:
        filename = row['filename']
        gt_label = str(row.get('disease_label', ''))
        hierarchical_gt = str(row.get('hierarchical_disease_label', ''))

        image_path = os.path.join(image_base_dir, filename)

        agent_trace = None  # 에이전트 트레이스
        differentials = []

        try:
            diagnosis_result = agent.diagnose(image_path)

            all_predictions = []  # Top-K용 전체 예측 리스트

            if hasattr(diagnosis_result, 'primary_diagnosis'):
                pred_label = diagnosis_result.primary_diagnosis
                confidence = getattr(diagnosis_result, 'confidence', 0.5)

                # differential_diagnoses가 있으면 전체 예측에 추가
                differentials = getattr(diagnosis_result, 'differential_diagnoses', [])
                if isinstance(pred_label, list):
                    all_predictions = [str(p) for p in pred_label if p]
                    pred_label = pred_label[0] if pred_label else ""
                else:
                    all_predictions = [str(pred_label)] if pred_label else []

                # differentials 추가
                if differentials:
                    for diff in differentials:
                        if isinstance(diff, dict):
                            diff_name = diff.get('disease', diff.get('name', ''))
                        else:
                            diff_name = str(diff)
                        if diff_name and diff_name not in all_predictions:
                            all_predictions.append(diff_name)

                # 추론 체인을 reasoning으로
                if hasattr(diagnosis_result, 'reasoning_chain'):
                    reasoning_parts = []
                    for step in diagnosis_result.reasoning_chain:
                        if hasattr(step, 'thought'):
                            reasoning_parts.append(f"Step {step.step_num}: {step.thought}")
                    reasoning = " | ".join(reasoning_parts)
                else:
                    reasoning = ""
            elif isinstance(diagnosis_result, dict):
                pred_label = diagnosis_result.get('primary_diagnosis', '')
                confidence = diagnosis_result.get('confidence', 0.5)
                reasoning = str(diagnosis_result.get('reasoning_chain', ''))
                differentials = diagnosis_result.get('differential_diagnoses', [])

                if isinstance(pred_label, list):
                    all_predictions = [str(p) for p in pred_label if p]
                    pred_label = pred_label[0] if pred_label else ""
                else:
                    all_predictions = [str(pred_label)] if pred_label else []

                for diff in differentials:
                    if isinstance(diff, dict):
                        diff_name = diff.get('disease', diff.get('name', ''))
                    else:
                        diff_name = str(diff)
                    if diff_name and diff_name not in all_predictions:
                        all_predictions.append(diff_name)
            else:
                pred_label = str(diagnosis_result)
                confidence = 0.5
                reasoning = ""
                all_predictions = [pred_label] if pred_label else []

            # 빈 값이면 "no definitive diagnosis"로 설정
            if not pred_label or (isinstance(pred_label, str) and pred_label.strip() == ""):
                pred_label = "no definitive diagnosis"
                if not all_predictions:
                    all_predictions = ["no definitive diagnosis"]

            # AgentTrace 생성 (ReActAgent)
            agent_trace = _extract_react_agent_trace(
                idx, filename, image_path, diagnosis_result, pred_label, confidence, differentials
            )

        except Exception as e:
            logger.error(f"  [오류] {filename}: {str(e)}")
            pred_label = "no definitive diagnosis"
            confidence = 0.0
            reasoning = str(e)
            all_predictions = ["no definitive diagnosis"]
            # 오류 시에도 트레이스 생성
            agent_trace = AgentTrace(
                sample_id=idx,
                filename=filename,
                agent_type="react_agent",
                image_path=image_path,
                primary_diagnosis=pred_label,
                confidence=confidence,
                errors=[str(e)]
            )

        result = MethodResult(
            sample_id=idx,
            filename=filename,
            ground_truth=gt_label,
            hierarchical_gt=hierarchical_gt,
            prediction=pred_label,
            confidence=float(confidence),
            reasoning=str(reasoning)[:500],  # 너무 긴 reasoning 자르기
            all_predictions=all_predictions,
            agent_trace=agent_trace
        )
        results.append(result)

        if verbose:
            logger.info(f"  [{idx}] GT: {gt_label} | Pred: {pred_label}")

    logger.info(f"방법 5 완료: {len(results)}개 샘플 처리")
    return results


def _extract_react_agent_trace(
    sample_id: int,
    filename: str,
    image_path: str,
    diagnosis_result,
    pred_label: str,
    confidence: float,
    differentials: List
) -> AgentTrace:
    """ReActAgent 결과에서 AgentTrace 추출"""
    trace = AgentTrace(
        sample_id=sample_id,
        filename=filename,
        agent_type="react_agent",
        image_path=image_path,
        primary_diagnosis=pred_label,
        confidence=confidence
    )

    # differentials 설정
    diff_list = []
    for diff in differentials:
        if isinstance(diff, dict):
            diff_name = diff.get('disease', diff.get('name', ''))
        else:
            diff_name = str(diff)
        if diff_name:
            diff_list.append(diff_name)
    trace.differential_diagnoses = diff_list

    # DiagnosisResult 객체인 경우 상세 정보 추출
    if hasattr(diagnosis_result, 'reasoning_chain'):
        reasoning_chain = diagnosis_result.reasoning_chain
        for react_step in reasoning_chain:
            step = AgentStep(
                step_num=getattr(react_step, 'step_num', 0),
                thought=getattr(react_step, 'thought', '') or '',
                action=getattr(react_step, 'action', '').value if hasattr(getattr(react_step, 'action', None), 'value') else str(getattr(react_step, 'action', '')),
                action_input=getattr(react_step, 'action_input', {}) or {},
                observation=str(getattr(react_step, 'observation', ''))[:1000] if getattr(react_step, 'observation', None) else ''
            )
            trace.steps.append(step)
        trace.total_steps = len(trace.steps)

    # 관찰 결과
    if hasattr(diagnosis_result, 'observations') and diagnosis_result.observations:
        obs = diagnosis_result.observations
        trace.observations = {
            'morphology': getattr(obs, 'morphology', []),
            'color': getattr(obs, 'color', []),
            'distribution': getattr(obs, 'distribution', []),
            'surface': getattr(obs, 'surface', []),
            'border': getattr(obs, 'border', []),
            'location': getattr(obs, 'location', ''),
            'additional_notes': getattr(obs, 'additional_notes', '')
        }

    # 온톨로지 경로
    if hasattr(diagnosis_result, 'ontology_path'):
        trace.ontology_path = diagnosis_result.ontology_path or []

    # 경고
    if hasattr(diagnosis_result, 'warnings'):
        trace.warnings = diagnosis_result.warnings or []

    # 최종 추론 (마지막 thought)
    if trace.steps:
        last_step = trace.steps[-1]
        trace.final_reasoning = last_step.thought

    return trace


def run_sam_baseline(
    vlm,
    df,
    image_base_dir: str,
    logger,
    verbose: bool = True,
    segmenter_type: str = "sam",
    segmentation_strategy: str = "center"
) -> List[MethodResult]:
    """
    방법 6: SAM Baseline
    SAM 세그멘테이션 + VLM 진단

    전략:
    - center: 이미지 중앙 기반 세그멘테이션 (빠름, 기본값)
    - lesion_features: 병변 특징 기반 세그멘테이션 (정확도↑)
    - both: 두 전략 중 더 좋은 결과 선택
    - llm_guided: LLM이 병변 위치를 가이드하고 SAM이 분할 (정확도↑↑, API 비용↑)
    """
    strategy_desc = {
        "center": "이미지 중앙 기반 세그멘테이션",
        "lesion_features": "병변 특징 기반 세그멘테이션",
        "both": "최적 전략 자동 선택",
        "llm_guided": "LLM 가이드 세그멘테이션"
    }
    method_num = "7" if segmentation_strategy == "llm_guided" else "6"
    desc = strategy_desc.get(segmentation_strategy, segmentation_strategy)

    logger.info("=" * 60)
    logger.info(f"방법 {method_num}: SAM + {segmentation_strategy.upper()}")
    logger.info(f"  세그멘터: {segmenter_type.upper()}")
    logger.info(f"  전략: {desc}")
    logger.info("=" * 60)

    results = []

    if not SAM_AVAILABLE:
        logger.error("SAM 모듈을 사용할 수 없습니다.")
        for idx, row in df.iterrows():
            results.append(MethodResult(
                sample_id=idx,
                filename=row['filename'],
                ground_truth=str(row.get('disease_label', '')),
                hierarchical_gt=str(row.get('hierarchical_disease_label', '')),
                prediction="SAM not available",
                confidence=0.0,
                reasoning="SAM module not available"
            ))
        return results

    # SAM Wrapper 초기화
    try:
        sam_wrapper = SAMBaselineWrapper(
            vlm_model=vlm,
            segmenter_type=segmenter_type,
            segmentation_strategy=segmentation_strategy,
            verbose=False
        )
    except Exception as e:
        logger.error(f"SAM Wrapper 초기화 실패: {e}")
        for idx, row in df.iterrows():
            results.append(MethodResult(
                sample_id=idx,
                filename=row['filename'],
                ground_truth=str(row.get('disease_label', '')),
                hierarchical_gt=str(row.get('hierarchical_disease_label', '')),
                prediction="Error",
                confidence=0.0,
                reasoning=f"SAM initialization failed: {e}"
            ))
        return results

    try:
        from tqdm import tqdm
        iterator = tqdm(df.iterrows(), total=len(df), desc=f"SAM-{segmenter_type}")
    except ImportError:
        iterator = df.iterrows()

    for idx, row in iterator:
        filename = row['filename']
        gt_label = str(row.get('disease_label', ''))
        hierarchical_gt = str(row.get('hierarchical_disease_label', ''))

        image_path = os.path.join(image_base_dir, filename)

        try:
            analysis = sam_wrapper.analyze_image(image_path)
            pred_label = analysis.get('disease_label', '')
            confidence = float(analysis.get('confidence', 0.5))
            reasoning = analysis.get('caption', '')
            raw_response = analysis.get('raw_response', '')

            # 빈 값이면 "no definitive diagnosis"로 설정
            if not pred_label or pred_label.strip() == "":
                pred_label = "no definitive diagnosis"

        except Exception as e:
            logger.error(f"  [오류] {filename}: {str(e)}")
            pred_label = "no definitive diagnosis"
            confidence = 0.0
            reasoning = str(e)
            raw_response = ""

        result = MethodResult(
            sample_id=idx,
            filename=filename,
            ground_truth=gt_label,
            hierarchical_gt=hierarchical_gt,
            prediction=pred_label,
            confidence=confidence,
            reasoning=reasoning,
            raw_response=raw_response
        )
        results.append(result)

        if verbose:
            logger.info(f"  [{idx}] GT: {gt_label} | Pred: {pred_label}")

    logger.info(f"방법 6 완료: {len(results)}개 샘플 처리")
    return results


def run_sam_llm_guided(
    vlm,
    df,
    image_base_dir: str,
    logger,
    verbose: bool = True
) -> List[MethodResult]:
    """
    방법 7: SAM + LLM-Guided Pipeline
    LLM이 병변 위치를 가이드하고 SAM이 분할
    """
    return run_sam_baseline(
        vlm=vlm,
        df=df,
        image_base_dir=image_base_dir,
        logger=logger,
        verbose=verbose,
        segmenter_type="sam",
        segmentation_strategy="llm_guided"
    )


def run_dermatology_agent_sam(
    vlm,
    df,
    image_base_dir: str,
    logger,
    verbose: bool = True,
    segmenter_type: str = "sam",
    segmentation_strategy: str = "center"
) -> List[MethodResult]:
    """
    방법 8: DermatologyAgentSAM
    SAM 세그멘테이션 + 온톨로지 기반 계층적 에이전트

    SAM으로 병변 영역을 먼저 분리한 후, 세그멘테이션된 이미지를 사용하여
    온톨로지 기반 계층적 진단을 수행
    """
    logger.info("=" * 60)
    logger.info("방법 8: DermatologyAgentSAM 실행 시작")
    logger.info(f"  세그멘터: {segmenter_type.upper()}")
    logger.info(f"  전략: {segmentation_strategy}")
    logger.info("=" * 60)

    results = []

    if not SAM_AGENT_AVAILABLE:
        logger.error("DermatologyAgentSAM 모듈을 사용할 수 없습니다.")
        for idx, row in df.iterrows():
            results.append(MethodResult(
                sample_id=idx,
                filename=row['filename'],
                ground_truth=str(row.get('disease_label', '')),
                hierarchical_gt=str(row.get('hierarchical_disease_label', '')),
                prediction="DermatologyAgentSAM not available",
                confidence=0.0,
                reasoning="DermatologyAgentSAM module not available"
            ))
        return results

    # 에이전트 초기화
    try:
        agent = DermatologyAgentSAM(
            vlm_model=vlm,
            segmenter_type=segmenter_type,
            segmentation_strategy=segmentation_strategy,
            use_segmentation=True,
            verbose=False
        )
    except Exception as e:
        logger.error(f"DermatologyAgentSAM 초기화 실패: {e}")
        for idx, row in df.iterrows():
            results.append(MethodResult(
                sample_id=idx,
                filename=row['filename'],
                ground_truth=str(row.get('disease_label', '')),
                hierarchical_gt=str(row.get('hierarchical_disease_label', '')),
                prediction="Error",
                confidence=0.0,
                reasoning=f"Agent initialization failed: {e}"
            ))
        return results

    try:
        from tqdm import tqdm
        iterator = tqdm(df.iterrows(), total=len(df), desc="DermAgentSAM")
    except ImportError:
        iterator = df.iterrows()

    for idx, row in iterator:
        filename = row['filename']
        gt_label = str(row.get('disease_label', ''))
        hierarchical_gt = str(row.get('hierarchical_disease_label', ''))

        image_path = os.path.join(image_base_dir, filename)

        agent_trace = None

        try:
            diagnosis_result = agent.diagnose(image_path)

            all_predictions = []

            if isinstance(diagnosis_result, dict):
                pred_label = diagnosis_result.get('final_diagnosis', '')
                confidence = diagnosis_result.get('confidence_scores', {})

                # final_diagnosis가 리스트면 모든 예측 저장
                if isinstance(pred_label, list):
                    all_predictions = [str(p) for p in pred_label if p]
                    pred_label = pred_label[0] if pred_label else ""
                else:
                    all_predictions = [str(pred_label)] if pred_label else []

                # confidence 계산 (dict인 경우 첫 번째 진단의 점수)
                if isinstance(confidence, dict) and all_predictions:
                    confidence = confidence.get(all_predictions[0], 0.5)
                elif not isinstance(confidence, (int, float)):
                    confidence = 0.5

                reasoning = diagnosis_result.get('reasoning_history', [])
                if reasoning:
                    reasoning = str(reasoning[-1].get('reasoning', ''))[:500]
                else:
                    reasoning = ""
            else:
                pred_label = str(diagnosis_result)
                confidence = 0.5
                reasoning = ""
                all_predictions = [pred_label] if pred_label else []

            # 빈 값이면 "no definitive diagnosis"로 설정
            if not pred_label or (isinstance(pred_label, str) and pred_label.strip() == ""):
                pred_label = "no definitive diagnosis"
                if not all_predictions:
                    all_predictions = ["no definitive diagnosis"]

            # AgentTrace 생성
            agent_trace = _extract_sam_agent_trace(
                idx, filename, image_path, diagnosis_result, pred_label, confidence
            )

        except Exception as e:
            logger.error(f"  [오류] {filename}: {str(e)}")
            pred_label = "no definitive diagnosis"
            confidence = 0.0
            reasoning = str(e)
            all_predictions = ["no definitive diagnosis"]
            agent_trace = AgentTrace(
                sample_id=idx,
                filename=filename,
                agent_type="dermatology_agent_sam",
                image_path=image_path,
                primary_diagnosis=pred_label,
                confidence=confidence,
                errors=[str(e)]
            )

        result = MethodResult(
            sample_id=idx,
            filename=filename,
            ground_truth=gt_label,
            hierarchical_gt=hierarchical_gt,
            prediction=pred_label,
            confidence=float(confidence),
            reasoning=str(reasoning),
            all_predictions=all_predictions,
            agent_trace=agent_trace
        )
        results.append(result)

        if verbose:
            logger.info(f"  [{idx}] GT: {gt_label} | Pred: {pred_label}")

    logger.info(f"방법 8 완료: {len(results)}개 샘플 처리")
    return results


def _extract_sam_agent_trace(
    sample_id: int,
    filename: str,
    image_path: str,
    diagnosis_result,
    pred_label: str,
    confidence: float
) -> AgentTrace:
    """DermatologyAgentSAM 결과에서 AgentTrace 추출"""
    trace = AgentTrace(
        sample_id=sample_id,
        filename=filename,
        agent_type="dermatology_agent_sam",
        image_path=image_path,
        primary_diagnosis=pred_label,
        confidence=confidence
    )

    if isinstance(diagnosis_result, dict):
        # 관찰 결과
        if 'observations' in diagnosis_result:
            trace.observations = diagnosis_result['observations']

        # 온톨로지 경로
        if 'diagnosis_path' in diagnosis_result:
            trace.ontology_path = diagnosis_result['diagnosis_path']

        # 후보군
        if 'candidates_considered' in diagnosis_result:
            trace.candidates_considered = diagnosis_result['candidates_considered']

        # 후보 점수
        if 'confidence_scores' in diagnosis_result:
            trace.candidate_scores = diagnosis_result['confidence_scores']

        # 감별 진단
        if 'final_diagnosis' in diagnosis_result:
            final_diag = diagnosis_result['final_diagnosis']
            if isinstance(final_diag, list) and len(final_diag) > 1:
                trace.differential_diagnoses = final_diag[1:]

        # 세그멘테이션 정보 추가
        if 'segmentation' in diagnosis_result:
            seg_info = diagnosis_result['segmentation']
            # observations에 segmentation 정보 추가
            if trace.observations is None:
                trace.observations = {}
            trace.observations['segmentation'] = seg_info

        # 추론 과정
        if 'reasoning_history' in diagnosis_result:
            for i, step_data in enumerate(diagnosis_result['reasoning_history']):
                step = AgentStep(
                    step_num=i + 1,
                    thought=str(step_data.get('step', '')),
                    action=step_data.get('step', ''),
                    action_input={},
                    observation=str(step_data.get('reasoning', ''))[:500]
                )
                trace.steps.append(step)
            trace.total_steps = len(trace.steps)

        # 경고/오류
        if 'warnings' in diagnosis_result:
            trace.warnings = diagnosis_result['warnings']
        if 'errors' in diagnosis_result:
            trace.errors = diagnosis_result['errors']

        # 최종 추론
        if 'reasoning_history' in diagnosis_result and diagnosis_result['reasoning_history']:
            last_step = diagnosis_result['reasoning_history'][-1]
            trace.final_reasoning = str(last_step.get('reasoning', ''))

    return trace


# ============ 유틸리티 함수 ============

def _parse_json_response(response) -> Dict:
    """JSON 응답 파싱 (None 및 비문자열 타입 처리)"""
    import re
    # None이나 비문자열 타입 처리
    if response is None:
        return {}
    if not isinstance(response, str):
        response = str(response)
    if not response.strip():
        return {}

    try:
        json_match = re.search(r'\{[\s\S]*\}', response)
        if json_match:
            return json.loads(json_match.group())
    except (json.JSONDecodeError, TypeError):
        pass
    return {}


def evaluate_method_results(
    results: List[MethodResult],
    evaluator: HierarchicalEvaluator,
    method_name: str,
    use_top_k: bool = False
) -> MethodEvaluation:
    """
    방법 결과 평가

    Args:
        results: MethodResult 리스트
        evaluator: HierarchicalEvaluator
        method_name: 방법 이름
        use_top_k: Top-K 평가 사용 여부 (DermatologyAgent, ReActAgent 등)

    Returns:
        MethodEvaluation
    """
    ground_truths = [[r.ground_truth] for r in results]

    # Top-K 평가를 사용하는 경우 all_predictions 사용
    if use_top_k:
        # all_predictions가 있으면 사용, 없으면 prediction만 사용
        predictions = [
            r.all_predictions if r.all_predictions else [r.prediction]
            for r in results
        ]
        eval_result = evaluator.evaluate_batch_with_top_k(
            ground_truths, predictions, k_values=[1, 3, 5]
        )
    else:
        predictions = [[r.prediction] for r in results]
        eval_result = evaluator.evaluate_batch(ground_truths, predictions)

    return MethodEvaluation(
        method_name=method_name,
        exact_match=eval_result.exact_match,
        partial_match=eval_result.partial_match,
        hierarchical_f1=eval_result.hierarchical_f1,
        avg_distance=eval_result.avg_hierarchical_distance,
        partial_credit=eval_result.avg_partial_credit,
        level_accuracy=eval_result.level_accuracy,
        total_samples=eval_result.total_samples,
        valid_samples=eval_result.valid_samples,
        top_k_accuracy=eval_result.top_k_accuracy if use_top_k else {},
        top_k_hierarchical_f1=eval_result.top_k_hierarchical_f1 if use_top_k else {}
    )


# ============ 메인 함수 ============

def main():
    parser = argparse.ArgumentParser(description="8가지 피부과 진단 방법 비교 실험")

    # 기본 데이터셋 경로
    default_csv = SCRIPT_DIR.parent.parent / "dataset" / "Derm1M" / "Derm1M_v2_pretrain_ontology_sampled_100.csv"

    # 기본 출력 디렉터리 (experiments/outputs에 고정)
    default_output_dir = SCRIPT_DIR / "outputs"

    # 입력 인자
    parser.add_argument("--input_csv", type=str, default=str(default_csv),
                        help=f"입력 CSV 파일 경로 (기본값: {default_csv})")
    parser.add_argument("--output_dir", type=str, default=str(default_output_dir),
                        help=f"출력 디렉터리 (기본값: {default_output_dir})")
    parser.add_argument("--api_key", type=str, default=None,
                        help="OpenAI API 키 (없으면 .env 또는 환경변수에서 로드)")

    # 선택적 인자
    parser.add_argument("--image_base_dir", type=str, default=None,
                        help="이미지 기본 디렉터리 (기본값: CSV와 동일 위치)")
    parser.add_argument("--model", type=str, default="gpt-4o",
                        help="사용할 모델 (기본값: gpt-4o)")
    parser.add_argument("--test_mode", action="store_true",
                        help="테스트 모드 활성화")
    parser.add_argument("--num_samples", type=int, default=5,
                        help="테스트 모드 샘플 수 (기본값: 5)")
    parser.add_argument("--methods", type=str, default=None,
                        help="실행할 방법들 (쉼표 구분, 기본값: 전체)")
    parser.add_argument("--verbose", action="store_true",
                        help="상세 출력")

    # SAM 관련 인자
    parser.add_argument("--sam_segmenter", type=str, default="sam",
                        choices=["sam", "sam2", "medsam2"],
                        help="SAM 세그멘터 종류 (기본값: sam)")
    parser.add_argument("--sam_strategy", type=str, default="center",
                        choices=["center", "lesion_features", "both", "llm_guided"],
                        help="SAM 세그멘테이션 전략 (기본값: center)")

    args = parser.parse_args()

    # API 키 결정: 인자 > 환경변수
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("[오류] OpenAI API 키가 필요합니다.")
        print("  방법 1: --api_key 인자로 전달")
        print("  방법 2: .env 파일에 OPENAI_API_KEY 설정")
        print("  방법 3: 환경변수 OPENAI_API_KEY 설정")
        sys.exit(1)

    # ============ 설정 ============

    # 출력 디렉터리 생성
    output_dir = create_output_directory(args.output_dir, args.test_mode)
    logger = setup_logging(output_dir)

    logger.info("=" * 60)
    logger.info("8가지 피부과 진단 방법 비교 실험")
    logger.info("=" * 60)
    logger.info(f"입력 CSV: {args.input_csv}")
    logger.info(f"출력 디렉터리: {output_dir}")
    logger.info(f"모델: {args.model}")
    logger.info(f"테스트 모드: {args.test_mode}")
    logger.info(f"SAM 세그멘터: {args.sam_segmenter}")
    logger.info(f"SAM 전략: {args.sam_strategy}")

    # 데이터 로드
    num_samples = args.num_samples if args.test_mode else None
    df = load_dataset(args.input_csv, num_samples)
    logger.info(f"로드된 샘플 수: {len(df)}")

    # 이미지 기본 디렉터리
    if args.image_base_dir:
        image_base_dir = args.image_base_dir
    else:
        # CSV와 같은 위치 사용 (이미지가 CSV와 같은 폴더의 하위에 있음)
        image_base_dir = str(Path(args.input_csv).parent)

    logger.info(f"이미지 기본 디렉터리: {image_base_dir}")

    # 실행할 방법 결정
    all_methods = [
        "baseline_labels",
        "baseline_no_labels",
        "baseline_hierarchical",
        "dermatology_agent",
        "react_agent",
        "sam_baseline",
        "sam_llm_guided",
        "dermatology_agent_sam"
    ]

    if args.methods:
        methods_to_run = [m.strip() for m in args.methods.split(",")]
    else:
        methods_to_run = all_methods

    logger.info(f"실행할 방법: {methods_to_run}")

    # VLM 초기화
    logger.info("VLM 모델 초기화 중...")
    vlm = GPT4o(api_key=api_key, use_labels_prompt=False)
    logger.info("VLM 모델 초기화 완료")

    # 평가기 초기화
    evaluator = HierarchicalEvaluator()

    # 실험 설정 저장
    config = ExperimentConfig(
        timestamp=datetime.now().isoformat(),
        input_csv=args.input_csv,
        output_dir=str(output_dir),
        model=args.model,
        num_samples=len(df),
        test_mode=args.test_mode,
        methods=methods_to_run
    )
    # SAM 설정 추가
    config_dict = config.to_dict()
    config_dict["sam_segmenter"] = args.sam_segmenter
    config_dict["sam_strategy"] = args.sam_strategy
    with open(output_dir / "experiment_config.json", 'w', encoding='utf-8') as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)

    # ============ 실험 실행 ============

    all_results: Dict[str, List[MethodResult]] = {}
    all_evaluations: Dict[str, MethodEvaluation] = {}

    method_functions = {
        "baseline_labels": run_baseline_with_labels,
        "baseline_no_labels": run_baseline_without_labels,
        "baseline_hierarchical": run_baseline_hierarchical,
        "dermatology_agent": run_dermatology_agent,
        "react_agent": run_react_agent,
        "sam_baseline": run_sam_baseline,
        "sam_llm_guided": run_sam_llm_guided,
        "dermatology_agent_sam": run_dermatology_agent_sam
    }

    # SAM 관련 방법들
    sam_methods = ["sam_baseline", "sam_llm_guided", "dermatology_agent_sam"]

    for i, method in enumerate(methods_to_run, 1):
        if method not in method_functions:
            logger.warning(f"알 수 없는 방법: {method}, 건너뜀")
            continue

        logger.info(f"\n[{i}/{len(methods_to_run)}] {method} 실행 중...")

        try:
            # SAM 관련 메서드는 추가 인자 전달
            if method in sam_methods:
                # sam_llm_guided는 전략이 llm_guided로 고정
                if method == "sam_llm_guided":
                    results = method_functions[method](
                        vlm=vlm,
                        df=df,
                        image_base_dir=image_base_dir,
                        logger=logger,
                        verbose=args.verbose
                    )
                else:
                    results = method_functions[method](
                        vlm=vlm,
                        df=df,
                        image_base_dir=image_base_dir,
                        logger=logger,
                        verbose=args.verbose,
                        segmenter_type=args.sam_segmenter,
                        segmentation_strategy=args.sam_strategy
                    )
            else:
                results = method_functions[method](
                    vlm=vlm,
                    df=df,
                    image_base_dir=image_base_dir,
                    logger=logger,
                    verbose=args.verbose
                )

            all_results[method] = results

            # 예측 결과 CSV 저장
            pred_csv_path = output_dir / "predictions" / f"{i}_{method}.csv"
            save_predictions_csv(results, pred_csv_path, method)
            logger.info(f"예측 결과 저장: {pred_csv_path}")

            # 평가 (에이전트 방법은 Top-K 평가 사용)
            use_top_k = method in ["dermatology_agent", "react_agent", "dermatology_agent_sam"]
            evaluation = evaluate_method_results(results, evaluator, method, use_top_k=use_top_k)
            all_evaluations[method] = evaluation

            logger.info(f"  Exact Match: {evaluation.exact_match:.4f}")
            logger.info(f"  Hierarchical F1: {evaluation.hierarchical_f1:.4f}")

            # Top-K 메트릭 출력 (에이전트 방법인 경우)
            if use_top_k and evaluation.top_k_accuracy:
                for k, acc in sorted(evaluation.top_k_accuracy.items()):
                    logger.info(f"  Top-{k} Accuracy: {acc:.4f}")

        except Exception as e:
            logger.error(f"{method} 실행 중 오류: {e}")
            import traceback
            logger.error(traceback.format_exc())

    # ============ 결과 저장 ============

    logger.info("\n" + "=" * 60)
    logger.info("결과 저장 중...")
    logger.info("=" * 60)

    # 메트릭 요약 CSV
    metrics_csv_path = output_dir / "evaluation" / "metrics_summary.csv"
    save_metrics_summary_csv(all_evaluations, metrics_csv_path)
    logger.info(f"메트릭 요약 저장: {metrics_csv_path}")

    # 샘플별 비교 CSV
    if len(all_results) > 0:
        comparison_csv_path = output_dir / "evaluation" / "per_sample_comparison.csv"
        save_per_sample_comparison_csv(all_results, evaluator, comparison_csv_path)
        logger.info(f"샘플별 비교 저장: {comparison_csv_path}")

        # 상세 분석 CSV
        detailed_csv_path = output_dir / "evaluation" / "detailed_analysis.csv"
        save_detailed_analysis_csv(all_results, evaluator, detailed_csv_path)
        logger.info(f"상세 분석 저장: {detailed_csv_path}")

    # 방법 간 차이 분석
    if len(all_results) > 1:
        diff_analysis = analyze_method_differences(all_results)
        diff_path = output_dir / "evaluation" / "method_differences.json"
        with open(diff_path, 'w', encoding='utf-8') as f:
            json.dump(diff_analysis, f, indent=2, ensure_ascii=False, default=str)
        logger.info(f"방법 간 차이 분석 저장: {diff_path}")

    # 에이전트 트레이스 저장 (DermatologyAgent, ReActAgent, DermatologyAgentSAM)
    agent_methods_in_results = [m for m in ["dermatology_agent", "react_agent", "dermatology_agent_sam"] if m in all_results]
    if agent_methods_in_results:
        logger.info("\n에이전트 트레이스 저장 중...")
        save_all_agent_traces(all_results, output_dir)
        for method_name in agent_methods_in_results:
            trace_dir = output_dir / "agent_traces" / method_name
            logger.info(f"  {method_name} 트레이스: {trace_dir}")

    # ============ 결과 출력 ============

    print_metrics_summary(all_evaluations)

    logger.info("\n" + "=" * 60)
    logger.info("실험 완료!")
    logger.info(f"결과 저장 위치: {output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
