"""
실험 유틸리티 모듈

5가지 진단 방법 비교 실험을 위한 유틸리티 함수들:
- 로깅 설정
- CSV 저장/로드
- 결과 정리 및 비교
- 타임스탬프 기반 출력 디렉터리 관리
"""

import os
import sys
import json
import logging
import csv
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
import pandas as pd

# 경로 설정
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPT_DIR.parent / "eval"))
sys.path.insert(0, str(SCRIPT_DIR.parent / "baseline"))

from ontology_utils import OntologyTree


# ============ 데이터 클래스 ============

@dataclass
class AgentToolCall:
    """에이전트 도구 호출 기록"""
    tool_name: str
    tool_input: Dict[str, Any]
    tool_output: str
    timestamp: str = ""

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class AgentStep:
    """에이전트 단일 추론 단계"""
    step_num: int
    thought: str = ""
    action: str = ""
    action_input: Dict[str, Any] = field(default_factory=dict)
    observation: str = ""
    tool_calls: List[AgentToolCall] = field(default_factory=list)

    def to_dict(self) -> Dict:
        result = {
            "step_num": self.step_num,
            "thought": self.thought,
            "action": self.action,
            "action_input": self.action_input,
            "observation": self.observation[:1000] if self.observation else "",  # 너무 길면 자르기
        }
        if self.tool_calls:
            result["tool_calls"] = [tc.to_dict() for tc in self.tool_calls]
        return result


@dataclass
class AgentTrace:
    """에이전트 전체 추론 트레이스"""
    sample_id: int
    filename: str
    agent_type: str  # "dermatology_agent" or "react_agent"

    # 입력
    image_path: str = ""

    # 추론 과정
    steps: List[AgentStep] = field(default_factory=list)

    # 관찰 결과 (초기 이미지 분석)
    observations: Dict[str, Any] = field(default_factory=dict)

    # 온톨로지 탐색 경로
    ontology_path: List[str] = field(default_factory=list)
    explored_categories: List[str] = field(default_factory=list)

    # 후보군
    candidates_considered: List[str] = field(default_factory=list)
    candidate_scores: Dict[str, float] = field(default_factory=dict)

    # 최종 결과
    primary_diagnosis: str = ""
    differential_diagnoses: List[str] = field(default_factory=list)
    confidence: float = 0.0
    final_reasoning: str = ""

    # 메타 정보
    total_steps: int = 0
    total_vlm_calls: int = 0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "sample_id": self.sample_id,
            "filename": self.filename,
            "agent_type": self.agent_type,
            "image_path": self.image_path,
            "steps": [s.to_dict() for s in self.steps],
            "observations": self.observations,
            "ontology_path": self.ontology_path,
            "explored_categories": self.explored_categories,
            "candidates_considered": self.candidates_considered[:20],  # 상위 20개만
            "candidate_scores": dict(list(self.candidate_scores.items())[:20]),
            "primary_diagnosis": self.primary_diagnosis,
            "differential_diagnoses": self.differential_diagnoses,
            "confidence": self.confidence,
            "final_reasoning": self.final_reasoning[:500] if self.final_reasoning else "",
            "total_steps": self.total_steps,
            "total_vlm_calls": self.total_vlm_calls,
            "errors": self.errors,
            "warnings": self.warnings,
        }

    def to_readable_text(self) -> str:
        """가독성 좋은 텍스트 형식으로 변환"""
        lines = []
        lines.append("=" * 80)
        lines.append(f"[Sample {self.sample_id}] {self.filename}")
        lines.append(f"Agent: {self.agent_type}")
        lines.append("=" * 80)

        # 관찰 결과
        if self.observations:
            lines.append("\n📷 [Initial Observations]")
            for key, value in self.observations.items():
                if isinstance(value, list):
                    lines.append(f"  • {key}: {', '.join(str(v) for v in value)}")
                else:
                    lines.append(f"  • {key}: {value}")

        # 추론 단계
        if self.steps:
            lines.append(f"\n🔄 [Reasoning Steps] ({len(self.steps)} steps)")
            for step in self.steps:
                lines.append(f"\n  --- Step {step.step_num} ---")
                if step.thought:
                    lines.append(f"  💭 Thought: {step.thought[:200]}...")
                if step.action:
                    lines.append(f"  🔧 Action: {step.action}")
                if step.action_input:
                    lines.append(f"     Input: {json.dumps(step.action_input, ensure_ascii=False)[:200]}")
                if step.observation:
                    obs_preview = step.observation[:300].replace('\n', ' ')
                    lines.append(f"  📋 Observation: {obs_preview}...")

        # 온톨로지 경로
        if self.ontology_path:
            lines.append(f"\n🌳 [Ontology Path]")
            lines.append(f"  {' → '.join(self.ontology_path)}")

        # 후보군
        if self.candidates_considered:
            lines.append(f"\n🎯 [Candidates] ({len(self.candidates_considered)} considered)")
            top_candidates = self.candidates_considered[:5]
            for cand in top_candidates:
                score = self.candidate_scores.get(cand, 0)
                lines.append(f"  • {cand} (score: {score:.2f})")

        # 최종 결과
        lines.append(f"\n✅ [Final Diagnosis]")
        lines.append(f"  Primary: {self.primary_diagnosis}")
        lines.append(f"  Confidence: {self.confidence:.2f}")
        if self.differential_diagnoses:
            lines.append(f"  Differentials: {', '.join(self.differential_diagnoses[:3])}")
        if self.final_reasoning:
            lines.append(f"  Reasoning: {self.final_reasoning[:300]}...")

        # 메타 정보
        lines.append(f"\n📊 [Stats]")
        lines.append(f"  Total Steps: {self.total_steps}")
        lines.append(f"  VLM Calls: {self.total_vlm_calls}")
        if self.errors:
            lines.append(f"  ⚠️ Errors: {len(self.errors)}")
        if self.warnings:
            lines.append(f"  ⚠️ Warnings: {len(self.warnings)}")

        lines.append("\n" + "=" * 80 + "\n")
        return "\n".join(lines)


@dataclass
class MethodResult:
    """단일 방법의 단일 샘플 결과"""
    sample_id: int
    filename: str
    ground_truth: str
    hierarchical_gt: str
    prediction: str  # 주요 예측 (Top-1)
    confidence: float = 0.0
    reasoning: str = ""
    raw_response: str = ""
    all_predictions: List[str] = field(default_factory=list)  # 전체 예측 리스트 (Top-K용)
    agent_trace: Optional[AgentTrace] = None  # 에이전트 상세 트레이스

    def to_dict(self) -> Dict:
        result = asdict(self)
        # agent_trace는 별도 처리 (너무 크므로 기본 dict에서 제외)
        if 'agent_trace' in result:
            del result['agent_trace']
        return result


@dataclass
class MethodEvaluation:
    """단일 방법의 전체 평가 결과"""
    method_name: str
    exact_match: float = 0.0
    partial_match: float = 0.0
    hierarchical_f1: float = 0.0
    avg_distance: float = 0.0
    partial_credit: float = 0.0
    level_accuracy: Dict[int, float] = field(default_factory=dict)
    total_samples: int = 0
    valid_samples: int = 0
    # Top-K 메트릭 추가
    top_k_accuracy: Dict[int, float] = field(default_factory=dict)  # {k: accuracy}
    top_k_hierarchical_f1: Dict[int, float] = field(default_factory=dict)  # {k: h_f1}

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class ExperimentConfig:
    """실험 설정"""
    timestamp: str
    input_csv: str
    output_dir: str
    model: str
    num_samples: int
    test_mode: bool
    methods: List[str]

    def to_dict(self) -> Dict:
        return asdict(self)


# ============ 로깅 설정 ============

def setup_logging(output_dir: Path, name: str = "experiment") -> logging.Logger:
    """
    실험 로깅 설정

    Args:
        output_dir: 로그 파일 저장 디렉터리
        name: 로거 이름

    Returns:
        설정된 로거
    """
    # 로그 디렉터리 생성
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # 로거 생성
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # 기존 핸들러 제거
    logger.handlers = []

    # 파일 핸들러 (한국어 지원)
    log_file = log_dir / f"{name}.log"
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # 콘솔 핸들러
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('[%(levelname)s] %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    return logger


# ============ 출력 디렉터리 관리 ============

def create_output_directory(base_dir: str, test_mode: bool = False) -> Path:
    """
    타임스탬프 기반 출력 디렉터리 생성

    Args:
        base_dir: 기본 출력 디렉터리
        test_mode: 테스트 모드 여부

    Returns:
        생성된 출력 디렉터리 경로
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = "_test" if test_mode else ""
    output_dir = Path(base_dir) / f"{timestamp}{suffix}"

    # 하위 디렉터리 생성
    (output_dir / "logs").mkdir(parents=True, exist_ok=True)
    (output_dir / "predictions").mkdir(parents=True, exist_ok=True)
    (output_dir / "evaluation").mkdir(parents=True, exist_ok=True)

    return output_dir


# ============ CSV 저장 함수 ============

def save_predictions_csv(
    results: List[MethodResult],
    output_path: Path,
    method_name: str
) -> None:
    """
    예측 결과를 CSV로 저장

    Args:
        results: 예측 결과 리스트
        output_path: 저장 경로
        method_name: 방법 이름
    """
    df = pd.DataFrame([r.to_dict() for r in results])
    df.to_csv(output_path, index=False, encoding='utf-8-sig')


def save_metrics_summary_csv(
    evaluations: Dict[str, MethodEvaluation],
    output_path: Path
) -> None:
    """
    메트릭 요약을 CSV로 저장

    Args:
        evaluations: {method_name: MethodEvaluation} 딕셔너리
        output_path: 저장 경로
    """
    rows = []
    for method_name, eval_result in evaluations.items():
        row = {
            'method': method_name,
            'exact_match': eval_result.exact_match,
            'partial_match': eval_result.partial_match,
            'hierarchical_f1': eval_result.hierarchical_f1,
            'avg_distance': eval_result.avg_distance,
            'partial_credit': eval_result.partial_credit,
            'total_samples': eval_result.total_samples,
            'valid_samples': eval_result.valid_samples,
        }

        # 레벨별 정확도 추가
        for level, acc in eval_result.level_accuracy.items():
            row[f'level_{level}_acc'] = acc

        # Top-K 정확도 추가 (있는 경우)
        if eval_result.top_k_accuracy:
            for k, acc in eval_result.top_k_accuracy.items():
                row[f'top_{k}_accuracy'] = acc

        # Top-K Hierarchical F1 추가 (있는 경우)
        if eval_result.top_k_hierarchical_f1:
            for k, f1 in eval_result.top_k_hierarchical_f1.items():
                row[f'top_{k}_h_f1'] = f1

        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False, encoding='utf-8-sig')


def save_per_sample_comparison_csv(
    all_results: Dict[str, List[MethodResult]],
    evaluator,  # HierarchicalEvaluator
    output_path: Path
) -> None:
    """
    샘플별 비교 CSV 저장

    Args:
        all_results: {method_name: [MethodResult, ...]} 딕셔너리
        evaluator: HierarchicalEvaluator 인스턴스
        output_path: 저장 경로
    """
    # 방법 이름 정렬
    method_names = sorted(all_results.keys())

    if not method_names or not all_results[method_names[0]]:
        return

    # 샘플 수
    num_samples = len(all_results[method_names[0]])

    rows = []
    for i in range(num_samples):
        row = {
            'sample_id': i,
            'filename': all_results[method_names[0]][i].filename,
            'ground_truth': all_results[method_names[0]][i].ground_truth,
            'hierarchical_gt': all_results[method_names[0]][i].hierarchical_gt,
        }

        # 각 방법별 결과 추가
        for j, method in enumerate(method_names, 1):
            result = all_results[method][i]
            pred = result.prediction
            gt = result.ground_truth

            # 예측값이 리스트인 경우 문자열로 변환
            if isinstance(pred, list):
                pred = pred[0] if pred else ""
            if isinstance(gt, list):
                gt = gt[0] if gt else ""

            # 거리 계산
            try:
                dist = evaluator.tree.get_hierarchical_distance(gt, pred) if pred else -1
            except Exception:
                dist = -1
            exact = 1 if gt == pred else 0

            row[f'm{j}_pred'] = pred
            row[f'm{j}_exact'] = exact
            row[f'm{j}_dist'] = dist
            row[f'm{j}_conf'] = result.confidence

        # 최선의 방법 결정
        best_methods = []
        min_dist = float('inf')
        for j, method in enumerate(method_names, 1):
            dist = row.get(f'm{j}_dist', -1)
            if dist >= 0 and dist < min_dist:
                min_dist = dist
                best_methods = [f'm{j}']
            elif dist >= 0 and dist == min_dist:
                best_methods.append(f'm{j}')

        row['best_method'] = '/'.join(best_methods) if best_methods else 'none'
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False, encoding='utf-8-sig')


def save_detailed_analysis_csv(
    all_results: Dict[str, List[MethodResult]],
    evaluator,  # HierarchicalEvaluator
    output_path: Path
) -> None:
    """
    상세 분석 CSV 저장 (각 샘플-방법 조합별 한 행)

    Args:
        all_results: {method_name: [MethodResult, ...]} 딕셔너리
        evaluator: HierarchicalEvaluator 인스턴스
        output_path: 저장 경로
    """
    rows = []

    for method_name, results in all_results.items():
        for result in results:
            gt = result.ground_truth
            pred = result.prediction

            # 예측값이 리스트인 경우 문자열로 변환
            if isinstance(pred, list):
                pred = pred[0] if pred else ""
            if isinstance(gt, list):
                gt = gt[0] if gt else ""

            # 계층적 유사도 계산
            if pred and gt:
                try:
                    similarity = evaluator.hierarchical_similarity(gt, pred)
                    distance = evaluator.tree.get_hierarchical_distance(gt, pred)
                    lca = evaluator.tree.get_lca(gt, pred)
                except Exception:
                    similarity = 0.0
                    distance = -1
                    lca = ""
            else:
                similarity = 0.0
                distance = -1
                lca = ""

            row = {
                'sample_id': result.sample_id,
                'filename': result.filename,
                'ground_truth': gt,
                'method': method_name,
                'prediction': pred,
                'exact_match': 1 if gt == pred else 0,
                'hierarchical_similarity': round(similarity, 4),
                'tree_distance': distance,
                'common_ancestor': lca or "",
                'confidence': result.confidence,
                'reasoning_summary': result.reasoning[:200] if result.reasoning else ""
            }
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False, encoding='utf-8-sig')


def save_experiment_config(config: ExperimentConfig, output_path: Path) -> None:
    """
    실험 설정을 JSON으로 저장

    Args:
        config: 실험 설정
        output_path: 저장 경로
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(config.to_dict(), f, indent=2, ensure_ascii=False)


# ============ 에이전트 트레이스 저장 함수 ============

def save_agent_traces_json(
    results: List[MethodResult],
    output_path: Path,
    method_name: str
) -> None:
    """
    에이전트 트레이스를 JSON으로 저장 (구조화된 형식)

    Args:
        results: MethodResult 리스트 (agent_trace 포함)
        output_path: 저장 경로
        method_name: 방법 이름
    """
    traces = []
    for result in results:
        if result.agent_trace:
            trace_dict = result.agent_trace.to_dict()
            trace_dict["ground_truth"] = result.ground_truth
            trace_dict["hierarchical_gt"] = result.hierarchical_gt
            traces.append(trace_dict)

    if traces:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({
                "method": method_name,
                "total_samples": len(traces),
                "traces": traces
            }, f, indent=2, ensure_ascii=False)


def save_agent_traces_readable(
    results: List[MethodResult],
    output_path: Path,
    method_name: str
) -> None:
    """
    에이전트 트레이스를 가독성 좋은 텍스트로 저장

    Args:
        results: MethodResult 리스트 (agent_trace 포함)
        output_path: 저장 경로
        method_name: 방법 이름
    """
    lines = []
    lines.append("=" * 80)
    lines.append(f"Agent Trace Report: {method_name}")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 80)
    lines.append("")

    correct_count = 0
    total_with_trace = 0

    for result in results:
        if result.agent_trace:
            total_with_trace += 1
            trace = result.agent_trace

            # 정답 여부 표시
            is_correct = result.ground_truth == result.prediction
            if is_correct:
                correct_count += 1
            status = "✅ CORRECT" if is_correct else "❌ WRONG"

            lines.append(f"\n{'#' * 80}")
            lines.append(f"# Sample {result.sample_id}: {result.filename}")
            lines.append(f"# Status: {status}")
            lines.append(f"# Ground Truth: {result.ground_truth}")
            lines.append(f"# Prediction: {result.prediction}")
            lines.append(f"{'#' * 80}")

            # 트레이스 내용 추가
            lines.append(trace.to_readable_text())

    # 요약 정보
    summary = f"""
{'=' * 80}
SUMMARY
{'=' * 80}
Method: {method_name}
Total Samples with Trace: {total_with_trace}
Correct: {correct_count}
Accuracy: {correct_count / total_with_trace * 100:.1f}% (of traced samples)
{'=' * 80}
"""
    lines.insert(4, summary)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


def save_agent_single_trace(
    trace: AgentTrace,
    output_dir: Path,
    ground_truth: str = ""
) -> None:
    """
    단일 에이전트 트레이스를 개별 파일로 저장

    Args:
        trace: AgentTrace 인스턴스
        output_dir: 출력 디렉터리
        ground_truth: 정답 라벨
    """
    # 파일명 생성 (안전한 문자만 사용)
    safe_filename = trace.filename.replace('/', '_').replace('\\', '_')
    json_path = output_dir / f"trace_{trace.sample_id:04d}_{safe_filename}.json"
    txt_path = output_dir / f"trace_{trace.sample_id:04d}_{safe_filename}.txt"

    # JSON 저장
    trace_dict = trace.to_dict()
    trace_dict["ground_truth"] = ground_truth
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(trace_dict, f, indent=2, ensure_ascii=False)

    # 텍스트 저장
    text_content = trace.to_readable_text()
    text_content = f"Ground Truth: {ground_truth}\n\n" + text_content
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(text_content)


def save_all_agent_traces(
    all_results: Dict[str, List[MethodResult]],
    output_dir: Path
) -> None:
    """
    모든 에이전트 결과의 트레이스를 저장

    Args:
        all_results: {method_name: [MethodResult, ...]} 딕셔너리
        output_dir: 출력 디렉터리
    """
    agent_methods = ["dermatology_agent", "react_agent"]

    for method_name in agent_methods:
        if method_name not in all_results:
            continue

        results = all_results[method_name]

        # 에이전트 트레이스 디렉터리 생성
        trace_dir = output_dir / "agent_traces" / method_name
        trace_dir.mkdir(parents=True, exist_ok=True)

        # JSON 통합 파일
        json_path = trace_dir / f"{method_name}_all_traces.json"
        save_agent_traces_json(results, json_path, method_name)

        # 가독성 좋은 텍스트 파일
        txt_path = trace_dir / f"{method_name}_traces_readable.txt"
        save_agent_traces_readable(results, txt_path, method_name)

        # 개별 트레이스 파일 (선택적)
        individual_dir = trace_dir / "individual"
        individual_dir.mkdir(parents=True, exist_ok=True)

        for result in results:
            if result.agent_trace:
                save_agent_single_trace(
                    result.agent_trace,
                    individual_dir,
                    result.ground_truth
                )


# ============ 데이터 로드 함수 ============

def load_dataset(csv_path: str, num_samples: Optional[int] = None) -> pd.DataFrame:
    """
    데이터셋 CSV 로드

    Args:
        csv_path: CSV 파일 경로
        num_samples: 로드할 샘플 수 (None이면 전체)

    Returns:
        DataFrame
    """
    df = pd.read_csv(csv_path, encoding='utf-8-sig')

    if num_samples is not None and num_samples < len(df):
        df = df.head(num_samples)

    return df


# ============ 온톨로지 트리 텍스트 생성 ============

def build_ontology_tree_text(
    ontology: Dict,
    node: str = "root",
    indent: int = 0,
    max_depth: int = 10,
    prefix: str = ""
) -> str:
    """
    온톨로지 JSON을 트리 텍스트로 변환

    Args:
        ontology: 온톨로지 딕셔너리
        node: 현재 노드
        indent: 들여쓰기 레벨
        max_depth: 최대 깊이
        prefix: 접두사 (트리 구조 표현)

    Returns:
        트리 텍스트
    """
    if indent > max_depth:
        return ""

    lines = []
    children = ontology.get(node, [])

    for i, child in enumerate(children):
        is_last = (i == len(children) - 1)

        # 현재 노드 출력
        connector = "└── " if is_last else "├── "
        lines.append(f"{prefix}{connector}{child}")

        # 자식 노드들의 prefix 결정
        new_prefix = prefix + ("    " if is_last else "│   ")

        # 재귀적으로 자식 노드 처리
        child_tree = build_ontology_tree_text(
            ontology, child, indent + 1, max_depth, new_prefix
        )
        if child_tree:
            lines.append(child_tree)

    return "\n".join(lines)


def get_ontology_tree_for_prompt(ontology_path: Optional[str] = None) -> str:
    """
    프롬프트용 온톨로지 트리 텍스트 생성

    Args:
        ontology_path: 온톨로지 JSON 경로

    Returns:
        프롬프트에 포함할 트리 텍스트
    """
    tree = OntologyTree(ontology_path)
    tree_text = build_ontology_tree_text(tree.ontology, "root", 0, 10, "")
    return tree_text


# ============ 결과 분석 함수 ============

def analyze_method_differences(
    all_results: Dict[str, List[MethodResult]]
) -> Dict[str, Any]:
    """
    방법 간 차이 분석

    Args:
        all_results: {method_name: [MethodResult, ...]} 딕셔너리

    Returns:
        분석 결과 딕셔너리
    """
    method_names = list(all_results.keys())
    if not method_names:
        return {}

    num_samples = len(all_results[method_names[0]])

    # 모든 방법이 맞춘 샘플
    all_correct = []
    # 모든 방법이 틀린 샘플
    all_wrong = []
    # 방법 간 차이가 나는 샘플
    different = []

    for i in range(num_samples):
        gt = all_results[method_names[0]][i].ground_truth
        preds = [all_results[m][i].prediction for m in method_names]

        correct_count = sum(1 for p in preds if p == gt)

        if correct_count == len(method_names):
            all_correct.append(i)
        elif correct_count == 0:
            all_wrong.append(i)
        else:
            different.append({
                'sample_id': i,
                'ground_truth': gt,
                'predictions': {m: all_results[m][i].prediction for m in method_names},
                'correct_methods': [m for m in method_names if all_results[m][i].prediction == gt]
            })

    return {
        'all_correct_count': len(all_correct),
        'all_wrong_count': len(all_wrong),
        'different_count': len(different),
        'different_samples': different
    }


# ============ 결과 출력 함수 ============

def print_metrics_summary(evaluations: Dict[str, MethodEvaluation]) -> None:
    """
    메트릭 요약 출력

    Args:
        evaluations: {method_name: MethodEvaluation} 딕셔너리
    """
    print("\n" + "=" * 80)
    print("메트릭 요약 (METRICS SUMMARY)")
    print("=" * 80)

    # 헤더 출력
    header = f"{'Method':<25} {'Exact':>8} {'Partial':>8} {'H-F1':>8} {'Dist':>8} {'Credit':>8}"
    print(header)
    print("-" * 80)

    for method_name, eval_result in evaluations.items():
        row = (
            f"{method_name:<25} "
            f"{eval_result.exact_match:>8.4f} "
            f"{eval_result.partial_match:>8.4f} "
            f"{eval_result.hierarchical_f1:>8.4f} "
            f"{eval_result.avg_distance:>8.2f} "
            f"{eval_result.partial_credit:>8.4f}"
        )
        print(row)

    print("=" * 80)

    # 레벨별 정확도
    print("\n레벨별 정확도 (LEVEL ACCURACY)")
    print("-" * 80)

    # 모든 레벨 수집
    all_levels = set()
    for eval_result in evaluations.values():
        all_levels.update(eval_result.level_accuracy.keys())

    if all_levels:
        levels = sorted(all_levels)
        header = f"{'Method':<25}" + "".join([f" L{l}:>8" for l in levels])
        print(f"{'Method':<25}" + "".join([f"{'L'+str(l):>10}" for l in levels]))
        print("-" * 80)

        for method_name, eval_result in evaluations.items():
            row = f"{method_name:<25}"
            for level in levels:
                acc = eval_result.level_accuracy.get(level, 0.0)
                row += f"{acc:>10.4f}"
            print(row)

    # Top-K 정확도 (있는 경우)
    has_top_k = any(eval_result.top_k_accuracy for eval_result in evaluations.values())
    if has_top_k:
        print("\n" + "=" * 80)
        print("Top-K 정확도 (TOP-K ACCURACY)")
        print("-" * 80)

        # 모든 K값 수집
        all_k_values = set()
        for eval_result in evaluations.values():
            if eval_result.top_k_accuracy:
                all_k_values.update(eval_result.top_k_accuracy.keys())

        if all_k_values:
            k_values = sorted(all_k_values)
            print(f"{'Method':<25}" + "".join([f"{'Top-'+str(k):>12}" for k in k_values]))
            print("-" * 80)

            for method_name, eval_result in evaluations.items():
                row = f"{method_name:<25}"
                for k in k_values:
                    acc = eval_result.top_k_accuracy.get(k, -1)
                    if acc >= 0:
                        row += f"{acc:>12.4f}"
                    else:
                        row += f"{'N/A':>12}"
                print(row)

    print("=" * 80)
