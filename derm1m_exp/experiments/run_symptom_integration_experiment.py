#!/usr/bin/env python3
"""
증상 분석 통합 실험 스크립트

SAM 세그멘테이션 + 증상 스키마 추출 + VLM 진단 파이프라인

실험 구성:
1. SAM 전략별 비교 (center, lesion_features, both, llm_guided)
2. 증상 정보 유무에 따른 비교
3. caption vs truncated_caption 비교

사용법 (Colab):
    !python run_symptom_integration_experiment.py \
        --input_csv /content/dataset/random100.csv \
        --image_dir /content/images \
        --output_dir /content/outputs \
        --api_key $OPENAI_API_KEY \
        --num_samples 10

Author: DermAgent Team
"""

import os
import sys
import json
import re
import argparse
import csv
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
import pandas as pd
import numpy as np

# 경로 설정
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPT_DIR.parent / "eval"))
sys.path.insert(0, str(SCRIPT_DIR.parent / "baseline"))
sys.path.insert(0, str(SCRIPT_DIR.parent / "DermAgent" / "agent"))

# .env 파일 로드
def load_env_file():
    env_paths = [
        SCRIPT_DIR / ".env",
        SCRIPT_DIR.parent / "baseline" / ".env",
        PROJECT_ROOT / ".env",
    ]
    for env_path in env_paths:
        if env_path.exists():
            print(f"[INFO] .env 파일 로드: {env_path}")
            with open(env_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        if key.strip() not in os.environ:
                            os.environ[key.strip()] = value.strip()
            return True
    return False

load_env_file()


# ============ 증상 스키마 정의 ============

class SymptomSeverity(Enum):
    NONE = "none"
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"


class SymptomDuration(Enum):
    ACUTE = "acute"
    SUBACUTE = "subacute"
    CHRONIC = "chronic"
    UNKNOWN = "unknown"


@dataclass
class DermatologySymptomSchema:
    """피부 질환 도메인 특화 증상 스키마"""

    # 주관적 증상
    pruritus: bool = False
    pruritus_severity: str = "none"
    pain: bool = False
    pain_type: str = ""
    pain_severity: str = "none"
    tenderness: bool = False
    burning: bool = False
    tingling: bool = False

    # 객관적 증상
    erythema: bool = False
    swelling: bool = False
    warmth: bool = False
    discharge: bool = False
    discharge_type: str = ""
    scaling: bool = False
    crusting: bool = False
    bleeding: bool = False

    # 전신 증상
    fever: bool = False
    malaise: bool = False

    # 시간적 특성
    duration: str = "unknown"
    onset: str = ""
    recurrent: bool = False

    # 메타데이터
    raw_text: str = ""
    extraction_source: str = ""
    extraction_confidence: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    def to_summary_string(self) -> str:
        """증상 요약 문자열 생성"""
        symptoms = []
        if self.pruritus:
            symptoms.append(f"pruritus({self.pruritus_severity})")
        if self.pain:
            symptoms.append(f"pain({self.pain_type or 'general'},{self.pain_severity})")
        if self.tenderness:
            symptoms.append("tenderness")
        if self.burning:
            symptoms.append("burning")
        if self.tingling:
            symptoms.append("tingling")
        if self.erythema:
            symptoms.append("erythema")
        if self.swelling:
            symptoms.append("swelling")
        if self.scaling:
            symptoms.append("scaling")
        if self.crusting:
            symptoms.append("crusting")
        if self.discharge:
            symptoms.append(f"discharge({self.discharge_type or 'unknown'})")
        if self.fever:
            symptoms.append("fever")

        return ", ".join(symptoms) if symptoms else "no symptoms extracted"


class SymptomExtractor:
    """텍스트에서 증상을 추출하는 모듈"""

    SYMPTOM_KEYWORDS = {
        "pruritus": ["itchy", "itching", "pruritus", "itch", "pruritic"],
        "pain": ["pain", "painful", "hurts", "aching", "sore", "painless"],
        "burning": ["burning", "burn", "hot sensation", "stinging"],
        "tenderness": ["tender", "tenderness", "sensitive to touch"],
        "tingling": ["tingling", "prickling", "pins and needles", "numbness"],
        "erythema": ["red", "redness", "erythema", "erythematous", "inflamed", "inflammation"],
        "swelling": ["swollen", "swelling", "edema", "puffy", "swells"],
        "scaling": ["scaly", "scaling", "scale", "flaky", "flaking", "desquamation"],
        "crusting": ["crusted", "crusting", "crust", "scab", "eschar"],
        "discharge": ["discharge", "oozing", "weeping", "pus", "purulent", "exudate"],
        "bleeding": ["bleeding", "hemorrhage", "blood"],
        "fever": ["fever", "febrile", "high temperature", "pyrexia"],
        "severe": ["severe", "intense", "extreme", "unbearable", "significant"],
        "mild": ["mild", "slight", "minor", "subtle"],
    }

    def __init__(self, vlm_model=None):
        self.vlm = vlm_model

    def extract_from_text_keyword(self, text: str) -> DermatologySymptomSchema:
        """키워드 기반 증상 추출"""
        schema = DermatologySymptomSchema()
        text_lower = text.lower()

        # 키워드 매칭
        for symptom, keywords in self.SYMPTOM_KEYWORDS.items():
            for keyword in keywords:
                if keyword in text_lower:
                    if symptom in ["pruritus", "pain", "burning", "tenderness",
                                   "tingling", "erythema", "swelling", "scaling",
                                   "crusting", "discharge", "bleeding", "fever"]:
                        setattr(schema, symptom, True)
                    break

        # 심각도 판단
        if "severe" in text_lower or "intense" in text_lower:
            if schema.pruritus:
                schema.pruritus_severity = "severe"
            if schema.pain:
                schema.pain_severity = "severe"
        elif "mild" in text_lower or "slight" in text_lower:
            if schema.pruritus:
                schema.pruritus_severity = "mild"
            if schema.pain:
                schema.pain_severity = "mild"
        else:
            if schema.pruritus:
                schema.pruritus_severity = "moderate"
            if schema.pain:
                schema.pain_severity = "moderate"

        schema.raw_text = text
        schema.extraction_source = "keyword"
        schema.extraction_confidence = 0.6

        return schema

    def extract_with_vlm(self, text: str, image_path: str = None) -> DermatologySymptomSchema:
        """VLM 기반 고급 증상 추출"""
        if self.vlm is None:
            return self.extract_from_text_keyword(text)

        prompt = f"""Analyze the following dermatological description and extract symptoms.

Description: "{text}"

Extract symptoms and respond in JSON format ONLY:
{{
    "subjective": {{
        "pruritus": {{"present": true/false, "severity": "none/mild/moderate/severe"}},
        "pain": {{"present": true/false, "type": "burning/stinging/throbbing/aching", "severity": "none/mild/moderate/severe"}},
        "tenderness": true/false,
        "burning": true/false,
        "tingling": true/false
    }},
    "objective": {{
        "erythema": true/false,
        "swelling": true/false,
        "warmth": true/false,
        "scaling": true/false,
        "crusting": true/false,
        "discharge": {{"present": true/false, "type": "serous/purulent/hemorrhagic"}},
        "bleeding": true/false
    }},
    "systemic": {{
        "fever": true/false,
        "malaise": true/false
    }},
    "temporal": {{
        "duration": "acute/subacute/chronic/unknown",
        "onset": "sudden/gradual/unknown",
        "recurrent": true/false
    }}
}}"""

        try:
            if image_path and os.path.exists(image_path):
                response = self.vlm.chat_img(prompt, [image_path], max_tokens=800)
            else:
                # 텍스트만 사용
                response = self.vlm.chat_img(prompt, [], max_tokens=800)

            schema = self._parse_vlm_response(response)
            schema.raw_text = text
            schema.extraction_source = "vlm"
            schema.extraction_confidence = 0.85
            return schema

        except Exception as e:
            print(f"[WARNING] VLM 증상 추출 실패: {e}")
            return self.extract_from_text_keyword(text)

    def _parse_vlm_response(self, response: str) -> DermatologySymptomSchema:
        """VLM 응답 파싱"""
        schema = DermatologySymptomSchema()

        try:
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                data = json.loads(json_match.group())

                # Subjective
                subj = data.get("subjective", {})
                pruritus_data = subj.get("pruritus", {})
                if isinstance(pruritus_data, dict):
                    schema.pruritus = pruritus_data.get("present", False)
                    schema.pruritus_severity = pruritus_data.get("severity", "none")

                pain_data = subj.get("pain", {})
                if isinstance(pain_data, dict):
                    schema.pain = pain_data.get("present", False)
                    schema.pain_type = pain_data.get("type", "")
                    schema.pain_severity = pain_data.get("severity", "none")

                schema.tenderness = subj.get("tenderness", False)
                schema.burning = subj.get("burning", False)
                schema.tingling = subj.get("tingling", False)

                # Objective
                obj = data.get("objective", {})
                schema.erythema = obj.get("erythema", False)
                schema.swelling = obj.get("swelling", False)
                schema.warmth = obj.get("warmth", False)
                schema.scaling = obj.get("scaling", False)
                schema.crusting = obj.get("crusting", False)
                schema.bleeding = obj.get("bleeding", False)

                discharge_data = obj.get("discharge", {})
                if isinstance(discharge_data, dict):
                    schema.discharge = discharge_data.get("present", False)
                    schema.discharge_type = discharge_data.get("type", "")

                # Systemic
                syst = data.get("systemic", {})
                schema.fever = syst.get("fever", False)
                schema.malaise = syst.get("malaise", False)

                # Temporal
                temp = data.get("temporal", {})
                schema.duration = temp.get("duration", "unknown")
                schema.onset = temp.get("onset", "")
                schema.recurrent = temp.get("recurrent", False)

        except Exception as e:
            print(f"[WARNING] JSON 파싱 실패: {e}")

        return schema


# ============ 실험 결과 데이터 클래스 ============

@dataclass
class ExperimentResult:
    """단일 실험 결과"""
    sample_id: int
    filename: str
    ground_truth: str
    hierarchical_gt: str

    # 예측 결과
    prediction: str
    confidence: float
    differential_diagnoses: List[str] = field(default_factory=list)

    # 실험 설정
    experiment_name: str = ""
    sam_strategy: str = ""
    use_symptoms: bool = False
    symptom_source: str = ""

    # 증상 분석 결과
    extracted_symptoms: str = ""
    symptom_extraction_confidence: float = 0.0

    # 세그멘테이션 결과
    segmentation_score: float = 0.0
    segmentation_method: str = ""

    # 평가 메트릭 (나중에 계산)
    exact_match: bool = False
    hierarchical_distance: int = -1
    hierarchical_f1: float = 0.0
    partial_credit: float = 0.0

    # 원본 응답
    raw_response: str = ""
    reasoning: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ExperimentSummary:
    """실험 요약 통계"""
    experiment_name: str
    sam_strategy: str
    use_symptoms: bool
    symptom_source: str

    total_samples: int = 0
    valid_samples: int = 0

    exact_match_accuracy: float = 0.0
    avg_hierarchical_distance: float = 0.0
    hierarchical_f1: float = 0.0
    partial_credit: float = 0.0

    avg_segmentation_score: float = 0.0
    avg_confidence: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


# ============ 메인 실험 클래스 ============

class SymptomIntegrationExperiment:
    """증상 통합 실험 클래스"""

    def __init__(
        self,
        vlm_model,
        ontology_path: Optional[str] = None,
        sam_checkpoint_dir: Optional[str] = None,
        verbose: bool = True
    ):
        self.vlm = vlm_model
        self.verbose = verbose

        # 온톨로지 로드
        try:
            from ontology_utils import OntologyTree
            self.tree = OntologyTree(ontology_path)
        except Exception as e:
            print(f"[WARNING] 온톨로지 로드 실패: {e}")
            self.tree = None

        # 평가기 로드
        try:
            from evaluation_metrics import HierarchicalEvaluator
            self.evaluator = HierarchicalEvaluator(ontology_path)
        except Exception as e:
            print(f"[WARNING] 평가기 로드 실패: {e}")
            self.evaluator = None

        # 증상 추출기
        self.symptom_extractor = SymptomExtractor(vlm_model=vlm_model)

        # SAM 세그멘터 초기화 시도
        self.sam_available = False
        self.segmenter = None
        try:
            sam_path = sam_checkpoint_dir or str(SCRIPT_DIR.parent / "SA-project-SAM")
            sys.path.insert(0, sam_path)
            from sam_masking import SAMSegmenter, apply_mask_to_image, load_image
            self.sam_segmenter_class = SAMSegmenter
            self.apply_mask_to_image = apply_mask_to_image
            self.load_image = load_image
            self.sam_available = True
            print("[INFO] SAM 모듈 로드 성공")
        except ImportError as e:
            print(f"[WARNING] SAM 모듈 로드 실패: {e}")
            print("[INFO] SAM 없이 실험 진행")

        # 결과 저장
        self.results: List[ExperimentResult] = []
        self.summaries: List[ExperimentSummary] = []

    def _log(self, message: str):
        if self.verbose:
            print(f"[Experiment] {message}")

    def _sanitize_caption(self, caption: str, disease_label: str) -> str:
        """caption에서 진단명 제거 (편향 방지)"""
        if not caption or not disease_label:
            return caption

        sanitized = caption

        # 다양한 형태의 진단명 제거
        patterns = [
            disease_label,
            disease_label.lower(),
            disease_label.upper(),
            disease_label.replace(" ", "_"),
            disease_label.replace("_", " "),
            disease_label.replace("-", " "),
        ]

        for pattern in patterns:
            if len(pattern) > 3:  # 너무 짧은 패턴은 제외
                sanitized = re.sub(re.escape(pattern), "[CONDITION]", sanitized, flags=re.IGNORECASE)

        return sanitized

    def _init_sam_segmenter(self, checkpoint_dir: str = None):
        """SAM 세그멘터 초기화 (지연 로딩)"""
        if not self.sam_available:
            return None

        if self.segmenter is None:
            try:
                self.segmenter = self.sam_segmenter_class(checkpoint_dir=checkpoint_dir)
                self._log("SAM 세그멘터 초기화 완료")
            except Exception as e:
                self._log(f"SAM 세그멘터 초기화 실패: {e}")
                self.sam_available = False
                return None

        return self.segmenter

    def _segment_image(self, image_path: str, strategy: str = "center") -> Dict[str, Any]:
        """이미지 세그멘테이션 수행"""
        if not self.sam_available:
            return {"mask": None, "score": 0.0, "method": "none", "overlay": None}

        segmenter = self._init_sam_segmenter()
        if segmenter is None:
            return {"mask": None, "score": 0.0, "method": "none", "overlay": None}

        try:
            image = self.load_image(image_path)

            if strategy == "center":
                result = segmenter.segment_center_focused(image)
            elif strategy == "lesion_features":
                result = segmenter.segment_lesion_features(image)
            elif strategy == "both":
                center_result = segmenter.segment_center_focused(image)
                feature_result = segmenter.segment_lesion_features(image)
                result = feature_result if feature_result.get("score", 0) > center_result.get("score", 0) else center_result
            else:  # llm_guided 또는 기타
                result = segmenter.segment_center_focused(image)

            # 오버레이 생성
            if result.get("mask") is not None:
                overlay = self.apply_mask_to_image(image, result["mask"], color=(255, 0, 0), alpha=0.4)
                result["overlay"] = overlay

            result["method"] = strategy
            return result

        except Exception as e:
            self._log(f"세그멘테이션 실패: {e}")
            return {"mask": None, "score": 0.0, "method": strategy, "overlay": None, "error": str(e)}

    def _diagnose_with_vlm(
        self,
        image_path: str,
        symptom_schema: Optional[DermatologySymptomSchema] = None,
        segmentation_result: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """VLM을 사용한 진단"""

        # 프롬프트 구성
        base_prompt = """You are a board-certified dermatology expert. Analyze this skin image and provide a diagnosis.

IMPORTANT: If no clear skin lesion is visible, the image does not show identifiable human skin, or you cannot make a confident diagnosis, set disease_label to "no definitive diagnosis".
"""

        # 증상 정보 추가
        symptom_info = ""
        if symptom_schema and symptom_schema.to_summary_string() != "no symptoms extracted":
            symptom_info = f"""
Patient-reported symptoms:
{symptom_schema.to_summary_string()}

Consider these symptoms when making your diagnosis.
"""

        # 세그멘테이션 정보 추가
        seg_info = ""
        if segmentation_result and segmentation_result.get("mask") is not None:
            seg_info = """
Note: The lesion area has been highlighted with a red overlay. Focus on the highlighted region.
"""

        prompt = f"""{base_prompt}
{symptom_info}
{seg_info}
Provide your diagnosis in JSON format:
{{
    "disease_label": "specific diagnosis or 'no definitive diagnosis'",
    "confidence": 0.0-1.0,
    "differential_diagnoses": ["other possible diagnoses"],
    "reasoning": "brief clinical reasoning"
}}

Provide ONLY the JSON output."""

        try:
            response = self.vlm.chat_img(prompt, [image_path], max_tokens=600)

            # JSON 파싱
            parsed = {}
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                parsed = json.loads(json_match.group())

            return {
                "disease_label": parsed.get("disease_label", "no definitive diagnosis"),
                "confidence": float(parsed.get("confidence", 0.5)),
                "differential_diagnoses": parsed.get("differential_diagnoses", []),
                "reasoning": parsed.get("reasoning", ""),
                "raw_response": response
            }

        except Exception as e:
            self._log(f"VLM 진단 실패: {e}")
            return {
                "disease_label": "no definitive diagnosis",
                "confidence": 0.0,
                "differential_diagnoses": [],
                "reasoning": f"Error: {e}",
                "raw_response": ""
            }

    def run_single_experiment(
        self,
        sample_id: int,
        filename: str,
        image_path: str,
        ground_truth: str,
        hierarchical_gt: str,
        caption: str = "",
        experiment_name: str = "",
        sam_strategy: str = "none",
        use_symptoms: bool = False,
        symptom_source: str = ""
    ) -> ExperimentResult:
        """단일 샘플에 대한 실험 수행"""

        result = ExperimentResult(
            sample_id=sample_id,
            filename=filename,
            ground_truth=ground_truth,
            hierarchical_gt=hierarchical_gt,
            experiment_name=experiment_name,
            sam_strategy=sam_strategy,
            use_symptoms=use_symptoms,
            symptom_source=symptom_source
        )

        # Step 0: 증상 추출
        symptom_schema = None
        if use_symptoms and caption:
            sanitized_caption = self._sanitize_caption(caption, ground_truth)
            symptom_schema = self.symptom_extractor.extract_with_vlm(sanitized_caption, image_path)
            result.extracted_symptoms = symptom_schema.to_summary_string()
            result.symptom_extraction_confidence = symptom_schema.extraction_confidence

        # Step 0: SAM 세그멘테이션
        seg_result = None
        if sam_strategy != "none" and self.sam_available:
            seg_result = self._segment_image(image_path, sam_strategy)
            result.segmentation_score = seg_result.get("score", 0.0)
            result.segmentation_method = seg_result.get("method", "")

        # Step 1: VLM 진단
        diagnosis = self._diagnose_with_vlm(image_path, symptom_schema, seg_result)

        result.prediction = diagnosis.get("disease_label", "no definitive diagnosis")
        result.confidence = diagnosis.get("confidence", 0.0)
        result.differential_diagnoses = diagnosis.get("differential_diagnoses", [])
        result.reasoning = diagnosis.get("reasoning", "")
        result.raw_response = diagnosis.get("raw_response", "")

        # 평가 메트릭 계산
        if self.evaluator:
            try:
                eval_result = self.evaluator.evaluate_single([ground_truth], [result.prediction])
                result.exact_match = eval_result.get("exact_match", 0.0) > 0
                result.hierarchical_f1 = eval_result.get("hierarchical_f1", 0.0)
                result.partial_credit = eval_result.get("partial_credit", 0.0)

                if self.tree:
                    dist = self.tree.get_hierarchical_distance(ground_truth, result.prediction)
                    result.hierarchical_distance = dist if dist >= 0 else -1
            except Exception as e:
                self._log(f"평가 실패: {e}")

        return result

    def run_experiment_batch(
        self,
        df: pd.DataFrame,
        image_base_dir: str,
        experiment_name: str,
        sam_strategy: str = "none",
        use_symptoms: bool = False,
        symptom_source: str = "caption",  # caption or truncated_caption
        num_samples: Optional[int] = None
    ) -> List[ExperimentResult]:
        """배치 실험 수행"""

        self._log(f"=" * 60)
        self._log(f"실험 시작: {experiment_name}")
        self._log(f"  SAM 전략: {sam_strategy}")
        self._log(f"  증상 사용: {use_symptoms}")
        self._log(f"  증상 소스: {symptom_source}")
        self._log(f"=" * 60)

        if num_samples:
            df = df.head(num_samples)

        results = []

        try:
            from tqdm import tqdm
            iterator = tqdm(df.iterrows(), total=len(df), desc=experiment_name)
        except ImportError:
            iterator = df.iterrows()

        for idx, row in iterator:
            filename = row.get('filename', '')
            image_path = os.path.join(image_base_dir, filename)

            if not os.path.exists(image_path):
                self._log(f"  [SKIP] 이미지 없음: {filename}")
                continue

            # 증상 소스 선택
            caption = ""
            if use_symptoms:
                if symptom_source == "truncated_caption":
                    caption = str(row.get('truncated_caption', ''))
                else:
                    caption = str(row.get('caption', ''))

            result = self.run_single_experiment(
                sample_id=idx,
                filename=filename,
                image_path=image_path,
                ground_truth=str(row.get('disease_label', '')),
                hierarchical_gt=str(row.get('hierarchical_disease_label', '')),
                caption=caption,
                experiment_name=experiment_name,
                sam_strategy=sam_strategy,
                use_symptoms=use_symptoms,
                symptom_source=symptom_source
            )

            results.append(result)

            if self.verbose:
                status = "✓" if result.exact_match else "✗"
                self._log(f"  [{status}] {filename}: GT={result.ground_truth} | Pred={result.prediction}")

        self.results.extend(results)

        # 요약 계산
        summary = self._calculate_summary(results, experiment_name, sam_strategy, use_symptoms, symptom_source)
        self.summaries.append(summary)

        return results

    def _calculate_summary(
        self,
        results: List[ExperimentResult],
        experiment_name: str,
        sam_strategy: str,
        use_symptoms: bool,
        symptom_source: str
    ) -> ExperimentSummary:
        """실험 결과 요약 계산"""

        summary = ExperimentSummary(
            experiment_name=experiment_name,
            sam_strategy=sam_strategy,
            use_symptoms=use_symptoms,
            symptom_source=symptom_source,
            total_samples=len(results)
        )

        if not results:
            return summary

        valid_results = [r for r in results if r.prediction != "no definitive diagnosis"]
        summary.valid_samples = len(valid_results)

        # Exact Match
        exact_matches = sum(1 for r in results if r.exact_match)
        summary.exact_match_accuracy = exact_matches / len(results) if results else 0.0

        # Hierarchical Distance
        distances = [r.hierarchical_distance for r in results if r.hierarchical_distance >= 0]
        summary.avg_hierarchical_distance = np.mean(distances) if distances else -1

        # Hierarchical F1
        h_f1_scores = [r.hierarchical_f1 for r in results if r.hierarchical_f1 > 0]
        summary.hierarchical_f1 = np.mean(h_f1_scores) if h_f1_scores else 0.0

        # Partial Credit
        pc_scores = [r.partial_credit for r in results if r.partial_credit > 0]
        summary.partial_credit = np.mean(pc_scores) if pc_scores else 0.0

        # Segmentation Score
        seg_scores = [r.segmentation_score for r in results if r.segmentation_score > 0]
        summary.avg_segmentation_score = np.mean(seg_scores) if seg_scores else 0.0

        # Confidence
        confidences = [r.confidence for r in results]
        summary.avg_confidence = np.mean(confidences) if confidences else 0.0

        return summary

    def save_results(self, output_dir: str):
        """결과를 CSV로 저장"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 1. 상세 결과 CSV
        detailed_path = output_path / f"detailed_results_{timestamp}.csv"
        detailed_df = pd.DataFrame([r.to_dict() for r in self.results])
        detailed_df.to_csv(detailed_path, index=False, encoding='utf-8-sig')
        self._log(f"상세 결과 저장: {detailed_path}")

        # 2. 실험 요약 CSV
        summary_path = output_path / f"experiment_summary_{timestamp}.csv"
        summary_df = pd.DataFrame([s.to_dict() for s in self.summaries])
        summary_df.to_csv(summary_path, index=False, encoding='utf-8-sig')
        self._log(f"실험 요약 저장: {summary_path}")

        # 3. SAM 전략별 비교 CSV
        sam_comparison_path = output_path / f"sam_strategy_comparison_{timestamp}.csv"
        sam_comparison = self._create_sam_comparison_df()
        sam_comparison.to_csv(sam_comparison_path, index=False, encoding='utf-8-sig')
        self._log(f"SAM 전략 비교 저장: {sam_comparison_path}")

        # 4. 전체 비교 CSV (LLM 문서화용)
        comparison_path = output_path / f"full_comparison_{timestamp}.csv"
        comparison_df = self._create_full_comparison_df()
        comparison_df.to_csv(comparison_path, index=False, encoding='utf-8-sig')
        self._log(f"전체 비교 저장: {comparison_path}")

        # 5. 실험 설정 JSON
        config_path = output_path / f"experiment_config_{timestamp}.json"
        config = {
            "timestamp": timestamp,
            "total_experiments": len(self.summaries),
            "total_samples": len(self.results),
            "experiments": [s.experiment_name for s in self.summaries]
        }
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        self._log(f"실험 설정 저장: {config_path}")

        return {
            "detailed": str(detailed_path),
            "summary": str(summary_path),
            "sam_comparison": str(sam_comparison_path),
            "full_comparison": str(comparison_path),
            "config": str(config_path)
        }

    def _create_sam_comparison_df(self) -> pd.DataFrame:
        """SAM 전략별 비교 DataFrame 생성"""
        rows = []

        for summary in self.summaries:
            rows.append({
                "SAM Strategy": summary.sam_strategy,
                "Use Symptoms": summary.use_symptoms,
                "Symptom Source": summary.symptom_source,
                "Exact Match (%)": f"{summary.exact_match_accuracy * 100:.2f}",
                "Hierarchical F1": f"{summary.hierarchical_f1:.4f}",
                "Partial Credit": f"{summary.partial_credit:.4f}",
                "Avg H-Distance": f"{summary.avg_hierarchical_distance:.2f}",
                "Avg Seg Score": f"{summary.avg_segmentation_score:.3f}",
                "Avg Confidence": f"{summary.avg_confidence:.3f}",
                "Valid Samples": summary.valid_samples,
                "Total Samples": summary.total_samples
            })

        return pd.DataFrame(rows)

    def _create_full_comparison_df(self) -> pd.DataFrame:
        """전체 비교 DataFrame 생성 (LLM 문서화용)"""

        # 실험별로 그룹화
        experiment_groups = {}
        for result in self.results:
            key = result.experiment_name
            if key not in experiment_groups:
                experiment_groups[key] = []
            experiment_groups[key].append(result)

        rows = []

        # 샘플별로 모든 실험 결과 병합
        sample_ids = sorted(set(r.sample_id for r in self.results))

        for sample_id in sample_ids:
            row = {"sample_id": sample_id}

            # 첫 번째 결과에서 기본 정보 가져오기
            first_result = next((r for r in self.results if r.sample_id == sample_id), None)
            if first_result:
                row["filename"] = first_result.filename
                row["ground_truth"] = first_result.ground_truth
                row["hierarchical_gt"] = first_result.hierarchical_gt

            # 각 실험 결과 추가
            for exp_name, results in experiment_groups.items():
                result = next((r for r in results if r.sample_id == sample_id), None)
                if result:
                    short_name = exp_name.replace("_", "")[:15]
                    row[f"{short_name}_pred"] = result.prediction
                    row[f"{short_name}_exact"] = 1 if result.exact_match else 0
                    row[f"{short_name}_hf1"] = f"{result.hierarchical_f1:.3f}"
                    row[f"{short_name}_conf"] = f"{result.confidence:.2f}"

            rows.append(row)

        return pd.DataFrame(rows)

    def print_summary(self):
        """실험 요약 출력"""
        print("\n" + "=" * 80)
        print("실험 결과 요약")
        print("=" * 80)

        for summary in self.summaries:
            print(f"\n[{summary.experiment_name}]")
            print(f"  SAM 전략: {summary.sam_strategy}")
            print(f"  증상 사용: {summary.use_symptoms} ({summary.symptom_source})")
            print(f"  샘플 수: {summary.valid_samples}/{summary.total_samples}")
            print(f"  Exact Match: {summary.exact_match_accuracy * 100:.2f}%")
            print(f"  Hierarchical F1: {summary.hierarchical_f1:.4f}")
            print(f"  Partial Credit: {summary.partial_credit:.4f}")
            print(f"  Avg H-Distance: {summary.avg_hierarchical_distance:.2f}")
            if summary.avg_segmentation_score > 0:
                print(f"  Avg Seg Score: {summary.avg_segmentation_score:.3f}")

        print("\n" + "=" * 80)


# ============ 기본 경로 설정 ============

def get_default_paths():
    """기본 경로 자동 탐지"""
    # CSV 경로 후보
    csv_candidates = [
        PROJECT_ROOT / "dataset" / "Derm1M" / "Derm1M_v2_pretrain_ontology_sampled_100.csv",
        PROJECT_ROOT / "dataset" / "random100.csv",
        Path("/content/DermAgent-sam-real2/dataset/Derm1M/Derm1M_v2_pretrain_ontology_sampled_100.csv"),  # Colab
        Path("/content/dataset/Derm1M/Derm1M_v2_pretrain_ontology_sampled_100.csv"),  # Colab alt
    ]

    # 이미지 디렉터리 후보
    image_candidates = [
        PROJECT_ROOT / "dataset" / "Derm1M" / "images",
        PROJECT_ROOT / "images",
        Path("/content/drive/MyDrive/DermAgent_Data/images"),  # Colab Google Drive
        Path("/content/images"),  # Colab
    ]

    default_csv = None
    for path in csv_candidates:
        if path.exists():
            default_csv = str(path)
            break

    default_image_dir = None
    for path in image_candidates:
        if path.exists():
            default_image_dir = str(path)
            break

    return default_csv, default_image_dir


# ============ CLI ============

def main():
    # 기본 경로 탐지
    default_csv, default_image_dir = get_default_paths()

    parser = argparse.ArgumentParser(
        description="증상 분석 통합 실험",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
사용 예시:
  # 가장 간단한 실행 (기본 경로 자동 탐지)
  python run_symptom_integration_experiment.py --image_dir /path/to/images

  # Colab에서 실행
  !python run_symptom_integration_experiment.py \\
      --image_dir /content/drive/MyDrive/images \\
      --num_samples 10

  # 전체 옵션 지정
  python run_symptom_integration_experiment.py \\
      --input_csv /path/to/data.csv \\
      --image_dir /path/to/images \\
      --output_dir ./outputs \\
      --sam_strategies none,center \\
      --num_samples 20

  # 증상만 테스트 (SAM 없이)
  python run_symptom_integration_experiment.py \\
      --image_dir /path/to/images \\
      --sam_strategies none \\
      --skip_no_symptom

감지된 기본 경로:
  CSV: {default_csv or '(없음 - --input_csv 필수)'}
  이미지: {default_image_dir or '(없음 - --image_dir 필수)'}
        """
    )

    parser.add_argument('--input_csv', type=str, default=default_csv,
                        help=f'입력 CSV 파일 경로 (기본: {default_csv})')
    parser.add_argument('--image_dir', type=str, default=default_image_dir,
                        help=f'이미지 디렉터리 (기본: {default_image_dir})')
    parser.add_argument('--output_dir', type=str, default='./outputs', help='출력 디렉터리')
    parser.add_argument('--api_key', type=str, default=None, help='OpenAI API 키')
    parser.add_argument('--num_samples', type=int, default=None, help='샘플 수 제한')
    parser.add_argument('--sam_strategies', type=str, default='none,center,lesion_features,both',
                        help='SAM 전략 (쉼표 구분)')
    parser.add_argument('--skip_no_symptom', action='store_true', help='증상 없는 실험 건너뛰기')
    parser.add_argument('--skip_no_sam', action='store_true', help='SAM 없는 실험 건너뛰기')
    parser.add_argument('--verbose', action='store_true', help='상세 로그')

    args = parser.parse_args()

    # 필수 경로 검사
    if not args.input_csv:
        print("오류: --input_csv가 필요합니다.")
        print("  기본 CSV 파일을 찾을 수 없습니다.")
        print("  dataset/Derm1M/Derm1M_v2_pretrain_ontology_sampled_100.csv 파일이 있는지 확인하세요.")
        return

    if not args.image_dir:
        print("오류: --image_dir가 필요합니다.")
        print("  이미지 디렉터리를 지정하세요.")
        return

    if not os.path.exists(args.input_csv):
        print(f"오류: CSV 파일을 찾을 수 없습니다: {args.input_csv}")
        return

    if not os.path.exists(args.image_dir):
        print(f"오류: 이미지 디렉터리를 찾을 수 없습니다: {args.image_dir}")
        print("  Google Drive를 마운트했는지 확인하세요.")
        return

    # API 키 설정
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("오류: OPENAI_API_KEY가 필요합니다.")
        print("  --api_key 옵션 또는 환경 변수로 설정하세요.")
        print("  Colab에서는: os.environ['OPENAI_API_KEY'] = 'sk-...'")
        return

    # 설정 출력
    print("=" * 60)
    print("증상 통합 실험 설정")
    print("=" * 60)
    print(f"  CSV: {args.input_csv}")
    print(f"  이미지 디렉터리: {args.image_dir}")
    print(f"  출력 디렉터리: {args.output_dir}")
    print(f"  SAM 전략: {args.sam_strategies}")
    print(f"  샘플 수: {args.num_samples or '전체'}")
    print("=" * 60)

    # VLM 초기화
    try:
        from vlm_wrapper import GPT4oWrapper
        vlm = GPT4oWrapper(api_key=api_key, use_labels_prompt=False)
        print("[INFO] VLM (GPT-4o) 초기화 완료")
    except ImportError:
        print("오류: vlm_wrapper 모듈을 찾을 수 없습니다.")
        return

    # 데이터 로드
    print(f"[INFO] 데이터 로드: {args.input_csv}")
    df = pd.read_csv(args.input_csv)
    print(f"[INFO] 총 {len(df)}개 샘플")

    # caption 컬럼 확인
    if 'caption' not in df.columns:
        print("경고: 'caption' 컬럼이 없습니다. 증상 추출이 제한될 수 있습니다.")
    else:
        non_empty_captions = df['caption'].notna().sum()
        print(f"[INFO] caption 데이터: {non_empty_captions}/{len(df)}개")

    if args.num_samples:
        df = df.head(args.num_samples)
        print(f"[INFO] {args.num_samples}개 샘플로 제한")

    # 실험 초기화
    experiment = SymptomIntegrationExperiment(
        vlm_model=vlm,
        verbose=args.verbose
    )

    # SAM 전략 파싱
    sam_strategies = [s.strip() for s in args.sam_strategies.split(',')]

    # 실험 실행
    experiments_to_run = []

    for sam_strategy in sam_strategies:
        # SAM 없는 실험 건너뛰기 옵션
        if args.skip_no_sam and sam_strategy == "none":
            continue

        # 증상 없는 실험
        if not args.skip_no_symptom:
            experiments_to_run.append({
                "name": f"sam_{sam_strategy}_no_symptom",
                "sam_strategy": sam_strategy,
                "use_symptoms": False,
                "symptom_source": ""
            })

        # 증상 있는 실험 (caption)
        experiments_to_run.append({
            "name": f"sam_{sam_strategy}_symptom_caption",
            "sam_strategy": sam_strategy,
            "use_symptoms": True,
            "symptom_source": "caption"
        })

        # 증상 있는 실험 (truncated_caption)
        experiments_to_run.append({
            "name": f"sam_{sam_strategy}_symptom_truncated",
            "sam_strategy": sam_strategy,
            "use_symptoms": True,
            "symptom_source": "truncated_caption"
        })

    print(f"\n[INFO] 총 {len(experiments_to_run)}개 실험 예정")

    for exp_config in experiments_to_run:
        experiment.run_experiment_batch(
            df=df,
            image_base_dir=args.image_dir,
            experiment_name=exp_config["name"],
            sam_strategy=exp_config["sam_strategy"],
            use_symptoms=exp_config["use_symptoms"],
            symptom_source=exp_config["symptom_source"],
            num_samples=args.num_samples
        )

    # 결과 요약 출력
    experiment.print_summary()

    # 결과 저장
    saved_files = experiment.save_results(args.output_dir)

    print("\n[INFO] 저장된 파일:")
    for name, path in saved_files.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
