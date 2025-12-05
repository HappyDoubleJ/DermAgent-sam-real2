"""
Dermatology Diagnosis Agent Framework with SAM Integration

SAM(Segment Anything Model)을 초반에 사용하여 병변 영역을 세그멘테이션하고,
온톨로지 기반 계층적 탐색과 도구 기반 추론을 활용한 피부과 진단 에이전트

파이프라인:
Step 0: SAM 세그멘테이션 - 병변 영역 분리
Step 1: 초기 평가 - 세그멘테이션된 영역 기반 특징 추출
Step 2: 대분류 - 루트 카테고리 선택
Step 3: 중분류/소분류 - 하위 카테고리 탐색
Step 4: 감별 진단 - 후보 질환들 비교
Step 5: 최종 진단
"""

import json
import re
import sys
import os
import base64
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
from io import BytesIO
import numpy as np
from PIL import Image

# 경로 설정
SCRIPT_DIR = Path(__file__).parent
AGENT_DIR = SCRIPT_DIR.parent  # DermAgent
DERM1M_EXP_DIR = AGENT_DIR.parent  # derm1m_exp
sys.path.insert(0, str(DERM1M_EXP_DIR / "eval"))
sys.path.insert(0, str(DERM1M_EXP_DIR / "SA-project-SAM"))
sys.path.insert(0, str(DERM1M_EXP_DIR / "experiments"))

from ontology_utils import OntologyTree

# SAM 모듈 import
SAM_AVAILABLE = False
try:
    from sam_masking import (
        SAMSegmenter,
        SAM2Segmenter,
        MedSAM2Segmenter,
        apply_mask_to_image,
        crop_masked_region,
        load_image,
        save_image
    )
    SAM_AVAILABLE = True
except ImportError as e:
    print(f"[SAM Agent] SAM 모듈 임포트 실패: {e}")

# LLM-Guided Pipeline import
LLM_GUIDED_AVAILABLE = False
try:
    from llm_guided_pipeline import LLMGuidedSegmenter
    LLM_GUIDED_AVAILABLE = True
except ImportError:
    pass


class DiagnosisStep(Enum):
    """진단 단계"""
    SEGMENTATION = "segmentation"  # SAM 세그멘테이션
    INITIAL_ASSESSMENT = "initial_assessment"
    CATEGORY_CLASSIFICATION = "category_classification"
    SUBCATEGORY_CLASSIFICATION = "subcategory_classification"
    DIFFERENTIAL_DIAGNOSIS = "differential_diagnosis"
    FINAL_DIAGNOSIS = "final_diagnosis"


# 신뢰도 임계값 상수
MIN_CATEGORY_CONFIDENCE = 0.4
MIN_SUBCATEGORY_CONFIDENCE = 0.3
MIN_DIFFERENTIAL_CONFIDENCE = 0.3
FALLBACK_CONFIDENCE_FLAG = 0.25


@dataclass
class SegmentationResult:
    """세그멘테이션 결과"""
    mask: Optional[np.ndarray] = None
    score: float = 0.0
    method: str = ""  # center, lesion_features, llm_guided
    overlay_image: Optional[np.ndarray] = None
    cropped_image: Optional[np.ndarray] = None
    lesion_count: int = 0
    bbox: Optional[Tuple[int, int, int, int]] = None  # (x1, y1, x2, y2)


@dataclass
class ObservationResult:
    """관찰 결과"""
    morphology: List[str] = field(default_factory=list)
    color: List[str] = field(default_factory=list)
    distribution: List[str] = field(default_factory=list)
    location: str = ""
    surface: List[str] = field(default_factory=list)
    symptoms: List[str] = field(default_factory=list)
    raw_description: str = ""


@dataclass
class DiagnosisState:
    """진단 상태 추적"""
    current_step: DiagnosisStep = DiagnosisStep.SEGMENTATION
    current_path: List[str] = field(default_factory=list)
    candidates: List[str] = field(default_factory=list)
    confidence_scores: Dict[str, float] = field(default_factory=dict)
    observations: Optional[ObservationResult] = None
    reasoning_history: List[Dict] = field(default_factory=list)
    final_diagnosis: List[str] = field(default_factory=list)

    # 세그멘테이션 결과
    segmentation: Optional[SegmentationResult] = None

    # 오류 추적 필드
    errors: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[Dict[str, Any]] = field(default_factory=list)
    has_fallback: bool = False
    vlm_failures: int = 0
    low_confidence_steps: List[str] = field(default_factory=list)

    # Backtracking 필드
    explored_paths: set = field(default_factory=set)
    backtrack_count: int = 0
    max_backtracks: int = 3
    backtrack_history: List[Dict[str, Any]] = field(default_factory=list)


class BaseTool(ABC):
    """도구 기본 클래스"""

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        pass

    @abstractmethod
    def execute(self, *args, **kwargs) -> Any:
        pass


class OntologyNavigator(BaseTool):
    """온톨로지 탐색 도구"""

    def __init__(self, tree: OntologyTree):
        self.tree = tree

    @property
    def name(self) -> str:
        return "ontology_navigator"

    @property
    def description(self) -> str:
        return "Navigate the disease ontology tree to find relevant categories and diseases"

    def execute(self, action: str, node: str = "root") -> Dict:
        if action == "get_children":
            if node == "root":
                children = self.tree.ontology.get("root", [])
            else:
                children = self.tree.get_children(node)
            return {"node": node, "children": children, "count": len(children)}

        elif action == "get_path":
            path = self.tree.get_path_to_root(node)
            return {"node": node, "path": path, "depth": len(path) - 1}

        elif action == "get_siblings":
            siblings = self.tree.get_siblings(node)
            return {"node": node, "siblings": siblings}

        elif action == "get_descendants":
            descendants = list(self.tree.get_all_descendants(node))
            return {"node": node, "descendants": descendants[:50], "total": len(descendants)}

        elif action == "validate":
            is_valid = self.tree.is_valid_node(node)
            canonical = self.tree.get_canonical_name(node) if is_valid else None
            return {"node": node, "valid": is_valid, "canonical_name": canonical}

        else:
            return {"error": f"Unknown action: {action}"}


class DifferentialDiagnosisTool(BaseTool):
    """VLM 기반 동적 감별 진단 도구"""

    def __init__(self, tree: OntologyTree, vlm_model=None, system_instruction: str = ""):
        self.tree = tree
        self.vlm = vlm_model
        self.system_instruction = system_instruction.strip()

    @property
    def name(self) -> str:
        return "differential_diagnosis"

    @property
    def description(self) -> str:
        return "Compare clinical features with candidate diseases using VLM-based dynamic comparison"

    def execute(
        self,
        candidates: List[str],
        observations: ObservationResult,
        image_path: str = None,
        overlay_image: np.ndarray = None
    ) -> Dict[str, float]:
        """VLM을 사용하여 후보 질환들과 관찰 결과를 비교"""
        if self.vlm is None or image_path is None:
            raise RuntimeError("VLM 모델과 이미지 경로가 모두 필요합니다.")

        scores = self._compare_with_vlm_batch(candidates, observations, image_path, overlay_image)
        return dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))

    def _compare_with_vlm_batch(
        self,
        candidates: List[str],
        observations: ObservationResult,
        image_path: str,
        overlay_image: np.ndarray = None
    ) -> Dict[str, float]:
        if not candidates:
            return {}

        candidates_list = "\n".join([f"{i+1}. {c}" for i, c in enumerate(candidates)])

        obs_text = f"""
Morphology: {', '.join(observations.morphology) if observations.morphology else 'not specified'}
Color: {', '.join(observations.color) if observations.color else 'not specified'}
Distribution: {', '.join(observations.distribution) if observations.distribution else 'not specified'}
Surface: {', '.join(observations.surface) if observations.surface else 'not specified'}
Location: {observations.location if observations.location else 'not specified'}
"""

        focused_instruction = (
            "You are a dermatology expert. Compare this skin lesion with the candidate diagnoses listed below. "
            "The lesion area is highlighted with a red overlay. "
            "Only consider the diseases in the provided list."
        )
        prompt = f"""{focused_instruction}

Compare this skin lesion with the following candidate diagnoses and rate each one.

Candidate Diagnoses:
{candidates_list}

Observed Clinical Features:
{obs_text}

For EACH candidate diagnosis, evaluate:
1. How well do the observed features match the typical presentation of this disease?
2. What features support this diagnosis?
3. What features contradict this diagnosis?
4. Overall likelihood score (0-10)

Respond in JSON format:
{{
    "comparisons": [
        {{
            "disease": "exact disease name from the list",
            "likelihood_score": 0-10,
            "supporting_features": ["feature1", "feature2"],
            "contradicting_features": ["feature1", "feature2"],
            "brief_reasoning": "one sentence explanation"
        }},
        ... (one entry for each candidate)
    ]
}}

IMPORTANT: Include ALL {len(candidates)} candidates in your response. Provide ONLY the JSON output."""

        try:
            response = self.vlm.chat_img(prompt, [image_path], max_tokens=2048)
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                parsed = json.loads(json_match.group())
                comparisons = parsed.get("comparisons", [])

                scores = {}
                for comp in comparisons:
                    if not isinstance(comp, dict):
                        continue
                    disease = comp.get("disease", "").strip()
                    if not disease:
                        continue
                    likelihood = comp.get("likelihood_score", 5)
                    try:
                        scores[disease] = float(likelihood) / 10.0
                    except (TypeError, ValueError):
                        scores[disease] = 0.5

                for candidate in candidates:
                    if candidate not in scores:
                        scores[candidate] = 0.5

                return scores
            else:
                return {candidate: 0.5 for candidate in candidates}

        except Exception:
            return {candidate: 0.5 for candidate in candidates}


def encode_image_to_base64(image: np.ndarray) -> str:
    """numpy 배열을 base64로 인코딩"""
    pil_image = Image.fromarray(image)
    buffer = BytesIO()
    pil_image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def load_image_as_array(image_path: str) -> np.ndarray:
    """이미지 파일을 numpy 배열로 로드"""
    if SAM_AVAILABLE:
        return load_image(image_path)
    else:
        img = Image.open(image_path).convert('RGB')
        return np.array(img)


class DermatologyAgentSAM:
    """SAM 통합 피부과 진단 에이전트"""

    def __init__(
        self,
        ontology_path: Optional[str] = None,
        vlm_model: Any = None,
        segmenter_type: str = "sam",
        segmentation_strategy: str = "center",
        checkpoint_dir: Optional[str] = None,
        use_segmentation: bool = True,
        verbose: bool = True
    ):
        """
        초기화

        Args:
            ontology_path: 온톨로지 JSON 파일 경로
            vlm_model: Vision-Language Model
            segmenter_type: 세그멘터 종류 (sam, sam2, medsam2)
            segmentation_strategy: 세그멘테이션 전략 (center, lesion_features, both, llm_guided)
            checkpoint_dir: SAM 체크포인트 디렉터리
            use_segmentation: 세그멘테이션 사용 여부
            verbose: 상세 로그 출력
        """
        self.tree = OntologyTree(ontology_path)
        if vlm_model is None:
            raise ValueError("vlm_model을 지정해야 합니다.")
        self.vlm = vlm_model
        self.verbose = verbose
        self.use_segmentation = use_segmentation and SAM_AVAILABLE
        self.segmenter_type = segmenter_type
        self.segmentation_strategy = segmentation_strategy

        # SAM 세그멘터 초기화
        self.segmenter = None
        self.llm_guided = None
        if self.use_segmentation:
            if checkpoint_dir is None:
                checkpoint_dir = str(DERM1M_EXP_DIR / "SA-project-SAM" / "checkpoints")
            self.segmenter = self._init_segmenter(segmenter_type, checkpoint_dir)

            # LLM-Guided (선택적)
            if segmentation_strategy == "llm_guided" and LLM_GUIDED_AVAILABLE:
                api_key = os.environ.get("OPENAI_API_KEY")
                if api_key:
                    self.llm_guided = LLMGuidedSegmenter(api_key=api_key)

        # 도구 초기화
        self.valid_diseases = sorted(list(self.tree.valid_nodes))
        self.leaf_diseases = sorted([n for n in self.tree.valid_nodes if not self.tree.get_children(n)])
        self.system_instruction = self._build_system_instruction()
        self.tools = {
            "navigator": OntologyNavigator(self.tree),
            "differential": DifferentialDiagnosisTool(self.tree, self.vlm, self.system_instruction),
        }

        # 루트 카테고리
        self.root_categories = self.tree.ontology.get("root", [])

        # 프롬프트 템플릿
        self._load_prompts()

        if self.verbose:
            self._log(f"SAM Agent initialized")
            self._log(f"  Segmentation: {'Enabled' if self.use_segmentation else 'Disabled'}")
            if self.use_segmentation:
                self._log(f"  Segmenter: {segmenter_type.upper()}")
                self._log(f"  Strategy: {segmentation_strategy}")

    def _init_segmenter(self, segmenter_type: str, checkpoint_dir: str):
        """세그멘터 초기화"""
        if not SAM_AVAILABLE:
            return None

        segmenter_type = segmenter_type.lower()

        try:
            if segmenter_type == "sam":
                return SAMSegmenter(checkpoint_dir=checkpoint_dir)
            elif segmenter_type == "sam2":
                return SAM2Segmenter(checkpoint_dir=checkpoint_dir)
            elif segmenter_type == "medsam2":
                segmenter = MedSAM2Segmenter(checkpoint_dir=checkpoint_dir)
                if not segmenter.is_available():
                    self._log("MedSAM2 unavailable, falling back to SAM")
                    return SAMSegmenter(checkpoint_dir=checkpoint_dir)
                return segmenter
            else:
                return SAMSegmenter(checkpoint_dir=checkpoint_dir)
        except Exception as e:
            self._log(f"Segmenter initialization failed: {e}")
            return None

    def _build_system_instruction(self) -> str:
        return (
            "You are a board-certified dermatology expert. You are provided with a skin image where "
            "the lesion area has been segmented and highlighted. "
            "Analyze the highlighted region carefully and provide a detailed, professional assessment. "
            "You have access to tools to navigate a disease ontology tree. "
            "Your diagnosis must be a valid node from the ontology tree. "
            "Choose the most specific level you can confidently identify. "
            "If uncertain, share the top differentials."
        )

    def _load_prompts(self):
        """프롬프트 템플릿 로드"""
        self.prompts = {
            "initial_assessment": """Analyze this dermatological image. The lesion area has been highlighted with a red overlay.

IMPORTANT: Focus on the HIGHLIGHTED (RED) REGION which shows the segmented lesion area.

If no clear skin lesion is visible in the highlighted region, respond with:
{{
    "morphology": ["no visible lesion"],
    "color": ["not observed"],
    "distribution": ["not observed"],
    "surface": ["not observed"],
    "border": ["not observed"],
    "location": "not observed",
    "additional_notes": "no definitive diagnosis",
    "segmentation_quality": "poor"
}}

Focus on the PRIMARY LESION within the highlighted area:
1. Morphology: macule, papule, plaque, nodule, vesicle, bulla, pustule, erosion, ulcer, etc.
2. Color: red, pink, brown, black, white, yellow, purple, blue, skin-colored, etc.
3. Distribution: localized, generalized, symmetric, asymmetric, clustered, linear, etc.
4. Surface features: smooth, scaly, crusted, rough, verrucous, ulcerated, etc.
5. Border: well-defined, ill-defined, regular, irregular, raised, rolled
6. Body location: face, trunk, extremities, hands, feet, scalp, etc.

Also evaluate the segmentation quality:
- "good": The overlay accurately captures the lesion boundaries
- "partial": The overlay captures some but not all of the lesion
- "poor": The overlay misses the lesion or captures mostly normal skin

Provide your observations in JSON format:
{{
    "morphology": ["list of PRIMARY lesion types"],
    "color": ["list of colors observed in the lesion"],
    "distribution": ["distribution patterns"],
    "surface": ["surface features"],
    "border": ["border characteristics"],
    "location": "body location",
    "additional_notes": "any other relevant observations",
    "segmentation_quality": "good/partial/poor"
}}

Provide ONLY the JSON output.""",

            "category_classification": """Based on the clinical features observed in the HIGHLIGHTED lesion area,
classify this condition into ONE of the following major categories:

Categories:
{categories}

Consider the morphology, distribution, and clinical presentation of the segmented lesion.
Respond with JSON:
{{
    "selected_category": "the most likely category",
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation"
}}

Provide ONLY the JSON output.""",

            "subcategory_classification": """Given that this skin condition belongs to the "{parent_category}" category,
further classify it into one of these subcategories:

Subcategories:
{subcategories}

Based on the image features of the highlighted lesion:
{observations}

Respond with JSON:
{{
    "selected_subcategory": "the most likely subcategory",
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation"
}}

Provide ONLY the JSON output.""",

            "final_diagnosis": """Based on the hierarchical classification path:
{path}

And the clinical observations from the segmented lesion:
{observations}

Select the most likely specific diagnosis from these candidates:
{candidates}

IMPORTANT: If no clear skin lesion is visible or you cannot make a confident diagnosis, set primary_diagnosis to "no definitive diagnosis".

Respond with JSON:
{{
    "primary_diagnosis": "most likely diagnosis (or 'no definitive diagnosis' if uncertain)",
    "confidence": 0.0-1.0,
    "differential_diagnoses": ["other possible diagnoses in order of likelihood"],
    "reasoning": "clinical reasoning for your diagnosis"
}}

Provide ONLY the JSON output."""
        }

    def _log(self, message: str):
        if self.verbose:
            print(f"[SAM-Agent] {message}")

    def _record_error(self, state: DiagnosisState, step: str, error_type: str, message: str, details: Dict = None):
        error_entry = {"step": step, "type": error_type, "message": message, "details": details or {}}
        state.errors.append(error_entry)
        self._log(f"ERROR [{step}]: {message}")

    def _record_warning(self, state: DiagnosisState, step: str, message: str, details: Dict = None):
        warning_entry = {"step": step, "message": message, "details": details or {}}
        state.warnings.append(warning_entry)
        self._log(f"WARNING [{step}]: {message}")

    def _check_confidence(self, state: DiagnosisState, step: str, confidence: float, threshold: float) -> bool:
        if confidence < threshold:
            state.low_confidence_steps.append(step)
            self._record_warning(state, step, f"Low confidence: {confidence:.2f} < {threshold:.2f}")
            return False
        return True

    def _parse_json_response(self, response) -> Dict:
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

    def _call_vlm(self, prompt: str, image_path: str, state: DiagnosisState = None, step: str = "unknown") -> Tuple[str, bool]:
        if self.vlm is None:
            if state:
                state.vlm_failures += 1
            raise RuntimeError("VLM 모델이 설정되어 있지 않습니다.")

        full_prompt = f"{self.system_instruction}\n\n{prompt}"

        try:
            response = self.vlm.chat_img(full_prompt, [image_path], max_tokens=1024)
            return response, True
        except Exception as e:
            if state:
                state.vlm_failures += 1
                self._record_error(state, step, "VLM_EXCEPTION", str(e))
            return "{}", False

    def _call_vlm_with_overlay(
        self,
        prompt: str,
        image_path: str,
        overlay_image: np.ndarray,
        state: DiagnosisState = None,
        step: str = "unknown"
    ) -> Tuple[str, bool]:
        """오버레이 이미지를 포함하여 VLM 호출"""
        if self.vlm is None:
            if state:
                state.vlm_failures += 1
            raise RuntimeError("VLM 모델이 설정되어 있지 않습니다.")

        full_prompt = f"{self.system_instruction}\n\n{prompt}"

        # 오버레이 이미지를 임시 파일로 저장
        import tempfile
        try:
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                pil_image = Image.fromarray(overlay_image)
                pil_image.save(tmp.name)
                overlay_path = tmp.name

            response = self.vlm.chat_img(full_prompt, [overlay_path], max_tokens=1024)

            # 임시 파일 삭제
            os.unlink(overlay_path)

            return response, True
        except Exception as e:
            if state:
                state.vlm_failures += 1
                self._record_error(state, step, "VLM_EXCEPTION", str(e))
            return "{}", False

    # ============ Step 0: SAM 세그멘테이션 ============

    def step_segmentation(self, image_path: str, state: DiagnosisState) -> DiagnosisState:
        """Step 0: SAM 세그멘테이션"""
        self._log("Step 0: SAM Segmentation")

        if not self.use_segmentation or self.segmenter is None:
            self._log("  Segmentation disabled or unavailable, skipping...")
            state.segmentation = SegmentationResult(method="none", score=0.0)
            state.current_step = DiagnosisStep.INITIAL_ASSESSMENT
            return state

        try:
            image = load_image_as_array(image_path)
        except Exception as e:
            self._record_error(state, "segmentation", "IMAGE_LOAD_FAILED", str(e))
            state.segmentation = SegmentationResult(method="error", score=0.0)
            state.current_step = DiagnosisStep.INITIAL_ASSESSMENT
            return state

        strategy = self.segmentation_strategy

        try:
            if strategy == "llm_guided" and self.llm_guided is not None:
                seg_result = self._segment_llm_guided(image)
            elif strategy == "center":
                seg_result = self.segmenter.segment_center_focused(image)
            elif strategy == "lesion_features":
                seg_result = self.segmenter.segment_lesion_features(image)
            elif strategy == "both":
                center_result = self.segmenter.segment_center_focused(image)
                feature_result = self.segmenter.segment_lesion_features(image)
                seg_result = feature_result if feature_result["score"] > center_result["score"] else center_result
            else:
                seg_result = self.segmenter.segment_center_focused(image)

            # 결과 저장
            mask = seg_result.get("mask")
            score = seg_result.get("score", 0.0)
            method = seg_result.get("method", strategy)

            if mask is not None:
                overlay = apply_mask_to_image(image, mask, color=(255, 0, 0), alpha=0.4)
                cropped = crop_masked_region(image, mask)

                state.segmentation = SegmentationResult(
                    mask=mask,
                    score=score,
                    method=method,
                    overlay_image=overlay,
                    cropped_image=cropped
                )
                self._log(f"  Segmentation successful: score={score:.3f}, method={method}")
            else:
                state.segmentation = SegmentationResult(method=method, score=0.0)
                self._log("  Segmentation failed: no mask generated")

        except Exception as e:
            self._record_error(state, "segmentation", "SEGMENTATION_FAILED", str(e))
            state.segmentation = SegmentationResult(method="error", score=0.0)

        state.reasoning_history.append({
            "step": "segmentation",
            "method": state.segmentation.method,
            "score": state.segmentation.score,
            "success": state.segmentation.mask is not None
        })

        state.current_step = DiagnosisStep.INITIAL_ASSESSMENT
        return state

    def _segment_llm_guided(self, image: np.ndarray) -> Dict:
        """LLM-Guided 세그멘테이션"""
        try:
            result = self.llm_guided.run_full_pipeline(
                image=image,
                segmenter=self.segmenter,
                save_results=False
            )

            seg_results = result.get("segmentation_results", [])
            if seg_results:
                best = max(seg_results, key=lambda x: x.get("score", 0))
                return {
                    "mask": best.get("mask"),
                    "score": best.get("score", 0.0),
                    "method": "llm_guided"
                }
            return {"mask": None, "score": 0.0, "method": "llm_guided"}

        except Exception:
            return {"mask": None, "score": 0.0, "method": "llm_guided_error"}

    # ============ Step 1: 초기 평가 ============

    def step_initial_assessment(self, image_path: str, state: DiagnosisState) -> DiagnosisState:
        """Step 1: 초기 평가 - 세그멘테이션 결과 기반"""
        self._log("Step 1: Initial Assessment (with segmentation)")

        prompt = self.prompts["initial_assessment"]

        # 세그멘테이션 오버레이가 있으면 사용
        if state.segmentation and state.segmentation.overlay_image is not None:
            response, success = self._call_vlm_with_overlay(
                prompt, image_path, state.segmentation.overlay_image, state, "initial_assessment"
            )
        else:
            response, success = self._call_vlm(prompt, image_path, state, "initial_assessment")

        parsed = self._parse_json_response(response)

        def _normalize_list(values):
            return values if values else ["not observed"]

        observations = ObservationResult(
            morphology=_normalize_list(parsed.get("morphology", [])),
            color=_normalize_list(parsed.get("color", [])),
            distribution=_normalize_list(parsed.get("distribution", [])),
            surface=_normalize_list(parsed.get("surface", [])),
            location=parsed.get("location", ""),
            raw_description=response
        )

        state.observations = observations
        state.reasoning_history.append({
            "step": "initial_assessment",
            "observations": parsed,
            "segmentation_quality": parsed.get("segmentation_quality", "unknown"),
            "raw_response": response[:500]
        })

        self._log(f"  Morphology: {observations.morphology}")
        self._log(f"  Color: {observations.color}")
        self._log(f"  Location: {observations.location}")
        self._log(f"  Segmentation quality: {parsed.get('segmentation_quality', 'unknown')}")

        if "no visible lesion" in [m.lower() for m in observations.morphology]:
            self._log("  No visible lesion detected - skipping to final diagnosis")
            state.final_diagnosis = ["no definitive diagnosis"]
            state.confidence_scores["no definitive diagnosis"] = 0.0
            state.current_step = DiagnosisStep.FINAL_DIAGNOSIS
            return state

        state.current_step = DiagnosisStep.CATEGORY_CLASSIFICATION
        return state

    # ============ Step 2-5: 기존 로직 (세그멘테이션 결과 활용) ============

    def step_category_classification(self, image_path: str, state: DiagnosisState) -> DiagnosisState:
        """Step 2: 대분류"""
        self._log("Step 2: Category Classification")

        categories_desc = "\n".join([f"- {cat}" for cat in self.root_categories])
        prompt = self.prompts["category_classification"].format(categories=categories_desc)

        # 세그멘테이션 오버레이 사용
        if state.segmentation and state.segmentation.overlay_image is not None:
            response, success = self._call_vlm_with_overlay(
                prompt, image_path, state.segmentation.overlay_image, state, "category_classification"
            )
        else:
            response, success = self._call_vlm(prompt, image_path, state, "category_classification")

        if not success:
            state.has_fallback = True
            state.final_diagnosis.append("no definitive diagnosis")
            state.confidence_scores["no definitive diagnosis"] = FALLBACK_CONFIDENCE_FLAG
            state.current_step = DiagnosisStep.FINAL_DIAGNOSIS
            return state

        parsed = self._parse_json_response(response)

        if not parsed:
            state.has_fallback = True
            state.final_diagnosis.append("no definitive diagnosis")
            state.confidence_scores["no definitive diagnosis"] = FALLBACK_CONFIDENCE_FLAG
            state.current_step = DiagnosisStep.FINAL_DIAGNOSIS
            return state

        selected = parsed.get("selected_category", "")
        confidence = parsed.get("confidence", 0.5)

        self._check_confidence(state, "category_classification", confidence, MIN_CATEGORY_CONFIDENCE)

        canonical = self.tree.get_canonical_name(selected)
        if canonical and canonical in self.root_categories:
            state.current_path.append(canonical)
            state.confidence_scores[canonical] = confidence
            self._log(f"  Selected: {canonical} (conf: {confidence:.2f})")
        else:
            self._record_warning(state, "category_classification", f"Invalid category: {selected}")
            state.has_fallback = True
            state.final_diagnosis.append("no definitive diagnosis")
            state.confidence_scores["no definitive diagnosis"] = FALLBACK_CONFIDENCE_FLAG
            state.current_step = DiagnosisStep.FINAL_DIAGNOSIS
            return state

        state.reasoning_history.append({
            "step": "category_classification",
            "selected": canonical,
            "confidence": confidence,
            "reasoning": parsed.get("reasoning", "")
        })

        state.current_step = DiagnosisStep.SUBCATEGORY_CLASSIFICATION
        return state

    def step_subcategory_classification(self, image_path: str, state: DiagnosisState, max_depth: int = 3) -> DiagnosisState:
        """Step 3: 중분류/소분류"""
        current_depth = len(state.current_path)

        while current_depth < max_depth:
            current_node = state.current_path[-1]
            children = self.tree.get_children(current_node)

            if not children:
                break

            self._log(f"Step 3.{current_depth}: Subcategory Classification")
            self._log(f"  Current: {current_node}")

            if len(children) > 1:
                subcategories_desc = "\n".join([f"- {child}" for child in children])
                obs_desc = json.dumps({
                    "morphology": state.observations.morphology if state.observations else [],
                    "color": state.observations.color if state.observations else [],
                    "location": state.observations.location if state.observations else ""
                }, indent=2)

                prompt = self.prompts["subcategory_classification"].format(
                    parent_category=current_node,
                    subcategories=subcategories_desc,
                    observations=obs_desc
                )

                if state.segmentation and state.segmentation.overlay_image is not None:
                    response, success = self._call_vlm_with_overlay(
                        prompt, image_path, state.segmentation.overlay_image, state, f"subcategory_level_{current_depth}"
                    )
                else:
                    response, success = self._call_vlm(prompt, image_path, state, f"subcategory_level_{current_depth}")

                parsed = self._parse_json_response(response)
                selected = parsed.get("selected_subcategory", "")
                confidence = parsed.get("confidence", 0.5)

                self._check_confidence(state, f"subcategory_level_{current_depth}", confidence, MIN_SUBCATEGORY_CONFIDENCE)

                canonical = self.tree.get_canonical_name(selected)
                if canonical and canonical in children:
                    state.current_path.append(canonical)
                    state.confidence_scores[canonical] = confidence
                else:
                    state.has_fallback = True
                    state.current_path.append(children[0])
                    state.confidence_scores[children[0]] = FALLBACK_CONFIDENCE_FLAG

                self._log(f"  Selected: {state.current_path[-1]} (conf: {state.confidence_scores.get(state.current_path[-1], confidence):.2f})")
            else:
                if children:
                    state.current_path.append(children[0])
                    state.confidence_scores[children[0]] = 0.9
                else:
                    break

            current_depth = len(state.current_path)

        state.current_step = DiagnosisStep.DIFFERENTIAL_DIAGNOSIS
        return state

    def step_differential_diagnosis(self, image_path: str, state: DiagnosisState) -> DiagnosisState:
        """Step 4: 감별 진단"""
        self._log("Step 4: Differential Diagnosis")

        if not state.current_path:
            self._record_error(state, "differential_diagnosis", "EMPTY_PATH", "Current path is empty")
            state.current_step = DiagnosisStep.FINAL_DIAGNOSIS
            return state

        current_node = state.current_path[-1]
        descendants = self.tree.get_all_descendants(current_node)

        all_candidates = set(descendants)
        all_candidates.update(state.current_path)
        all_candidates.add(current_node)

        if not all_candidates:
            all_candidates = {current_node}

        state.candidates = sorted(all_candidates)[:20]

        self._log(f"  Candidates: {len(state.candidates)} diseases")

        if state.observations:
            diff_scores = self.tools["differential"].execute(
                state.candidates,
                state.observations,
                image_path,
                state.segmentation.overlay_image if state.segmentation else None
            )
            state.confidence_scores.update(diff_scores)

            top_scores = sorted(diff_scores.values(), reverse=True)[:3]
            if top_scores:
                self._log(f"  Top 3 scores: {[f'{s:.2f}' for s in top_scores]}")

        state.current_step = DiagnosisStep.FINAL_DIAGNOSIS
        return state

    def step_final_diagnosis(self, image_path: str, state: DiagnosisState) -> DiagnosisState:
        """Step 5: 최종 진단"""
        self._log("Step 5: Final Diagnosis")

        path_str = " → ".join(state.current_path)
        candidates_str = "\n".join([f"- {c}" for c in state.candidates[:15]])
        obs_str = json.dumps({
            "morphology": state.observations.morphology if state.observations else [],
            "color": state.observations.color if state.observations else [],
            "distribution": state.observations.distribution if state.observations else [],
            "location": state.observations.location if state.observations else ""
        }, indent=2)

        prompt = self.prompts["final_diagnosis"].format(
            path=path_str,
            observations=obs_str,
            candidates=candidates_str
        )

        if state.segmentation and state.segmentation.overlay_image is not None:
            response, success = self._call_vlm_with_overlay(
                prompt, image_path, state.segmentation.overlay_image, state, "final_diagnosis"
            )
        else:
            response, success = self._call_vlm(prompt, image_path, state, "final_diagnosis")

        parsed = self._parse_json_response(response)

        primary = parsed.get("primary_diagnosis", "")
        differentials = parsed.get("differential_diagnoses", [])
        confidence = parsed.get("confidence", 0.5)

        if primary and primary.lower() == "no definitive diagnosis":
            state.final_diagnosis.append("no definitive diagnosis")
            state.confidence_scores["no definitive diagnosis"] = confidence
        else:
            canonical_primary = self.tree.get_canonical_name(primary)
            if canonical_primary:
                state.final_diagnosis.append(canonical_primary)
                state.confidence_scores[canonical_primary] = confidence

            for diff in differentials[:3]:
                canonical = self.tree.get_canonical_name(diff)
                if canonical and canonical not in state.final_diagnosis:
                    state.final_diagnosis.append(canonical)

            if not state.final_diagnosis:
                sorted_candidates = sorted(state.confidence_scores.items(), key=lambda x: x[1], reverse=True)
                for candidate, score in sorted_candidates[:3]:
                    if self.tree.is_valid_node(candidate):
                        canonical = self.tree.get_canonical_name(candidate)
                        if canonical and canonical not in state.final_diagnosis:
                            state.final_diagnosis.append(canonical)

            if not state.final_diagnosis:
                state.final_diagnosis.append("no definitive diagnosis")

        state.reasoning_history.append({
            "step": "final_diagnosis",
            "primary": primary,
            "differentials": differentials,
            "confidence": confidence,
            "reasoning": parsed.get("reasoning", "")
        })

        self._log(f"  Final diagnosis: {state.final_diagnosis}")
        self._log(f"  Confidence: {confidence:.2f}")

        return state

    # ============ 메인 진단 메서드 ============

    def diagnose(self, image_path: str, max_depth: int = 4) -> Dict[str, Any]:
        """전체 진단 파이프라인 실행"""
        self._log(f"\n{'='*50}")
        self._log(f"Starting SAM-enhanced diagnosis for: {image_path}")
        self._log(f"{'='*50}")

        state = DiagnosisState()

        # Step 0: SAM 세그멘테이션
        state = self.step_segmentation(image_path, state)

        # Step 1: 초기 평가
        state = self.step_initial_assessment(image_path, state)

        if state.current_step == DiagnosisStep.FINAL_DIAGNOSIS and "no definitive diagnosis" in state.final_diagnosis:
            self._log("No visible lesion detected - diagnosis complete")
        else:
            # Step 2: 대분류
            state = self.step_category_classification(image_path, state)

            if state.current_step != DiagnosisStep.FINAL_DIAGNOSIS:
                # Step 3: 중분류/소분류
                state = self.step_subcategory_classification(image_path, state, max_depth)

                # Step 4: 감별 진단
                state = self.step_differential_diagnosis(image_path, state)

            # Step 5: 최종 진단
            if "no definitive diagnosis" not in state.final_diagnosis:
                state = self.step_final_diagnosis(image_path, state)

        # 결과 정리
        result = {
            "image_path": image_path,
            "final_diagnosis": state.final_diagnosis,
            "diagnosis_path": state.current_path,
            "confidence_scores": state.confidence_scores,
            "observations": {
                "morphology": state.observations.morphology if state.observations else [],
                "color": state.observations.color if state.observations else [],
                "distribution": state.observations.distribution if state.observations else [],
                "location": state.observations.location if state.observations else "",
            },
            "segmentation": {
                "method": state.segmentation.method if state.segmentation else "none",
                "score": state.segmentation.score if state.segmentation else 0.0,
                "success": state.segmentation.mask is not None if state.segmentation else False
            },
            "reasoning_history": state.reasoning_history,
            "candidates_considered": state.candidates,
            "errors": state.errors,
            "warnings": state.warnings,
            "has_fallback": state.has_fallback,
            "vlm_failures": state.vlm_failures,
            "low_confidence_steps": state.low_confidence_steps,
        }

        self._log(f"\n{'='*50}")
        self._log(f"Diagnosis complete")
        self._log(f"Result: {state.final_diagnosis}")
        self._log(f"Path: {' → '.join(state.current_path)}")
        self._log(f"{'='*50}\n")

        return result

    def diagnose_batch(self, image_paths: List[str], max_depth: int = 4) -> List[Dict[str, Any]]:
        """배치 진단"""
        results = []
        for path in image_paths:
            result = self.diagnose(path, max_depth)
            results.append(result)
        return results


# 기존 DermatologyAgent와의 호환성을 위한 별칭
DermatologyAgent = DermatologyAgentSAM


def test_structure():
    """온톨로지 구조 테스트"""
    print("=" * 60)
    print("SAM-Enhanced Dermatology Agent - Structure Test")
    print("=" * 60)

    print(f"SAM Available: {SAM_AVAILABLE}")
    print(f"LLM-Guided Available: {LLM_GUIDED_AVAILABLE}")

    try:
        tree = OntologyTree()
        print(f"\nOntology loaded: {tree.ontology_path}")
        print(f"  Total nodes: {len(tree.valid_nodes)}")

        leaf_nodes = [n for n in tree.valid_nodes if not tree.get_children(n)]
        print(f"  Leaf nodes: {len(leaf_nodes)}")
        print(f"  Intermediate nodes: {len(tree.valid_nodes) - len(leaf_nodes)}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return False

    print("\nStructure test completed!")
    return True


def test_with_vlm(api_key: str, image_path: str):
    """VLM을 사용한 실제 테스트"""
    print("=" * 60)
    print("SAM-Enhanced Dermatology Agent - VLM Test")
    print("=" * 60)

    try:
        from vlm_wrapper import GPT4oWrapper
    except ImportError:
        from model import GPT4o as GPT4oWrapper

    vlm = GPT4oWrapper(api_key=api_key, use_labels_prompt=False)
    print("VLM initialized")

    agent = DermatologyAgentSAM(
        vlm_model=vlm,
        segmenter_type="sam",
        segmentation_strategy="center",
        use_segmentation=SAM_AVAILABLE,
        verbose=True
    )
    print(f"Agent initialized (valid nodes: {len(agent.valid_diseases)})")

    print(f"\nDiagnosing: {image_path}")
    result = agent.diagnose(image_path)

    print(f"\n=== Diagnosis Result ===")
    print(f"Final Diagnosis: {result.get('final_diagnosis', [])}")
    print(f"Diagnosis Path: {' → '.join(result.get('diagnosis_path', []))}")
    print(f"Segmentation: {result.get('segmentation', {})}")

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SAM-Enhanced Dermatology Agent Test")
    parser.add_argument("--api_key", type=str, default=None, help="OpenAI API key")
    parser.add_argument("--image", type=str, default=None, help="Test image path")
    parser.add_argument("--test", action="store_true", help="Run structure test only")

    args = parser.parse_args()

    test_structure()

    if args.api_key and args.image:
        print("\n")
        test_with_vlm(args.api_key, args.image)
    elif args.api_key or args.image:
        print("\n[Note] Both --api_key and --image are required for VLM test")
