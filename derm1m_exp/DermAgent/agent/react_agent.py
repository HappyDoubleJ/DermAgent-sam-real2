"""
ReAct (Reasoning + Acting) Pattern Dermatology Agent

ReAct 패턴을 적용한 피부과 진단 에이전트
- OpenAI GPT-4o API 사용 (Vision 지원)
- 외부 온톨로지 파일 사용 (ontology.json)
- 도구 기반 단계적 추론

Author: DermAgent Team
"""

import json
import re
import base64
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

# 경로 설정 - eval 폴더의 모듈을 import하기 위해
SCRIPT_DIR = Path(__file__).parent
AGENT_DIR = SCRIPT_DIR.parent  # DermAgent
DERM1M_EXP_DIR = AGENT_DIR.parent  # derm1m_exp
sys.path.insert(0, str(DERM1M_EXP_DIR / "eval"))
sys.path.insert(0, str(DERM1M_EXP_DIR / "experiments"))

from ontology_utils import OntologyTree


# ============ 데이터 클래스 ============

class ActionType(Enum):
    """에이전트 행동 유형"""
    OBSERVE = "observe"              # 이미지 관찰
    ANALYZE_FEATURES = "analyze"     # 특징 분석
    NAVIGATE_ONTOLOGY = "navigate"   # 온톨로지 탐색
    COMPARE_DISEASES = "compare"     # 질환 비교
    VERIFY = "verify"                # 진단 검증
    CONCLUDE = "conclude"            # 결론 도출


@dataclass
class ClinicalObservation:
    """임상 관찰 결과"""
    morphology: List[str] = field(default_factory=list)    # 형태: papule, plaque, vesicle 등
    color: List[str] = field(default_factory=list)         # 색상: red, brown, white 등
    distribution: List[str] = field(default_factory=list)  # 분포: localized, generalized 등
    surface: List[str] = field(default_factory=list)       # 표면: scaly, smooth, crusted 등
    border: List[str] = field(default_factory=list)        # 경계: well-defined, irregular 등
    location: str = ""                                      # 신체 위치
    size: str = ""                                          # 크기
    pattern: List[str] = field(default_factory=list)       # 패턴: reticular, linear 등
    symptoms: List[str] = field(default_factory=list)      # 증상: itching, burning 등
    duration: str = ""                                      # 지속 기간
    additional_notes: str = ""                              # 추가 소견
    confidence: float = 0.0                                 # 관찰 신뢰도

    def to_dict(self) -> Dict:
        return {
            "morphology": self.morphology,
            "color": self.color,
            "distribution": self.distribution,
            "surface": self.surface,
            "border": self.border,
            "location": self.location,
            "size": self.size,
            "pattern": self.pattern,
            "symptoms": self.symptoms,
            "duration": self.duration,
            "additional_notes": self.additional_notes,
            "confidence": self.confidence
        }

    def to_text(self) -> str:
        """텍스트 형태로 변환"""
        parts = []
        if self.morphology:
            parts.append(f"Morphology: {', '.join(self.morphology)}")
        if self.color:
            parts.append(f"Color: {', '.join(self.color)}")
        if self.distribution:
            parts.append(f"Distribution: {', '.join(self.distribution)}")
        if self.surface:
            parts.append(f"Surface: {', '.join(self.surface)}")
        if self.border:
            parts.append(f"Border: {', '.join(self.border)}")
        if self.pattern:
            parts.append(f"Pattern: {', '.join(self.pattern)}")
        if self.location:
            parts.append(f"Location: {self.location}")
        if self.symptoms:
            parts.append(f"Symptoms: {', '.join(self.symptoms)}")
        return "\n".join(parts) if parts else "No observation available"


@dataclass
class ThoughtStep:
    """ReAct 사고 단계"""
    step_num: int
    thought: str           # 현재 생각 (Reasoning)
    action: ActionType     # 수행할 행동 (Acting)
    action_input: Dict     # 행동 입력 파라미터
    observation: str       # 행동 결과 (Observation)

    def to_dict(self) -> Dict:
        return {
            "step": self.step_num,
            "thought": self.thought,
            "action": self.action.value if isinstance(self.action, ActionType) else str(self.action),
            "action_input": self.action_input,
            "observation": self.observation[:500] + "..." if len(self.observation) > 500 else self.observation
        }


@dataclass
class DiagnosisResult:
    """진단 결과"""
    primary_diagnosis: str = ""
    differential_diagnoses: List[str] = field(default_factory=list)
    confidence: float = 0.0
    severity: str = ""  # mild, moderate, severe, urgent
    ontology_path: List[str] = field(default_factory=list)
    observations: Optional[ClinicalObservation] = None
    reasoning_chain: List[ThoughtStep] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    is_urgent: bool = False
    verification_passed: bool = True
    timestamp: str = ""

    def to_dict(self) -> Dict:
        return {
            "primary_diagnosis": self.primary_diagnosis,
            "differential_diagnoses": self.differential_diagnoses,
            "confidence": self.confidence,
            "severity": self.severity,
            "ontology_path": self.ontology_path,
            "observations": self.observations.to_dict() if self.observations else {},
            "reasoning_chain": [s.to_dict() for s in self.reasoning_chain],
            "recommendations": self.recommendations,
            "warnings": self.warnings,
            "is_urgent": self.is_urgent,
            "verification_passed": self.verification_passed,
            "timestamp": self.timestamp
        }

    def summary(self) -> str:
        """진단 요약"""
        lines = [
            "=" * 60,
            "Dermatology Diagnosis Result",
            "=" * 60,
            f"Primary Diagnosis: {self.primary_diagnosis}",
            f"Confidence: {self.confidence:.1%}",
            f"Severity: {self.severity}",
        ]

        if self.is_urgent:
            lines.append("URGENT: Immediate specialist consultation required!")

        if self.differential_diagnoses:
            lines.append(f"Differential: {', '.join(self.differential_diagnoses[:3])}")

        if self.ontology_path:
            lines.append(f"Ontology Path: {' -> '.join(self.ontology_path)}")

        if self.recommendations:
            lines.append("\nRecommendations:")
            for rec in self.recommendations[:3]:
                lines.append(f"  - {rec}")

        if self.warnings:
            lines.append("\nWarnings:")
            for warn in self.warnings:
                lines.append(f"  - {warn}")

        lines.append("=" * 60)
        return "\n".join(lines)


# ============ 도구 클래스 ============

class OntologyNavigatorTool:
    """온톨로지 탐색 도구 - OntologyTree 사용"""

    def __init__(self, tree: OntologyTree):
        self.tree = tree

    @property
    def name(self) -> str:
        return "navigate_ontology"

    @property
    def description(self) -> str:
        return "Navigate the disease ontology tree to find relevant categories and diseases"

    def execute(self, action: str, node: str = "root", query: str = "") -> str:
        """
        온톨로지 탐색 실행

        Actions:
        - get_children: 자식 노드들 반환
        - get_info: 노드 정보 반환
        - get_path: 루트까지 경로 반환
        - search: 질환명 검색
        - validate: 노드 유효성 검증
        """
        try:
            if action == "get_children":
                # root는 특수 처리 (valid_nodes에 포함되지 않음)
                if node == "root":
                    children = self.tree.ontology.get("root", [])
                else:
                    children = self.tree.get_children(node)
                return json.dumps({
                    "node": node,
                    "children": children,
                    "count": len(children)
                })

            elif action == "get_info":
                is_valid = self.tree.is_valid_node(node)
                if not is_valid:
                    return json.dumps({"error": f"Node not found: {node}"})

                children = self.tree.get_children(node)
                path = self.tree.get_path_to_root(node)
                return json.dumps({
                    "node": node,
                    "valid": True,
                    "children": children,
                    "path_to_root": path,
                    "is_leaf": len(children) == 0
                })

            elif action == "get_path":
                path = self.tree.get_path_to_root(node)
                return json.dumps({
                    "node": node,
                    "path": path,
                    "depth": len(path) - 1
                })

            elif action == "search":
                # 검색 기능 - 이름에 query가 포함된 노드 찾기
                matches = []
                query_lower = query.lower().replace(" ", "_").replace("-", "_")

                for valid_node in self.tree.valid_nodes:
                    node_lower = valid_node.lower()
                    if query_lower in node_lower or node_lower in query_lower:
                        children = self.tree.get_children(valid_node)
                        matches.append({
                            "name": valid_node,
                            "is_leaf": len(children) == 0
                        })

                return json.dumps({
                    "query": query,
                    "matches": matches[:15]
                })

            elif action == "validate":
                is_valid = self.tree.is_valid_node(node)
                canonical = self.tree.get_canonical_name(node) if is_valid else None
                return json.dumps({
                    "node": node,
                    "valid": is_valid,
                    "canonical_name": canonical
                })

            else:
                return json.dumps({"error": f"Unknown action: {action}"})

        except Exception as e:
            return json.dumps({"error": str(e)})


class DiseaseComparatorTool:
    """질환 비교 도구 - VLM 기반"""

    def __init__(self, tree: OntologyTree, vlm_model=None):
        self.tree = tree
        self.vlm = vlm_model

    @property
    def name(self) -> str:
        return "compare_diseases"

    @property
    def description(self) -> str:
        return "Compare observed clinical features with candidate diseases using VLM"

    def execute(
        self,
        candidates: List[str],
        observations: ClinicalObservation,
        image_path: str = None
    ) -> str:
        """후보 질환들과 관찰된 특징 비교"""
        if self.vlm is None or image_path is None:
            # VLM 없이 기본 비교
            return json.dumps({
                "comparisons": [
                    {"disease": c, "score": 0.5, "matched_features": []}
                    for c in candidates
                ]
            })

        try:
            scores = self._compare_with_vlm_batch(candidates, observations, image_path)
            return json.dumps({"comparisons": scores})
        except Exception as e:
            return json.dumps({"error": str(e)})

    def _compare_with_vlm_batch(
        self,
        candidates: List[str],
        observations: ClinicalObservation,
        image_path: str
    ) -> List[Dict]:
        """VLM을 사용하여 후보 질환들 비교"""
        if not candidates:
            return []

        candidates_list = "\n".join([f"{i+1}. {c}" for i, c in enumerate(candidates)])
        obs_text = observations.to_text()

        prompt = f"""Compare this skin lesion with the following candidate diagnoses and rate each one.

Candidate Diagnoses:
{candidates_list}

Observed Clinical Features:
{obs_text}

For EACH candidate diagnosis, evaluate how well the observed features match.
Rate each with a likelihood score from 0-10.

Respond in JSON format:
{{
    "comparisons": [
        {{
            "disease": "exact disease name from the list",
            "likelihood_score": 0-10,
            "brief_reasoning": "one sentence explanation"
        }},
        ... (one entry for each candidate)
    ]
}}

IMPORTANT: Include ALL {len(candidates)} candidates. Provide ONLY the JSON output."""

        try:
            response = self.vlm.chat_img(prompt, [image_path], max_tokens=1500)
            json_match = re.search(r'\{[\s\S]*\}', response)

            if json_match:
                parsed = json.loads(json_match.group())
                comparisons = parsed.get("comparisons", [])

                results = []
                for comp in comparisons:
                    if not isinstance(comp, dict):
                        continue
                    disease = comp.get("disease", "").strip()
                    if not disease:
                        continue

                    try:
                        score = float(comp.get("likelihood_score", 5)) / 10.0
                    except (TypeError, ValueError):
                        score = 0.5

                    results.append({
                        "disease": disease,
                        "score": round(score, 3),
                        "reasoning": comp.get("brief_reasoning", "")
                    })

                # 누락된 후보들 추가
                result_diseases = {r["disease"].lower() for r in results}
                for candidate in candidates:
                    if candidate.lower() not in result_diseases:
                        results.append({
                            "disease": candidate,
                            "score": 0.5,
                            "reasoning": "Not evaluated"
                        })

                return sorted(results, key=lambda x: x["score"], reverse=True)

            return [{"disease": c, "score": 0.5, "reasoning": "Parse failed"} for c in candidates]

        except Exception:
            return [{"disease": c, "score": 0.5, "reasoning": "VLM error"} for c in candidates]


# ============ ReAct 에이전트 ============

class ReActDermatologyAgent:
    """
    ReAct 패턴 기반 피부질환 진단 에이전트

    Reasoning과 Acting을 반복하며 단계적으로 진단을 수행합니다.
    """

    def __init__(
        self,
        ontology_path: Optional[str] = None,
        vlm_model: Any = None,
        max_steps: int = 8,
        verbose: bool = True
    ):
        """
        초기화

        Args:
            ontology_path: 온톨로지 JSON 파일 경로 (None이면 자동 탐색)
            vlm_model: Vision-Language Model (GPT4oWrapper 등)
            max_steps: 최대 추론 단계 수
            verbose: 상세 로그 출력 여부
        """
        # 온톨로지 트리 초기화
        self.tree = OntologyTree(ontology_path)
        self.vlm = vlm_model
        self.max_steps = max_steps
        self.verbose = verbose

        if vlm_model is None:
            raise ValueError("vlm_model must be provided")

        # 도구 초기화
        self.tools = {
            "navigator": OntologyNavigatorTool(self.tree),
            "comparator": DiseaseComparatorTool(self.tree, self.vlm)
        }

        # 루트 카테고리
        self.root_categories = self.tree.ontology.get("root", [])

        # 유효 노드 목록
        self.valid_diseases = sorted(list(self.tree.valid_nodes))

        # 프롬프트 템플릿 로드
        self._load_prompts()

        if self.verbose:
            self._log(f"ReAct Agent initialized with {len(self.valid_diseases)} valid nodes")

    def _load_prompts(self):
        """프롬프트 템플릿 로드"""
        self.prompts = {
            "system": """You are a board-certified dermatology expert using ReAct (Reasoning + Acting) pattern for diagnosis.

Your role:
- Analyze skin images and identify clinical features
- Use step-by-step reasoning with explicit thought process
- Navigate the disease ontology to find the most appropriate diagnosis
- Provide differential diagnoses when uncertain

Important:
- This is for educational/reference purposes only
- Always recommend professional medical consultation
- Flag urgent conditions immediately""",

            "observation": """Analyze this dermatological image and describe what you observe.

IMPORTANT: If no clear skin lesion is visible, the image does not show identifiable human skin, or you cannot make a confident diagnosis, respond with:
{
    "morphology": ["no visible lesion"],
    "color": ["not observed"],
    "distribution": ["not observed"],
    "surface": ["not observed"],
    "border": ["not observed"],
    "location": "not observed",
    "additional_notes": "no definitive diagnosis"
}

Focus on PRIMARY LESION MORPHOLOGY - be VERY specific:
1. Morphology (primary lesion type): macule, patch, papule, plaque, nodule, wheal, vesicle, bulla, pustule, erosion, ulcer, etc.
2. Color: red, pink, brown, black, white, yellow, purple, skin-colored, etc.
3. Distribution: localized, generalized, symmetric, asymmetric, clustered, linear, dermatomal, etc.
4. Surface features: smooth, scaly, crusted, rough, verrucous, ulcerated, etc.
5. Border: well-defined, ill-defined, regular, irregular, raised, rolled
6. Body location: face, trunk, extremities, hands, feet, scalp, etc.

Provide your observations in JSON format:
{
    "morphology": ["list of PRIMARY lesion types"],
    "color": ["list of colors observed"],
    "distribution": ["distribution patterns"],
    "surface": ["surface features"],
    "border": ["border characteristics"],
    "location": "body location",
    "additional_notes": "any other relevant observations"
}

Provide ONLY the JSON output.""",

            "category_classification": """Based on the clinical features observed in this skin image,
classify this condition into ONE of the following major categories:

Categories:
{categories}

Consider the morphology, distribution, and clinical presentation.
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

Based on the image features:
{observations}

Respond with JSON:
{{
    "selected_subcategory": "the most likely subcategory",
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation"
}}

Provide ONLY the JSON output.""",

            "react_step": """You are performing step-by-step diagnosis using ReAct pattern.

Current diagnosis progress:
{history}

Clinical observations:
{observations}

Current candidates:
{candidates}

Think about what you should do next and take an action.

Available Actions:
1. navigate - Explore ontology (get children, validate nodes)
2. compare - Compare candidates with observed features
3. analyze - Perform deeper analysis
4. verify - Verify current diagnosis hypothesis
5. conclude - Make final diagnosis decision

Respond in JSON:
{{
    "thought": "Your current reasoning about the situation",
    "action": "the action to take (navigate/compare/analyze/verify/conclude)",
    "action_input": {{parameters for the action}}
}}

Provide ONLY the JSON output.""",

            "final_diagnosis": """Based on the hierarchical classification path:
{path}

And the clinical observations:
{observations}

Select the most likely specific diagnosis from these candidates:
{candidates}

IMPORTANT: If no clear skin lesion is visible, the image does not show identifiable human skin, or you cannot make a confident diagnosis, set primary_diagnosis to "no definitive diagnosis".

Respond with JSON:
{{
    "primary_diagnosis": "most likely diagnosis (or 'no definitive diagnosis' if uncertain)",
    "confidence": 0.0-1.0,
    "differential_diagnoses": ["other possible diagnoses in order of likelihood"],
    "reasoning": "clinical reasoning for your diagnosis"
}}

Provide ONLY the JSON output."""
        }

    def _log(self, message: str, level: str = "info"):
        """로깅"""
        if self.verbose:
            prefix = {
                "info": "[INFO]",
                "success": "[OK]",
                "warning": "[WARN]",
                "error": "[ERROR]",
                "step": "[STEP]"
            }.get(level, "")
            print(f"{prefix} [ReActAgent] {message}")

    def _call_vlm(self, prompt: str, image_path: str, max_tokens: int = 1024) -> str:
        """VLM 호출"""
        if self.vlm is None:
            raise RuntimeError("VLM model is not set")

        full_prompt = f"{self.prompts['system']}\n\n{prompt}"

        try:
            response = self.vlm.chat_img(full_prompt, [image_path], max_tokens=max_tokens)
            return response if response else "{}"
        except Exception as e:
            self._log(f"VLM Error: {e}", "error")
            return "{}"

    def _parse_json(self, response: str) -> Dict:
        """JSON 파싱"""
        if not response or not isinstance(response, str):
            return {}

        try:
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                return json.loads(json_match.group())
        except (json.JSONDecodeError, TypeError):
            pass
        return {}

    def _observe_image(self, image_path: str) -> ClinicalObservation:
        """Step 1: 이미지 관찰"""
        self._log("Observing image...", "step")

        response = self._call_vlm(self.prompts["observation"], image_path)
        parsed = self._parse_json(response)

        def _normalize_list(values):
            return values if values else ["not observed"]

        observation = ClinicalObservation(
            morphology=_normalize_list(parsed.get("morphology", [])),
            color=_normalize_list(parsed.get("color", [])),
            distribution=_normalize_list(parsed.get("distribution", [])),
            surface=_normalize_list(parsed.get("surface", [])),
            border=_normalize_list(parsed.get("border", [])),
            pattern=_normalize_list(parsed.get("pattern", [])),
            location=parsed.get("location", ""),
            additional_notes=parsed.get("additional_notes", ""),
            confidence=parsed.get("confidence", 0.5)
        )

        self._log(f"  Morphology: {observation.morphology}", "info")
        self._log(f"  Color: {observation.color}", "info")
        self._log(f"  Location: {observation.location}", "info")

        return observation

    def _classify_category(self, image_path: str) -> tuple:
        """Step 2: 대분류 선택"""
        self._log("Classifying category...", "step")

        categories_desc = "\n".join([f"- {cat}" for cat in self.root_categories])
        prompt = self.prompts["category_classification"].format(categories=categories_desc)

        response = self._call_vlm(prompt, image_path)
        parsed = self._parse_json(response)

        selected = parsed.get("selected_category", "")
        confidence = parsed.get("confidence", 0.5)

        # 유효성 확인
        canonical = self.tree.get_canonical_name(selected)
        if canonical and canonical in self.root_categories:
            self._log(f"  Selected category: {canonical} (conf: {confidence:.2f})", "info")
            return canonical, confidence

        # 폴백
        self._log(f"  Invalid category '{selected}', using fallback 'inflammatory'", "warning")
        return "inflammatory", 0.3

    def _classify_subcategory(
        self,
        image_path: str,
        parent_category: str,
        observations: ClinicalObservation
    ) -> tuple:
        """Step 3: 하위 카테고리 선택"""
        children = self.tree.get_children(parent_category)

        if not children:
            return parent_category, 0.9

        if len(children) == 1:
            return children[0], 0.9

        self._log(f"Classifying subcategory under '{parent_category}'...", "step")

        subcategories_desc = "\n".join([f"- {child}" for child in children])
        obs_desc = json.dumps({
            "morphology": observations.morphology,
            "color": observations.color,
            "location": observations.location
        }, indent=2)

        prompt = self.prompts["subcategory_classification"].format(
            parent_category=parent_category,
            subcategories=subcategories_desc,
            observations=obs_desc
        )

        response = self._call_vlm(prompt, image_path)
        parsed = self._parse_json(response)

        selected = parsed.get("selected_subcategory", "")
        confidence = parsed.get("confidence", 0.5)

        canonical = self.tree.get_canonical_name(selected)
        if canonical and canonical in children:
            self._log(f"  Selected: {canonical} (conf: {confidence:.2f})", "info")
            return canonical, confidence

        # 폴백: 첫 번째 자식
        self._log(f"  Invalid subcategory '{selected}', using first child: {children[0]}", "warning")
        return children[0], 0.3

    def _get_candidates(self, current_path: List[str]) -> List[str]:
        """현재 경로 기반 후보 질환 생성"""
        if not current_path:
            return []

        current_node = current_path[-1]
        descendants = self.tree.get_all_descendants(current_node)

        # 현재 노드 + 경로의 모든 노드 + 자손 노드
        all_candidates = set(descendants)
        all_candidates.update(current_path)
        all_candidates.add(current_node)

        return sorted(all_candidates)[:20]

    def _compare_candidates(
        self,
        candidates: List[str],
        observations: ClinicalObservation,
        image_path: str
    ) -> Dict[str, float]:
        """Step 4: 후보 질환 비교"""
        self._log(f"Comparing {len(candidates)} candidates...", "step")

        result = self.tools["comparator"].execute(candidates, observations, image_path)
        parsed = self._parse_json(result)

        scores = {}
        for comp in parsed.get("comparisons", []):
            if isinstance(comp, dict):
                disease = comp.get("disease", "")
                score = comp.get("score", 0.5)
                if disease:
                    scores[disease] = score

        # 누락된 후보 추가
        for candidate in candidates:
            if candidate not in scores:
                scores[candidate] = 0.5

        top_3 = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]
        self._log(f"  Top 3: {[(d, f'{s:.2f}') for d, s in top_3]}", "info")

        return scores

    def _make_final_diagnosis(
        self,
        image_path: str,
        current_path: List[str],
        candidates: List[str],
        observations: ClinicalObservation
    ) -> tuple:
        """Step 5: 최종 진단"""
        self._log("Making final diagnosis...", "step")

        path_str = " -> ".join(current_path)
        candidates_str = "\n".join([f"- {c}" for c in candidates[:15]])
        obs_str = json.dumps({
            "morphology": observations.morphology,
            "color": observations.color,
            "distribution": observations.distribution,
            "location": observations.location
        }, indent=2)

        prompt = self.prompts["final_diagnosis"].format(
            path=path_str,
            observations=obs_str,
            candidates=candidates_str
        )

        response = self._call_vlm(prompt, image_path, max_tokens=1500)
        parsed = self._parse_json(response)

        primary = parsed.get("primary_diagnosis", "")
        confidence = parsed.get("confidence", 0.5)
        differentials = parsed.get("differential_diagnoses", [])
        reasoning = parsed.get("reasoning", "")

        return primary, confidence, differentials, reasoning

    def diagnose(self, image_path: str) -> DiagnosisResult:
        """
        메인 진단 메서드

        Args:
            image_path: 피부 이미지 경로

        Returns:
            DiagnosisResult: 진단 결과
        """
        self._log("=" * 50)
        self._log(f"Starting diagnosis: {image_path}")
        self._log("=" * 50)

        result = DiagnosisResult(timestamp=datetime.now().isoformat())
        reasoning_chain = []
        current_path = []
        confidence_scores = {}

        try:
            # Step 1: 이미지 관찰
            observations = self._observe_image(image_path)
            result.observations = observations

            reasoning_chain.append(ThoughtStep(
                step_num=1,
                thought="Observing clinical features from the skin image",
                action=ActionType.OBSERVE,
                action_input={"image_path": image_path},
                observation=observations.to_text()
            ))

            # 병변이 보이지 않으면 조기 종료
            if "no visible lesion" in [m.lower() for m in observations.morphology]:
                self._log("No visible lesion detected", "warning")
                result.primary_diagnosis = "no definitive diagnosis"
                result.confidence = 0.0
                result.reasoning_chain = reasoning_chain
                result.warnings.append("No visible skin lesion detected in the image")
                return result

            # Step 2: 대분류
            category, cat_conf = self._classify_category(image_path)
            current_path.append(category)
            confidence_scores[category] = cat_conf

            reasoning_chain.append(ThoughtStep(
                step_num=2,
                thought=f"Classifying into major category: {category}",
                action=ActionType.NAVIGATE_ONTOLOGY,
                action_input={"action": "get_children", "node": category},
                observation=f"Selected category: {category} (confidence: {cat_conf:.2f})"
            ))

            # Step 3: 하위 카테고리 탐색 (최대 3단계)
            for depth in range(3):
                children = self.tree.get_children(current_path[-1])
                if not children:
                    break

                subcategory, sub_conf = self._classify_subcategory(
                    image_path, current_path[-1], observations
                )
                current_path.append(subcategory)
                confidence_scores[subcategory] = sub_conf

                reasoning_chain.append(ThoughtStep(
                    step_num=3 + depth,
                    thought=f"Classifying into subcategory: {subcategory}",
                    action=ActionType.NAVIGATE_ONTOLOGY,
                    action_input={"parent": current_path[-2], "selected": subcategory},
                    observation=f"Selected: {subcategory} (confidence: {sub_conf:.2f})"
                ))

            # Step 4: 후보 비교
            candidates = self._get_candidates(current_path)

            if candidates:
                scores = self._compare_candidates(candidates, observations, image_path)
                confidence_scores.update(scores)

                reasoning_chain.append(ThoughtStep(
                    step_num=len(reasoning_chain) + 1,
                    thought="Comparing candidate diseases with observed features",
                    action=ActionType.COMPARE_DISEASES,
                    action_input={"candidates": candidates[:10]},
                    observation=f"Compared {len(candidates)} candidates"
                ))

            # Step 5: 최종 진단
            primary, conf, differentials, reasoning = self._make_final_diagnosis(
                image_path, current_path, candidates, observations
            )

            # 유효성 검증 및 결과 설정
            if primary.lower() == "no definitive diagnosis":
                result.primary_diagnosis = "no definitive diagnosis"
                result.confidence = conf
            else:
                canonical_primary = self.tree.get_canonical_name(primary)
                if canonical_primary:
                    result.primary_diagnosis = canonical_primary
                    result.confidence = conf
                else:
                    # VLM이 반환한 진단이 온톨로지에 없는 경우
                    # 점수 기반으로 가장 좋은 후보 선택
                    best_candidate = max(confidence_scores.items(), key=lambda x: x[1])[0] if confidence_scores else ""
                    if self.tree.is_valid_node(best_candidate):
                        result.primary_diagnosis = self.tree.get_canonical_name(best_candidate)
                        result.confidence = confidence_scores.get(best_candidate, 0.5)
                        result.warnings.append(f"VLM diagnosis '{primary}' not in ontology, using '{result.primary_diagnosis}'")
                    else:
                        result.primary_diagnosis = "no definitive diagnosis"
                        result.confidence = 0.3
                        result.warnings.append(f"Could not validate diagnosis")

            # 감별 진단 설정
            valid_differentials = []
            for diff in differentials[:5]:
                canonical = self.tree.get_canonical_name(diff)
                if canonical and canonical != result.primary_diagnosis:
                    valid_differentials.append(canonical)
            result.differential_diagnoses = valid_differentials

            # 온톨로지 경로 설정
            if result.primary_diagnosis and result.primary_diagnosis != "no definitive diagnosis":
                result.ontology_path = self.tree.get_path_to_root(result.primary_diagnosis)
            else:
                result.ontology_path = current_path

            reasoning_chain.append(ThoughtStep(
                step_num=len(reasoning_chain) + 1,
                thought=f"Final diagnosis: {result.primary_diagnosis}",
                action=ActionType.CONCLUDE,
                action_input={"diagnosis": result.primary_diagnosis},
                observation=f"Confidence: {result.confidence:.2f}, Reasoning: {reasoning[:200]}"
            ))

            result.reasoning_chain = reasoning_chain

            # 중증도 및 권장사항
            result.severity = "moderate"  # 기본값
            result.recommendations = [
                "Please consult a dermatologist for professional diagnosis.",
                "Monitor the lesion for any changes in size, color, or symptoms."
            ]
            result.warnings.append("This is an AI-assisted analysis for reference only. Consult a medical professional.")

            self._log("=" * 50)
            self._log(f"Diagnosis complete: {result.primary_diagnosis}")
            self._log(f"Confidence: {result.confidence:.2f}")
            self._log(f"Path: {' -> '.join(result.ontology_path)}")
            self._log("=" * 50)

        except Exception as e:
            self._log(f"Error during diagnosis: {e}", "error")
            result.primary_diagnosis = "no definitive diagnosis"
            result.confidence = 0.0
            result.warnings.append(f"Error occurred: {str(e)}")
            result.reasoning_chain = reasoning_chain

        return result

    def diagnose_batch(self, image_paths: List[str]) -> List[DiagnosisResult]:
        """배치 진단"""
        results = []
        for path in image_paths:
            result = self.diagnose(path)
            results.append(result)
        return results


# ============ 테스트 함수 ============

def test_structure():
    """에이전트 구조 테스트 (VLM 없이)"""
    print("=" * 60)
    print("ReAct Agent - Structure Test")
    print("=" * 60)

    try:
        tree = OntologyTree()
        print(f"\nOntology loaded: {tree.ontology_path}")
        print(f"  Total nodes: {len(tree.valid_nodes)}")

        # 리프 노드와 중간 노드 수 계산
        leaf_nodes = [n for n in tree.valid_nodes if not tree.get_children(n)]
        print(f"  Leaf nodes: {len(leaf_nodes)}")
        print(f"  Intermediate nodes: {len(tree.valid_nodes) - len(leaf_nodes)}")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        return False

    print("\n[OntologyNavigatorTool Test]")
    nav = OntologyNavigatorTool(tree)

    print("\nRoot categories:")
    result = json.loads(nav.execute("get_children", "root"))
    for child in result["children"]:
        print(f"  - {child}")

    print("\n'inflammatory' children:")
    result = json.loads(nav.execute("get_children", "inflammatory"))
    for child in result["children"][:5]:
        print(f"  - {child}")

    print("\n'Tinea corporis' path:")
    result = json.loads(nav.execute("get_path", "Tinea corporis"))
    print(f"  {' -> '.join(result['path'])}")

    print("\n[Search 'eczema']")
    result = json.loads(nav.execute("search", query="eczema"))
    for match in result["matches"][:5]:
        print(f"  - {match['name']}")

    print("\nStructure test completed!")
    return True


def test_with_vlm(api_key: str, image_path: str):
    """VLM을 사용한 실제 테스트"""
    print("=" * 60)
    print("ReAct Agent - VLM Test")
    print("=" * 60)

    try:
        from vlm_wrapper import GPT4oWrapper
    except ImportError:
        print("vlm_wrapper not found, trying alternative import...")
        try:
            from model import GPT4o as GPT4oWrapper
        except ImportError:
            print("Error: Could not import VLM wrapper")
            return None

    # VLM 초기화
    vlm = GPT4oWrapper(api_key=api_key, use_labels_prompt=False)
    print("VLM initialized")

    # 에이전트 초기화
    agent = ReActDermatologyAgent(vlm_model=vlm, verbose=True)
    print(f"Agent initialized (valid nodes: {len(agent.valid_diseases)})")

    # 진단 수행
    print(f"\nDiagnosing: {image_path}")
    result = agent.diagnose(image_path)

    print(f"\n=== Diagnosis Result ===")
    print(f"Primary Diagnosis: {result.primary_diagnosis}")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"Differentials: {result.differential_diagnoses}")
    print(f"Path: {' -> '.join(result.ontology_path)}")

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ReAct Dermatology Agent")
    parser.add_argument("--api_key", type=str, default=None,
                        help="OpenAI API key")
    parser.add_argument("--image", type=str, default=None,
                        help="Image path for testing")
    parser.add_argument("--test", action="store_true",
                        help="Run structure test only")

    args = parser.parse_args()

    # 구조 테스트
    test_structure()

    # VLM 테스트
    if args.api_key and args.image:
        print("\n")
        test_with_vlm(args.api_key, args.image)
    elif args.api_key or args.image:
        print("\n[Note] Both --api_key and --image are required for VLM test")
