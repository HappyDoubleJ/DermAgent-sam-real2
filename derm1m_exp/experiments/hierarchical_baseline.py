"""
Hierarchical Baseline Model

전체 온톨로지 트리 구조를 프롬프트에 포함하여
모델이 계층적으로 탐색하도록 유도하는 베이스라인 모델
"""

import os
import sys
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Any

# 경로 설정
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPT_DIR.parent / "eval"))
sys.path.insert(0, str(SCRIPT_DIR.parent / "baseline"))

from ontology_utils import OntologyTree
from experiment_utils import build_ontology_tree_text


class HierarchicalBaselineModel:
    """
    온톨로지 트리 구조를 프롬프트에 포함하는 베이스라인 모델

    프롬프트에 전체 질환 계층 구조를 포함하여
    모델이 루트부터 리프까지 탐색하며 진단하도록 유도
    """

    def __init__(
        self,
        vlm_model,  # GPT4o 또는 다른 VLM
        ontology_path: Optional[str] = None,
        verbose: bool = True
    ):
        """
        Args:
            vlm_model: Vision-Language Model (GPT4o, QwenVL 등)
            ontology_path: ontology.json 경로 (None이면 자동 탐색)
            verbose: 상세 출력 여부
        """
        self.vlm = vlm_model
        self.verbose = verbose

        # 온톨로지 로드
        self.tree = OntologyTree(ontology_path)
        self.ontology_path = self.tree.ontology_path

        # 온톨로지 트리 텍스트 생성 (프롬프트용)
        self.tree_text = self._build_tree_text()

        # 전체 유효 노드 목록 (중간 노드 + 리프 노드 모두 포함)
        self.valid_diseases = sorted(list(self.tree.valid_nodes))

        # 리프 노드 목록 (참고용)
        self.leaf_diseases = sorted([
            n for n in self.tree.valid_nodes
            if not self.tree.get_children(n)
        ])

        # 시스템 프롬프트 구성
        self.system_instruction = self._build_system_instruction()

        if self.verbose:
            print(f"[HierarchicalBaseline] 온톨로지 로드 완료: {self.ontology_path}")
            print(f"[HierarchicalBaseline] 전체 유효 노드 수: {len(self.valid_diseases)}")
            print(f"[HierarchicalBaseline] 리프 노드 수: {len(self.leaf_diseases)}")

    def _build_tree_text(self) -> str:
        """온톨로지 트리를 텍스트로 변환"""
        return build_ontology_tree_text(self.tree.ontology, "root", 0, 10, "")

    def _build_system_instruction(self) -> str:
        """시스템 프롬프트 구성"""
        return f"""You are a board-certified dermatology expert. You will analyze skin images and provide diagnoses.

IMPORTANT: You must select your diagnosis from the following Disease Ontology Tree.
Navigate from the root categories and select the most appropriate node that matches the skin condition.
NOTE: Your diagnosis can be ANY node in the tree - both intermediate category nodes AND leaf nodes are valid diagnoses.
For example, if you can only identify the condition as "eczema" but cannot determine the specific subtype, "eczema" is a valid answer.

Disease Ontology Tree:
{self.tree_text}

Guidelines:
1. Start by identifying the broad category (inflammatory, proliferations, etc.)
2. Narrow down through subcategories as much as the clinical features allow
3. Select the most specific diagnosis that you can confidently identify from the tree
   - If you can identify a specific disease (e.g., "Tinea corporis"), select that
   - If you can only identify to a category level (e.g., "fungal" or "eczema"), that is also valid
4. If uncertain, provide your best match from the tree with lower confidence

IMPORTANT: If no clear skin lesion is visible, the image does not show identifiable human skin, or you cannot make a confident diagnosis, set disease_label to "no definitive diagnosis".

Always select from the ontology tree above. Do not invent new disease names."""

    def analyze_image(self, image_path: str) -> Dict[str, Any]:
        """
        이미지 분석 및 계층적 진단

        Args:
            image_path: 피부 이미지 경로

        Returns:
            {
                "disease_label": str,
                "ontology_path": List[str],
                "confidence": float,
                "reasoning": str,
                "body_location": str,
                "raw_response": str
            }
        """
        prompt = f"""{self.system_instruction}

Please analyze this dermatological image and provide your diagnosis.

Navigate the Disease Ontology Tree and select the most appropriate diagnosis.
Your diagnosis can be ANY node from the tree - intermediate categories OR specific diseases.
Provide your response in JSON format:
{{
    "disease_label": "the most appropriate diagnosis from the ontology tree (any level)",
    "ontology_path": ["category", "subcategory", "...", "your_diagnosis"],
    "confidence": 0.0-1.0,
    "body_location": "anatomical location of the lesion",
    "reasoning": "your clinical reasoning for this diagnosis"
}}

IMPORTANT:
- disease_label MUST be from the ontology tree provided above (any node is valid)
- Select the most specific level you can confidently identify
- If no clear skin lesion is visible, the image does not show identifiable human skin, or you cannot make a confident diagnosis, set disease_label to "no definitive diagnosis"

Provide ONLY the JSON output."""

        try:
            # 이미지 존재 확인
            if not os.path.exists(image_path):
                return {
                    "disease_label": "Error: Image not found",
                    "ontology_path": [],
                    "confidence": 0.0,
                    "body_location": "",
                    "reasoning": f"Image file does not exist: {image_path}",
                    "raw_response": ""
                }

            # VLM 호출
            response = self.vlm.chat_img(prompt, [image_path], max_tokens=1024)

            # JSON 파싱
            result = self._parse_response(response)
            result["raw_response"] = response

            # 진단 유효성 검증
            result = self._validate_diagnosis(result)

            return result

        except Exception as e:
            return {
                "disease_label": "Error",
                "ontology_path": [],
                "confidence": 0.0,
                "body_location": "",
                "reasoning": f"Error processing image: {str(e)}",
                "raw_response": ""
            }

    def _parse_response(self, response) -> Dict[str, Any]:
        """VLM 응답 파싱 (None 및 비문자열 타입 처리)"""
        # None이나 비문자열 타입 처리
        if response is None:
            return {
                "disease_label": "",
                "ontology_path": [],
                "confidence": 0.0,
                "body_location": "",
                "reasoning": "No response from VLM"
            }
        if not isinstance(response, str):
            response = str(response)

        try:
            # JSON 추출
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                parsed = json.loads(json_match.group())
                return {
                    "disease_label": parsed.get("disease_label", ""),
                    "ontology_path": parsed.get("ontology_path", []),
                    "confidence": float(parsed.get("confidence", 0.5)),
                    "body_location": parsed.get("body_location", ""),
                    "reasoning": parsed.get("reasoning", "")
                }
            else:
                # JSON 없으면 텍스트에서 진단 추출 시도
                return {
                    "disease_label": "",
                    "ontology_path": [],
                    "confidence": 0.0,
                    "body_location": "",
                    "reasoning": response
                }
        except (json.JSONDecodeError, TypeError):
            return {
                "disease_label": "",
                "ontology_path": [],
                "confidence": 0.0,
                "body_location": "",
                "reasoning": response if isinstance(response, str) else str(response)
            }

    def _validate_diagnosis(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """진단 유효성 검증 및 정규화"""
        disease_label = result.get("disease_label", "")

        # "no definitive diagnosis" 특수 케이스
        if disease_label.lower() == "no definitive diagnosis":
            result["disease_label"] = "no definitive diagnosis"
            return result

        # 온톨로지에서 정규화된 이름 찾기
        canonical = self.tree.get_canonical_name(disease_label)
        if canonical:
            result["disease_label"] = canonical

            # ontology_path 없으면 생성
            if not result.get("ontology_path"):
                path = self.tree.get_path_to_root(canonical)
                result["ontology_path"] = list(reversed(path))[1:]  # root 제외

        return result

    def analyze_batch(
        self,
        image_paths: List[str],
        show_progress: bool = True
    ) -> List[Dict[str, Any]]:
        """
        배치 이미지 분석

        Args:
            image_paths: 이미지 경로 리스트
            show_progress: 진행 상황 표시 여부

        Returns:
            분석 결과 리스트
        """
        results = []

        iterator = image_paths
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(image_paths, desc="Hierarchical Baseline")
            except ImportError:
                pass

        for path in iterator:
            result = self.analyze_image(path)
            results.append(result)

            if self.verbose and not show_progress:
                print(f"  {path}: {result['disease_label']}")

        return results


# ============ 테스트 함수 ============

def test_structure():
    """온톨로지 구조 테스트 (VLM 없이)"""
    print("=" * 60)
    print("Hierarchical Baseline - 구조 테스트")
    print("=" * 60)

    try:
        tree = OntologyTree()
        print(f"\n온톨로지 로드 완료: {tree.ontology_path}")
        print(f"전체 노드 수: {len(tree.valid_nodes)}")

        # 리프 노드와 중간 노드 수 계산
        leaf_nodes = [n for n in tree.valid_nodes if not tree.get_children(n)]
        intermediate_nodes = [n for n in tree.valid_nodes if tree.get_children(n)]
        print(f"리프 노드 수: {len(leaf_nodes)}")
        print(f"중간 노드 수: {len(intermediate_nodes)}")

        # 트리 텍스트 생성
        tree_text = build_ontology_tree_text(tree.ontology, "root", 0, 3, "")
        print(f"\n온톨로지 트리 (일부):\n{tree_text[:2000]}...")

        print("\n✓ 구조 테스트 완료")
        return True

    except FileNotFoundError as e:
        print(f"Error: {e}")
        return False


def test_with_vlm(api_key: str, image_path: str):
    """
    VLM을 사용한 실제 테스트

    Args:
        api_key: OpenAI API 키
        image_path: 테스트 이미지 경로

    Returns:
        분석 결과 딕셔너리
    """
    print("=" * 60)
    print("Hierarchical Baseline - VLM 테스트")
    print("=" * 60)

    # VLM 래퍼 임포트
    try:
        from vlm_wrapper import GPT4oWrapper
    except ImportError:
        from model import GPT4o as GPT4oWrapper

    # VLM 초기화
    vlm = GPT4oWrapper(api_key=api_key, use_labels_prompt=False)
    print("VLM 초기화 완료")

    # 모델 초기화
    model = HierarchicalBaselineModel(vlm, verbose=True)
    print(f"모델 초기화 완료 (유효 노드: {len(model.valid_diseases)}개)")

    # 이미지 분석
    print(f"\n이미지 분석 중: {image_path}")
    result = model.analyze_image(image_path)

    print(f"\n=== 분석 결과 ===")
    print(f"진단: {result.get('disease_label', 'N/A')}")
    print(f"경로: {' → '.join(result.get('ontology_path', []))}")
    print(f"신뢰도: {result.get('confidence', 0.0):.2f}")
    print(f"위치: {result.get('body_location', 'N/A')}")
    print(f"근거: {result.get('reasoning', 'N/A')[:200]}...")

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Hierarchical Baseline 테스트")
    parser.add_argument("--api_key", type=str, default=None,
                        help="OpenAI API 키 (없으면 구조 테스트만 실행)")
    parser.add_argument("--image", type=str, default=None,
                        help="테스트 이미지 경로")

    args = parser.parse_args()

    # 구조 테스트
    test_structure()

    # VLM 테스트 (API 키와 이미지가 있는 경우)
    if args.api_key and args.image:
        print("\n")
        test_with_vlm(args.api_key, args.image)
    elif args.api_key or args.image:
        print("\n[참고] VLM 테스트를 실행하려면 --api_key와 --image 둘 다 필요합니다.")
