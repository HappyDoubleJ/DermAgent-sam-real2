"""
SAM Wrapper for Experiment Framework

run_comparison_experiment.py에서 SAM 기반 진단 방법을 사용할 수 있도록
baseline_sam.py의 SAMDiagnosisModel을 래핑

Author: DermAgent Team
"""

import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any, List
import numpy as np
from PIL import Image
from io import BytesIO
import base64
import json

# 경로 설정
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR.parent / "SA-project-SAM"))
sys.path.insert(0, str(SCRIPT_DIR.parent / "baseline"))

# SAM 모듈 import
SAM_AVAILABLE = False
try:
    from sam_masking import (
        SAMSegmenter,
        SAM2Segmenter,
        MedSAM2Segmenter,
        apply_mask_to_image,
        load_image
    )
    SAM_AVAILABLE = True
except ImportError as e:
    print(f"[SAM Wrapper] SAM 모듈 임포트 실패: {e}")

# LLM-Guided Pipeline import
LLM_GUIDED_AVAILABLE = False
try:
    from llm_guided_pipeline import LLMGuidedSegmenter
    LLM_GUIDED_AVAILABLE = True
except ImportError:
    pass


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


class SAMBaselineWrapper:
    """
    SAM 기반 진단을 위한 래퍼 클래스

    run_comparison_experiment.py의 방법 함수에서 사용
    """

    DIAGNOSIS_PROMPT = """You are a dermatology expert. Analyze the provided skin images.

You are given:
1. Original skin image
2. Segmented view with the lesion area highlighted (red overlay)

Based on both images, provide a diagnosis.

IMPORTANT: If no clear skin lesion is visible, the image does not show identifiable human skin, or you cannot make a confident diagnosis, set disease_label to "no definitive diagnosis".

Respond in JSON format ONLY:
{
    "disease_label": "specific skin disease diagnosis (or 'no definitive diagnosis' if uncertain)",
    "body_location": "anatomical location where the condition appears",
    "caption": "detailed description of the skin condition",
    "confidence": 0.0-1.0
}"""

    def __init__(
        self,
        vlm_model,
        segmenter_type: str = "sam",
        segmentation_strategy: str = "center",
        checkpoint_dir: Optional[str] = None,
        verbose: bool = False
    ):
        """
        초기화

        Args:
            vlm_model: VLM 모델 (GPT4oWrapper 등)
            segmenter_type: 세그멘터 종류 (sam, sam2, medsam2)
            segmentation_strategy: 세그멘테이션 전략 (center, lesion_features, both, llm_guided)
            checkpoint_dir: 체크포인트 저장 디렉터리
            verbose: 상세 로그 출력
        """
        self.vlm = vlm_model
        self.verbose = verbose
        self.segmentation_strategy = segmentation_strategy
        self.segmenter_type = segmenter_type

        if checkpoint_dir is None:
            checkpoint_dir = str(SCRIPT_DIR / "checkpoints")

        # 세그멘터 초기화
        self.segmenter = self._init_segmenter(segmenter_type, checkpoint_dir)

        # LLM-Guided Pipeline (선택적)
        self.llm_guided = None
        if segmentation_strategy == "llm_guided" and LLM_GUIDED_AVAILABLE and hasattr(vlm_model, 'client'):
            # VLM에서 API 키 추출 시도
            api_key = getattr(vlm_model.client, 'api_key', None) or os.environ.get("OPENAI_API_KEY")
            if api_key:
                self.llm_guided = LLMGuidedSegmenter(api_key=api_key)

    def _init_segmenter(self, segmenter_type: str, checkpoint_dir: str):
        """세그멘터 초기화"""
        if not SAM_AVAILABLE:
            if self.verbose:
                print("[SAM Wrapper] SAM 모듈을 사용할 수 없습니다.")
            return None

        segmenter_type = segmenter_type.lower()

        if self.verbose:
            print(f"[SAM Wrapper] {segmenter_type.upper()} 세그멘터 초기화 중...")

        try:
            if segmenter_type == "sam":
                return SAMSegmenter(checkpoint_dir=checkpoint_dir)
            elif segmenter_type == "sam2":
                return SAM2Segmenter(checkpoint_dir=checkpoint_dir)
            elif segmenter_type == "medsam2":
                segmenter = MedSAM2Segmenter(checkpoint_dir=checkpoint_dir)
                if not segmenter.is_available():
                    if self.verbose:
                        print("[SAM Wrapper] MedSAM2를 사용할 수 없습니다. SAM으로 대체합니다.")
                    return SAMSegmenter(checkpoint_dir=checkpoint_dir)
                return segmenter
            else:
                if self.verbose:
                    print(f"[SAM Wrapper] 알 수 없는 세그멘터: {segmenter_type}. SAM으로 대체합니다.")
                return SAMSegmenter(checkpoint_dir=checkpoint_dir)
        except Exception as e:
            if self.verbose:
                print(f"[SAM Wrapper] 세그멘터 초기화 실패: {e}")
            return None

    def _segment_image(self, image: np.ndarray) -> Dict[str, Any]:
        """이미지 세그멘테이션 수행"""
        if self.segmenter is None:
            return {"mask": None, "score": 0.0, "method": "none"}

        strategy = self.segmentation_strategy

        try:
            if strategy == "center":
                return self.segmenter.segment_center_focused(image)
            elif strategy == "lesion_features":
                return self.segmenter.segment_lesion_features(image)
            elif strategy == "both":
                center_result = self.segmenter.segment_center_focused(image)
                feature_result = self.segmenter.segment_lesion_features(image)
                if feature_result["score"] > center_result["score"]:
                    return feature_result
                return center_result
            else:
                return self.segmenter.segment_center_focused(image)
        except Exception as e:
            if self.verbose:
                print(f"[SAM Wrapper] 세그멘테이션 실패: {e}")
            return {"mask": None, "score": 0.0, "method": "error", "error": str(e)}

    def _create_combined_prompt(self, has_segmentation: bool = True) -> str:
        """세그멘테이션 유무에 따른 프롬프트 생성"""
        if has_segmentation:
            return self.DIAGNOSIS_PROMPT
        else:
            return """You are a dermatology expert. Analyze this skin image.

IMPORTANT: If no clear skin lesion is visible, the image does not show identifiable human skin, or you cannot make a confident diagnosis, set disease_label to "no definitive diagnosis".

Respond in JSON format ONLY:
{
    "disease_label": "specific skin disease diagnosis (or 'no definitive diagnosis' if uncertain)",
    "body_location": "anatomical location where the condition appears",
    "caption": "detailed description of the skin condition",
    "confidence": 0.0-1.0
}"""

    def _parse_json_response(self, response: str) -> Dict:
        """JSON 응답 파싱"""
        if response is None:
            return {}
        try:
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start != -1 and json_end > json_start:
                return json.loads(response[json_start:json_end])
        except (json.JSONDecodeError, TypeError):
            pass
        return {}

    def analyze_image(self, image_path: str) -> Dict:
        """
        단일 이미지 분석

        Args:
            image_path: 이미지 경로

        Returns:
            분석 결과 딕셔너리
        """
        if not os.path.exists(image_path):
            return {
                "disease_label": "no definitive diagnosis",
                "body_location": "N/A",
                "caption": "Image file does not exist",
                "confidence": 0.0,
                "segmentation_score": 0.0
            }

        try:
            image = load_image_as_array(image_path)
        except Exception as e:
            return {
                "disease_label": "no definitive diagnosis",
                "body_location": "N/A",
                "caption": f"Failed to load image: {e}",
                "confidence": 0.0,
                "segmentation_score": 0.0
            }

        # LLM-Guided 파이프라인 사용
        if self.segmentation_strategy == "llm_guided" and self.llm_guided is not None:
            return self._analyze_with_llm_guided(image, image_path)

        # 세그멘테이션
        seg_result = self._segment_image(image)

        if seg_result.get("mask") is not None:
            # 오버레이 생성
            overlay = apply_mask_to_image(image, seg_result["mask"], color=(255, 0, 0), alpha=0.4)
            prompt = self._create_combined_prompt(has_segmentation=True)
            # 두 이미지를 함께 전송
            response = self.vlm.chat_img(prompt, [image_path])
        else:
            # 세그멘테이션 실패 시 원본만으로 진단
            prompt = self._create_combined_prompt(has_segmentation=False)
            response = self.vlm.chat_img(prompt, [image_path])

        parsed = self._parse_json_response(response)

        return {
            "disease_label": parsed.get("disease_label", "no definitive diagnosis"),
            "body_location": parsed.get("body_location", "Unknown"),
            "caption": parsed.get("caption", ""),
            "confidence": parsed.get("confidence", 0.5),
            "segmentation_score": seg_result.get("score", 0.0),
            "segmentation_method": seg_result.get("method", "unknown"),
            "raw_response": response
        }

    def _analyze_with_llm_guided(self, image: np.ndarray, image_path: str) -> Dict:
        """LLM-Guided 파이프라인으로 분석"""
        try:
            result = self.llm_guided.run_full_pipeline(
                image=image,
                segmenter=self.segmenter,
                save_results=False
            )

            diagnosis = result.get("diagnosis_result", {})
            conditions = diagnosis.get("possible_conditions", [])

            if conditions and isinstance(conditions, list) and len(conditions) > 0:
                top_condition = conditions[0]
                disease_label = top_condition.get("name", "Unknown")
                confidence_str = top_condition.get("confidence", "Medium")
                conf_map = {"High": 0.9, "Medium": 0.6, "Low": 0.3}
                confidence = conf_map.get(confidence_str, 0.5)
            else:
                disease_label = diagnosis.get("primary_impression", "no definitive diagnosis")
                confidence = 0.5

            seg_results = result.get("segmentation_results", [])
            avg_score = 0.0
            if seg_results:
                scores = [s.get("score", 0) for s in seg_results if "score" in s]
                avg_score = sum(scores) / len(scores) if scores else 0.0

            return {
                "disease_label": disease_label,
                "body_location": diagnosis.get("observed_features", {}).get("distribution", "Unknown"),
                "caption": str(diagnosis.get("observed_features", {})),
                "confidence": confidence,
                "segmentation_score": avg_score,
                "segmentation_method": "llm_guided",
                "lesion_count": result.get("location_result", {}).get("lesion_count", 0)
            }

        except Exception as e:
            return {
                "disease_label": "no definitive diagnosis",
                "body_location": "N/A",
                "caption": f"LLM-Guided pipeline error: {e}",
                "confidence": 0.0,
                "segmentation_score": 0.0,
                "segmentation_method": "llm_guided_error"
            }

    @staticmethod
    def is_available() -> bool:
        """SAM 모듈 사용 가능 여부"""
        return SAM_AVAILABLE


def test_sam_wrapper():
    """SAM Wrapper 테스트"""
    print("=" * 60)
    print("SAM Wrapper Test")
    print("=" * 60)

    print(f"SAM Available: {SAM_AVAILABLE}")
    print(f"LLM-Guided Available: {LLM_GUIDED_AVAILABLE}")

    if not SAM_AVAILABLE:
        print("\n[경고] SAM 모듈을 사용할 수 없습니다.")
        print("설치 방법:")
        print("  pip install git+https://github.com/facebookresearch/segment-anything.git")
        return

    # VLM 없이 세그멘터만 테스트
    print("\n세그멘터 초기화 테스트...")
    try:
        wrapper = SAMBaselineWrapper(
            vlm_model=None,
            segmenter_type="sam",
            segmentation_strategy="center",
            verbose=True
        )
        print("  세그멘터 초기화 성공!")
    except Exception as e:
        print(f"  세그멘터 초기화 실패: {e}")


if __name__ == "__main__":
    test_sam_wrapper()
