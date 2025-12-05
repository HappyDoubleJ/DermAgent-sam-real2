"""
SAM (Segment Anything Model) 기반 피부질환 진단 Baseline

SA-project-SAM의 파이프라인을 활용한 진단 시스템:
1. SAM/SAM2/MedSAM2로 병변 영역 세그멘테이션
2. 세그멘테이션 결과와 함께 VLM에 진단 요청
3. CSV 형식으로 결과 저장

지원하는 파이프라인:
- Center-focused: 이미지 중앙 기반 세그멘테이션
- Lesion-feature-based: 병변 특징 기반 세그멘테이션
- LLM-guided: LLM이 병변 위치를 가이드하고 SAM이 분할

Author: DermAgent Team
"""

import os
import sys
import argparse
import csv
import json
from pathlib import Path
from typing import Optional, Dict, List, Any
from tqdm import tqdm
import numpy as np
from PIL import Image
from io import BytesIO
import base64

# 경로 설정
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPT_DIR.parent / "SA-project-SAM"))
sys.path.insert(0, str(SCRIPT_DIR.parent / "experiments"))

from project_path import SAMPLED_DATA_CSV, OUTPUTS_ROOT, DERM1M_ROOT

# .env 파일 로드
def load_env():
    """Load environment variables from .env file"""
    env_paths = [
        SCRIPT_DIR / ".env",
        SCRIPT_DIR.parent / ".env",
        PROJECT_ROOT / ".env",
    ]
    for env_path in env_paths:
        if env_path.exists():
            with open(env_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        key = key.strip()
                        value = value.strip().strip('"').strip("'")
                        if key not in os.environ:
                            os.environ[key] = value
            return True
    return False

load_env()


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
    print(f"[경고] SAM 모듈 임포트 실패: {e}")

# LLM-Guided Pipeline import
LLM_GUIDED_AVAILABLE = False
try:
    from llm_guided_pipeline import LLMGuidedSegmenter
    LLM_GUIDED_AVAILABLE = True
except ImportError as e:
    print(f"[경고] LLM-Guided Pipeline 임포트 실패: {e}")

# OpenAI
OPENAI_AVAILABLE = False
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    pass


def load_openai_key(arg_key: Optional[str] = None) -> Optional[str]:
    """Load OPENAI_API_KEY from arg, env, or project .env."""
    if arg_key:
        return arg_key
    env_key = os.environ.get("OPENAI_API_KEY")
    if env_key:
        return env_key
    return None


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


class SAMDiagnosisModel:
    """
    SAM + VLM 기반 피부질환 진단 모델

    세그멘테이션 결과를 VLM에 함께 제공하여 진단 정확도 향상
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
    "caption": "detailed description of the skin condition and segmentation quality",
    "confidence": 0.0-1.0,
    "segmentation_quality": "good/partial/poor - how well does the overlay capture the lesion?"
}"""

    def __init__(
        self,
        api_key: str,
        segmenter_type: str = "sam",
        segmentation_strategy: str = "center",
        checkpoint_dir: str = "checkpoints",
        verbose: bool = True
    ):
        """
        초기화

        Args:
            api_key: OpenAI API 키
            segmenter_type: 세그멘터 종류 (sam, sam2, medsam2)
            segmentation_strategy: 세그멘테이션 전략 (center, lesion_features, both, llm_guided)
            checkpoint_dir: 체크포인트 저장 디렉터리
            verbose: 상세 로그 출력
        """
        self.api_key = api_key
        self.client = OpenAI(api_key=api_key)
        self.model = "gpt-4o"
        self.verbose = verbose
        self.segmentation_strategy = segmentation_strategy
        self.segmenter_type = segmenter_type

        # 세그멘터 초기화
        self.segmenter = self._init_segmenter(segmenter_type, checkpoint_dir)

        # LLM-Guided Pipeline (선택적)
        self.llm_guided = None
        if segmentation_strategy == "llm_guided" and LLM_GUIDED_AVAILABLE:
            self.llm_guided = LLMGuidedSegmenter(api_key=api_key)

    def _init_segmenter(self, segmenter_type: str, checkpoint_dir: str):
        """세그멘터 초기화"""
        if not SAM_AVAILABLE:
            print("[경고] SAM 모듈을 사용할 수 없습니다.")
            return None

        segmenter_type = segmenter_type.lower()

        if self.verbose:
            print(f"[INFO] {segmenter_type.upper()} 세그멘터 초기화 중...")

        try:
            if segmenter_type == "sam":
                return SAMSegmenter(checkpoint_dir=checkpoint_dir)
            elif segmenter_type == "sam2":
                return SAM2Segmenter(checkpoint_dir=checkpoint_dir)
            elif segmenter_type == "medsam2":
                segmenter = MedSAM2Segmenter(checkpoint_dir=checkpoint_dir)
                if not segmenter.is_available():
                    print("[경고] MedSAM2를 사용할 수 없습니다. SAM으로 대체합니다.")
                    return SAMSegmenter(checkpoint_dir=checkpoint_dir)
                return segmenter
            else:
                print(f"[경고] 알 수 없는 세그멘터: {segmenter_type}. SAM으로 대체합니다.")
                return SAMSegmenter(checkpoint_dir=checkpoint_dir)
        except Exception as e:
            print(f"[오류] 세그멘터 초기화 실패: {e}")
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
                # 두 전략 중 더 좋은 결과 선택
                center_result = self.segmenter.segment_center_focused(image)
                feature_result = self.segmenter.segment_lesion_features(image)
                if feature_result["score"] > center_result["score"]:
                    return feature_result
                return center_result
            else:
                # 기본: center
                return self.segmenter.segment_center_focused(image)
        except Exception as e:
            print(f"[오류] 세그멘테이션 실패: {e}")
            return {"mask": None, "score": 0.0, "method": "error", "error": str(e)}

    def _call_vlm(self, original: np.ndarray, overlay: np.ndarray) -> Dict:
        """VLM 호출 (원본 + 오버레이 이미지)"""
        base64_original = encode_image_to_base64(original)
        base64_overlay = encode_image_to_base64(overlay)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self.DIAGNOSIS_PROMPT},
                        {"type": "text", "text": "Image 1 - Original:"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_original}",
                                "detail": "high"
                            }
                        },
                        {"type": "text", "text": "Image 2 - Segmented overlay:"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_overlay}",
                                "detail": "high"
                            }
                        }
                    ]
                }],
                max_tokens=1024,
                temperature=0.3
            )

            content = response.choices[0].message.content
            if content is None:
                return {"disease_label": "no definitive diagnosis", "error": "Empty response"}

            # JSON 파싱
            return self._parse_json_response(content)

        except Exception as e:
            return {"disease_label": "Error", "error": str(e)}

    def _parse_json_response(self, response: str) -> Dict:
        """JSON 응답 파싱"""
        try:
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start != -1 and json_end > json_start:
                return json.loads(response[json_start:json_end])
        except json.JSONDecodeError:
            pass
        return {"disease_label": "Parse error", "raw_response": response}

    def analyze_image(self, image_path: str) -> Dict:
        """
        단일 이미지 분석

        Args:
            image_path: 이미지 경로

        Returns:
            분석 결과 딕셔너리
        """
        # 이미지 로드
        if not os.path.exists(image_path):
            return {
                "disease_label": "Image not found",
                "body_location": "N/A",
                "caption": "Image file does not exist",
                "confidence": 0.0,
                "segmentation_score": 0.0
            }

        try:
            image = load_image_as_array(image_path)
        except Exception as e:
            return {
                "disease_label": "Load error",
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

        if seg_result.get("mask") is None:
            # 세그멘테이션 실패 시 원본만으로 진단
            overlay = image
        else:
            # 오버레이 생성
            overlay = apply_mask_to_image(image, seg_result["mask"], color=(255, 0, 0), alpha=0.4)

        # VLM 호출
        vlm_result = self._call_vlm(image, overlay)

        # 결과 조합
        result = {
            "disease_label": vlm_result.get("disease_label", "Unknown"),
            "body_location": vlm_result.get("body_location", "Unknown"),
            "caption": vlm_result.get("caption", ""),
            "confidence": vlm_result.get("confidence", 0.5),
            "segmentation_score": seg_result.get("score", 0.0),
            "segmentation_method": seg_result.get("method", "unknown"),
            "segmentation_quality": vlm_result.get("segmentation_quality", "N/A")
        }

        return result

    def _analyze_with_llm_guided(self, image: np.ndarray, image_path: str) -> Dict:
        """LLM-Guided 파이프라인으로 분석"""
        try:
            result = self.llm_guided.run_full_pipeline(
                image=image,
                segmenter=self.segmenter,
                save_results=False
            )

            # 결과 추출
            diagnosis = result.get("diagnosis_result", {})
            conditions = diagnosis.get("possible_conditions", [])

            if conditions and isinstance(conditions, list) and len(conditions) > 0:
                top_condition = conditions[0]
                disease_label = top_condition.get("name", "Unknown")
                confidence_str = top_condition.get("confidence", "Medium")

                # confidence 문자열을 숫자로 변환
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
                "caption": diagnosis.get("observed_features", {}).get("color", "") + " " +
                          diagnosis.get("observed_features", {}).get("shape", ""),
                "confidence": confidence,
                "segmentation_score": avg_score,
                "segmentation_method": "llm_guided",
                "segmentation_quality": diagnosis.get("segmentation_quality", {}).get("accuracy", "N/A"),
                "lesion_count": result.get("location_result", {}).get("lesion_count", 0)
            }

        except Exception as e:
            return {
                "disease_label": "Error",
                "body_location": "N/A",
                "caption": f"LLM-Guided pipeline error: {e}",
                "confidence": 0.0,
                "segmentation_score": 0.0,
                "segmentation_method": "llm_guided_error"
            }


def process_csv(
    input_csv: str,
    output_csv: str,
    image_base_folder: str,
    model: SAMDiagnosisModel,
    save_visualizations: bool = False,
    vis_output_dir: Optional[str] = None
):
    """CSV 파일의 모든 이미지 처리"""

    print(f"[INFO] Reading CSV: {input_csv}")
    with open(input_csv, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"[INFO] Total images to process: {len(rows)}")

    results = []
    for row in tqdm(rows, desc="Processing images"):
        filename = row.get('filename', '')
        image_path = os.path.join(image_base_folder, filename)

        # 분석
        analysis = model.analyze_image(image_path)

        # 결과 저장
        result = {
            'filename': filename,
            'predicted_disease_label': analysis['disease_label'],
            'predicted_body_location': analysis['body_location'],
            'predicted_caption': analysis['caption'],
            'confidence': analysis.get('confidence', 0.5),
            'segmentation_score': analysis.get('segmentation_score', 0.0),
            'segmentation_method': analysis.get('segmentation_method', 'unknown'),
            'segmentation_quality': analysis.get('segmentation_quality', 'N/A'),
            'ground_truth_disease_label': row.get('disease_label', ''),
            'ground_truth_body_location': row.get('body_location', ''),
            'ground_truth_caption': row.get('caption', '')
        }
        results.append(result)

    # 출력 CSV 저장
    print(f"\n[INFO] Writing results to: {output_csv}")
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    fieldnames = [
        'filename',
        'predicted_disease_label',
        'predicted_body_location',
        'predicted_caption',
        'confidence',
        'segmentation_score',
        'segmentation_method',
        'segmentation_quality',
        'ground_truth_disease_label',
        'ground_truth_body_location',
        'ground_truth_caption'
    ]

    with open(output_csv, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"[INFO] Processing complete! Results saved to {output_csv}")
    return results


def main():
    parser = argparse.ArgumentParser(
        description="SAM 기반 피부질환 진단 Baseline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  # SAM + Center-focused 전략
  python baseline_sam.py --segmenter sam --strategy center

  # SAM2 + Lesion-features 전략
  python baseline_sam.py --segmenter sam2 --strategy lesion_features

  # LLM-Guided 파이프라인 (SAM)
  python baseline_sam.py --segmenter sam --strategy llm_guided

  # MedSAM2 사용
  python baseline_sam.py --segmenter medsam2 --strategy both
"""
    )

    parser.add_argument('--input_csv', default=SAMPLED_DATA_CSV,
                        help=f'Input CSV file (default: {SAMPLED_DATA_CSV})')
    parser.add_argument('--output_csv', default=None,
                        help='Output CSV file (default: auto-generated)')
    parser.add_argument('--image_base_folder', default=DERM1M_ROOT,
                        help=f'Base folder for images (default: {DERM1M_ROOT})')
    parser.add_argument('--api_key', default=None,
                        help='OpenAI API key')

    # SAM 관련 옵션
    parser.add_argument('--segmenter', choices=['sam', 'sam2', 'medsam2'], default='sam',
                        help='Segmenter to use (default: sam)')
    parser.add_argument('--strategy', choices=['center', 'lesion_features', 'both', 'llm_guided'],
                        default='center', help='Segmentation strategy (default: center)')
    parser.add_argument('--checkpoint_dir', default=str(SCRIPT_DIR / "checkpoints"),
                        help='Checkpoint directory for SAM models')

    # 기타 옵션
    parser.add_argument('--save_vis', action='store_true',
                        help='Save visualization images')
    parser.add_argument('--vis_dir', default=None,
                        help='Visualization output directory')
    parser.add_argument('--verbose', action='store_true',
                        help='Verbose output')

    args = parser.parse_args()

    # API 키 로드
    api_key = load_openai_key(args.api_key)
    if not api_key:
        print("[오류] OpenAI API 키가 필요합니다.")
        print("  방법 1: --api_key 인자로 전달")
        print("  방법 2: .env 파일에 OPENAI_API_KEY 설정")
        print("  방법 3: 환경변수 OPENAI_API_KEY 설정")
        sys.exit(1)

    # 출력 파일명 자동 생성
    if args.output_csv is None:
        output_name = f"sam_{args.segmenter}_{args.strategy}_predictions.csv"
        args.output_csv = os.path.join(OUTPUTS_ROOT, output_name)

    # SAM 모듈 확인
    if not SAM_AVAILABLE:
        print("[오류] SAM 모듈을 사용할 수 없습니다.")
        print("segment-anything 설치:")
        print("  pip install git+https://github.com/facebookresearch/segment-anything.git")
        sys.exit(1)

    # LLM-Guided 확인
    if args.strategy == "llm_guided" and not LLM_GUIDED_AVAILABLE:
        print("[경고] LLM-Guided Pipeline을 사용할 수 없습니다. 'center' 전략으로 대체합니다.")
        args.strategy = "center"

    print("=" * 60)
    print("SAM 기반 피부질환 진단 Baseline")
    print("=" * 60)
    print(f"  세그멘터: {args.segmenter.upper()}")
    print(f"  전략: {args.strategy}")
    print(f"  입력 CSV: {args.input_csv}")
    print(f"  출력 CSV: {args.output_csv}")
    print("=" * 60)

    # 모델 초기화
    model = SAMDiagnosisModel(
        api_key=api_key,
        segmenter_type=args.segmenter,
        segmentation_strategy=args.strategy,
        checkpoint_dir=args.checkpoint_dir,
        verbose=args.verbose
    )

    # CSV 처리
    process_csv(
        input_csv=args.input_csv,
        output_csv=args.output_csv,
        image_base_folder=args.image_base_folder,
        model=model,
        save_visualizations=args.save_vis,
        vis_output_dir=args.vis_dir
    )


if __name__ == "__main__":
    main()
