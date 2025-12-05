#!/usr/bin/env python3
"""
SAM (Segment Anything Model) Wrapper for Skin Lesion Segmentation

피부 병변 세그멘테이션을 위한 SAM 래퍼 모듈

전략:
- center: 이미지 중앙 포인트 기반 세그멘테이션
- lesion_features: 색상/텍스처 기반 병변 영역 탐지
- both: center + lesion_features 결합
- llm_guided: VLM이 제안한 좌표 기반 세그멘테이션

Author: DermAgent Team
"""

import os
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from PIL import Image

# SAM 관련 import (선택적)
SAM_AVAILABLE = False
try:
    import torch
    from segment_anything import sam_model_registry, SamPredictor
    SAM_AVAILABLE = True
except ImportError:
    pass


def load_image(image_path: str) -> np.ndarray:
    """이미지를 numpy 배열로 로드"""
    img = Image.open(image_path).convert("RGB")
    return np.array(img)


def apply_mask_to_image(
    image: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.5,
    color: Tuple[int, int, int] = (255, 0, 0)
) -> np.ndarray:
    """
    이미지에 마스크 오버레이 적용

    Args:
        image: 원본 이미지 (H, W, 3)
        mask: 바이너리 마스크 (H, W)
        alpha: 오버레이 투명도
        color: 마스크 색상 (R, G, B)

    Returns:
        마스크가 오버레이된 이미지
    """
    result = image.copy()

    # 마스크 영역에 색상 오버레이
    overlay = np.zeros_like(image)
    overlay[mask > 0] = color

    # 알파 블렌딩
    mask_3d = np.stack([mask] * 3, axis=-1)
    result = np.where(
        mask_3d > 0,
        (1 - alpha) * result + alpha * overlay,
        result
    ).astype(np.uint8)

    return result


def create_segmentation_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    highlight_color: Tuple[int, int, int] = (0, 255, 0),
    boundary_color: Tuple[int, int, int] = (255, 255, 0),
    alpha: float = 0.3
) -> np.ndarray:
    """
    세그멘테이션 결과를 시각화하는 오버레이 생성

    Args:
        image: 원본 이미지
        mask: 세그멘테이션 마스크
        highlight_color: 마스크 영역 하이라이트 색상
        boundary_color: 경계선 색상
        alpha: 투명도

    Returns:
        오버레이된 이미지
    """
    result = apply_mask_to_image(image, mask, alpha, highlight_color)

    # 경계선 추가 (선택적)
    try:
        import cv2
        contours, _ = cv2.findContours(
            mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(result, contours, -1, boundary_color, 2)
    except ImportError:
        pass

    return result


class SAMSegmenter:
    """SAM 기반 세그멘테이션 클래스"""

    # SAM 모델 유형별 설정
    MODEL_CONFIGS = {
        "vit_h": {"checkpoint": "sam_vit_h_4b8939.pth", "size": "huge"},
        "vit_l": {"checkpoint": "sam_vit_l_0b3195.pth", "size": "large"},
        "vit_b": {"checkpoint": "sam_vit_b_01ec64.pth", "size": "base"},
    }

    def __init__(
        self,
        model_type: str = "vit_b",
        checkpoint_dir: Optional[str] = None,
        device: str = "auto"
    ):
        """
        Args:
            model_type: SAM 모델 유형 (vit_h, vit_l, vit_b)
            checkpoint_dir: 체크포인트 디렉터리 경로
            device: cuda/cpu/auto
        """
        self.model_type = model_type
        self.device = self._get_device(device)
        self.predictor = None
        self.checkpoint_dir = checkpoint_dir

        if SAM_AVAILABLE:
            self._load_model()

    def _get_device(self, device: str) -> str:
        """디바이스 자동 선택"""
        if device == "auto":
            if SAM_AVAILABLE and torch.cuda.is_available():
                return "cuda"
            return "cpu"
        return device

    def _load_model(self):
        """SAM 모델 로드"""
        if not SAM_AVAILABLE:
            print("[WARNING] SAM not available. Install with: pip install segment-anything")
            return

        # 체크포인트 경로 찾기
        checkpoint_name = self.MODEL_CONFIGS[self.model_type]["checkpoint"]
        checkpoint_paths = [
            Path(self.checkpoint_dir) / checkpoint_name if self.checkpoint_dir else None,
            Path.home() / ".cache" / "sam" / checkpoint_name,
            Path("/content") / "sam_checkpoints" / checkpoint_name,  # Colab
            Path("./checkpoints") / checkpoint_name,
        ]

        checkpoint_path = None
        for path in checkpoint_paths:
            if path and path.exists():
                checkpoint_path = str(path)
                break

        if checkpoint_path is None:
            print(f"[WARNING] SAM checkpoint not found: {checkpoint_name}")
            print("  Download from: https://github.com/facebookresearch/segment-anything#model-checkpoints")
            return

        try:
            sam = sam_model_registry[self.model_type](checkpoint=checkpoint_path)
            sam.to(device=self.device)
            self.predictor = SamPredictor(sam)
            print(f"[INFO] SAM loaded: {self.model_type} on {self.device}")
        except Exception as e:
            print(f"[ERROR] SAM load failed: {e}")

    def set_image(self, image: Union[np.ndarray, str]):
        """세그멘테이션을 위한 이미지 설정"""
        if self.predictor is None:
            return False

        if isinstance(image, str):
            image = load_image(image)

        self.predictor.set_image(image)
        self.current_image = image
        return True

    def segment_point(
        self,
        point: Tuple[int, int],
        label: int = 1
    ) -> Dict[str, Any]:
        """
        단일 포인트 기반 세그멘테이션

        Args:
            point: (x, y) 좌표
            label: 1 = foreground, 0 = background

        Returns:
            세그멘테이션 결과 딕셔너리
        """
        if self.predictor is None:
            return {"mask": None, "score": 0.0, "method": "point"}

        point_coords = np.array([[point[0], point[1]]])
        point_labels = np.array([label])

        masks, scores, _ = self.predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=True
        )

        # 가장 높은 점수의 마스크 선택
        best_idx = np.argmax(scores)

        return {
            "mask": masks[best_idx],
            "score": float(scores[best_idx]),
            "method": "point",
            "point": point
        }

    def segment_center_focused(self, image: Union[np.ndarray, str]) -> Dict[str, Any]:
        """
        이미지 중앙 기반 세그멘테이션
        피부과 이미지에서 병변이 중앙에 있다고 가정
        """
        if isinstance(image, str):
            image = load_image(image)

        if self.predictor is None:
            return {"mask": None, "score": 0.0, "method": "center_focused"}

        self.set_image(image)

        h, w = image.shape[:2]
        center = (w // 2, h // 2)

        # 중앙 및 주변 포인트들
        points = [
            center,
            (w // 2 - w // 8, h // 2),
            (w // 2 + w // 8, h // 2),
            (w // 2, h // 2 - h // 8),
            (w // 2, h // 2 + h // 8),
        ]

        best_result = {"mask": None, "score": 0.0}

        for point in points:
            result = self.segment_point(point, label=1)
            if result["score"] > best_result["score"]:
                best_result = result

        best_result["method"] = "center_focused"
        return best_result

    def segment_lesion_features(self, image: Union[np.ndarray, str]) -> Dict[str, Any]:
        """
        병변 특징 기반 세그멘테이션
        색상 분석을 통해 병변 영역 탐지
        """
        if isinstance(image, str):
            image = load_image(image)

        if self.predictor is None:
            # SAM 없이 색상 기반 마스크 생성
            return self._color_based_segmentation(image)

        self.set_image(image)

        # 색상 분석으로 병변 후보 영역 찾기
        candidate_points = self._find_lesion_candidates(image)

        if not candidate_points:
            return self.segment_center_focused(image)

        best_result = {"mask": None, "score": 0.0}

        for point in candidate_points[:5]:  # 상위 5개 후보
            result = self.segment_point(point, label=1)
            if result["score"] > best_result["score"]:
                best_result = result

        best_result["method"] = "lesion_features"
        return best_result

    def _find_lesion_candidates(self, image: np.ndarray) -> List[Tuple[int, int]]:
        """
        색상 분석으로 병변 후보 포인트 찾기
        피부색과 다른 영역을 병변으로 추정
        """
        try:
            import cv2

            # HSV 변환
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

            # 피부색이 아닌 영역 찾기 (간단한 휴리스틱)
            # 피부색 범위: H=0-50, S=40-255, V=80-255
            lower_skin = np.array([0, 40, 80])
            upper_skin = np.array([50, 255, 255])

            skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
            lesion_mask = cv2.bitwise_not(skin_mask)

            # 노이즈 제거
            kernel = np.ones((5, 5), np.uint8)
            lesion_mask = cv2.morphologyEx(lesion_mask, cv2.MORPH_OPEN, kernel)
            lesion_mask = cv2.morphologyEx(lesion_mask, cv2.MORPH_CLOSE, kernel)

            # 컨투어 찾기
            contours, _ = cv2.findContours(
                lesion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            # 면적 기준 정렬 후 중심점 추출
            candidates = []
            for contour in sorted(contours, key=cv2.contourArea, reverse=True):
                M = cv2.moments(contour)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    candidates.append((cx, cy))

            return candidates

        except ImportError:
            # cv2 없으면 중앙 반환
            h, w = image.shape[:2]
            return [(w // 2, h // 2)]

    def _color_based_segmentation(self, image: np.ndarray) -> Dict[str, Any]:
        """SAM 없이 색상 기반 세그멘테이션"""
        try:
            import cv2

            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

            # 여러 색상 범위 시도 (빨간색, 갈색 병변)
            masks = []

            # 빨간색 계열
            lower_red1 = np.array([0, 50, 50])
            upper_red1 = np.array([10, 255, 255])
            mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)

            lower_red2 = np.array([160, 50, 50])
            upper_red2 = np.array([180, 255, 255])
            mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)

            # 갈색/어두운 영역
            lower_brown = np.array([10, 50, 20])
            upper_brown = np.array([30, 255, 200])
            mask_brown = cv2.inRange(hsv, lower_brown, upper_brown)

            # 마스크 결합
            combined_mask = cv2.bitwise_or(mask_red1, mask_red2)
            combined_mask = cv2.bitwise_or(combined_mask, mask_brown)

            # 모폴로지 연산
            kernel = np.ones((5, 5), np.uint8)
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)

            score = np.sum(combined_mask > 0) / combined_mask.size

            return {
                "mask": combined_mask > 0,
                "score": min(score * 2, 1.0),  # 정규화
                "method": "color_based"
            }

        except ImportError:
            # cv2 없으면 전체 이미지 마스크
            return {
                "mask": np.ones(image.shape[:2], dtype=bool),
                "score": 0.5,
                "method": "fallback"
            }

    def segment_with_box(
        self,
        image: Union[np.ndarray, str],
        box: Tuple[int, int, int, int]
    ) -> Dict[str, Any]:
        """
        바운딩 박스 기반 세그멘테이션

        Args:
            image: 입력 이미지
            box: (x1, y1, x2, y2) 바운딩 박스
        """
        if isinstance(image, str):
            image = load_image(image)

        if self.predictor is None:
            return {"mask": None, "score": 0.0, "method": "box"}

        self.set_image(image)

        input_box = np.array(box)

        masks, scores, _ = self.predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_box[None, :],
            multimask_output=True
        )

        best_idx = np.argmax(scores)

        return {
            "mask": masks[best_idx],
            "score": float(scores[best_idx]),
            "method": "box",
            "box": box
        }


class MockSAMSegmenter:
    """SAM 없이 테스트용 모의 세그멘터"""

    def __init__(self, *args, **kwargs):
        print("[INFO] MockSAMSegmenter: SAM 없이 색상 기반 세그멘테이션 사용")

    def segment_center_focused(self, image: Union[np.ndarray, str]) -> Dict[str, Any]:
        if isinstance(image, str):
            image = load_image(image)

        h, w = image.shape[:2]
        # 중앙 원형 마스크
        y, x = np.ogrid[:h, :w]
        center_y, center_x = h // 2, w // 2
        radius = min(h, w) // 4
        mask = ((x - center_x) ** 2 + (y - center_y) ** 2) <= radius ** 2

        return {
            "mask": mask,
            "score": 0.7,
            "method": "mock_center"
        }

    def segment_lesion_features(self, image: Union[np.ndarray, str]) -> Dict[str, Any]:
        if isinstance(image, str):
            image = load_image(image)

        # 색상 기반 세그멘테이션 시도
        try:
            import cv2
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

            # 채도가 높은 영역 추출
            mask = hsv[:, :, 1] > 50

            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

            return {
                "mask": mask > 0,
                "score": 0.6,
                "method": "mock_color"
            }
        except ImportError:
            return self.segment_center_focused(image)


# 테스트 코드
if __name__ == "__main__":
    print(f"SAM Available: {SAM_AVAILABLE}")

    # 테스트 이미지 생성
    test_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)

    if SAM_AVAILABLE:
        segmenter = SAMSegmenter(model_type="vit_b")
    else:
        segmenter = MockSAMSegmenter()

    result = segmenter.segment_center_focused(test_image)
    print(f"Center focused result: score={result['score']:.3f}, method={result['method']}")

    result = segmenter.segment_lesion_features(test_image)
    print(f"Lesion features result: score={result['score']:.3f}, method={result['method']}")
