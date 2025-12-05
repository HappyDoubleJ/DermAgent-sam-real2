#!/usr/bin/env python3
"""
VLM (Vision-Language Model) Wrapper

GPT-4o API 래퍼 모듈
증상 통합 실험 스크립트에서 사용

Author: DermAgent Team
"""

import os
import base64
from pathlib import Path
from typing import List, Optional, Union
from openai import OpenAI


def encode_image(image_path: str) -> str:
    """이미지를 base64로 인코딩"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


class GPT4oWrapper:
    """GPT-4o Vision API 래퍼"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o",
        use_labels_prompt: bool = True,
        max_tokens: int = 1024
    ):
        """
        Args:
            api_key: OpenAI API 키 (없으면 환경변수에서 가져옴)
            model: 사용할 모델 (기본: gpt-4o)
            use_labels_prompt: 질병 레이블 목록을 프롬프트에 포함할지 여부
            max_tokens: 최대 출력 토큰 수
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY가 필요합니다.")

        self.client = OpenAI(api_key=self.api_key)
        self.model = model
        self.max_tokens = max_tokens
        self.use_labels_prompt = use_labels_prompt

        # 질병 레이블 로드 (선택적)
        self.disease_labels = []
        if use_labels_prompt:
            self._load_disease_labels()

        # 시스템 프롬프트
        self.system_prompt = self._build_system_prompt()

    def _load_disease_labels(self):
        """질병 레이블 목록 로드"""
        script_dir = Path(__file__).parent
        label_paths = [
            script_dir.parent / "baseline" / "extracted_node_names.txt",
            script_dir / "extracted_node_names.txt",
        ]

        for path in label_paths:
            if path.exists():
                with open(path, "r") as f:
                    self.disease_labels = [
                        line.strip().split("→")[1] if "→" in line else line.strip()
                        for line in f.readlines() if line.strip()
                    ]
                break

    def _build_system_prompt(self) -> str:
        """시스템 프롬프트 생성"""
        base_prompt = """You are a dermatology expert. You are provided with a skin image and a question about it.
Please analyze the image carefully and provide a detailed diagnosis or answer based on your expertise.
Focus on identifying skin conditions, lesions, or abnormalities visible in the image."""

        if self.use_labels_prompt and self.disease_labels:
            labels_str = ", ".join(self.disease_labels[:100])  # 처음 100개만
            base_prompt += f"\n\nWhen identifying skin conditions, consider these possible diagnoses: {labels_str}"

        base_prompt += "\n\nProvide a clear and professional response."
        return base_prompt

    def analyze_image(
        self,
        image_paths: Union[str, List[str]],
        prompt: str,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        이미지 분석

        Args:
            image_paths: 이미지 경로 (단일 또는 리스트)
            prompt: 사용자 프롬프트
            system_prompt: 커스텀 시스템 프롬프트 (선택)

        Returns:
            모델 응답 텍스트
        """
        if isinstance(image_paths, str):
            image_paths = [image_paths]

        # 메시지 구성
        messages = [
            {"role": "system", "content": system_prompt or self.system_prompt},
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}]
            }
        ]

        # 이미지 추가
        for img_path in image_paths:
            if os.path.exists(img_path):
                base64_image = encode_image(img_path)
                messages[1]["content"].append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64_image}"
                    }
                })

        # API 호출
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error: {str(e)}"

    def diagnose(
        self,
        image_path: str,
        symptoms: Optional[str] = None,
        additional_context: Optional[str] = None
    ) -> str:
        """
        피부 질환 진단

        Args:
            image_path: 이미지 경로
            symptoms: 증상 정보 (선택)
            additional_context: 추가 컨텍스트 (선택)

        Returns:
            진단 결과 텍스트
        """
        prompt_parts = ["Please diagnose the skin condition shown in this image."]

        if symptoms:
            prompt_parts.append(f"\n\nPatient-reported symptoms: {symptoms}")

        if additional_context:
            prompt_parts.append(f"\n\nAdditional context: {additional_context}")

        prompt_parts.append("\n\nProvide your diagnosis with reasoning.")

        return self.analyze_image(image_path, "".join(prompt_parts))

    def extract_diagnosis_label(self, response: str) -> str:
        """
        응답에서 진단명 추출 (간단한 휴리스틱)

        Args:
            response: 모델 응답

        Returns:
            추출된 진단명
        """
        # 일반적인 패턴으로 진단명 추출 시도
        import re

        patterns = [
            r"diagnosis[:\s]+([A-Za-z\s]+)",
            r"condition[:\s]+([A-Za-z\s]+)",
            r"appears to be[:\s]+([A-Za-z\s]+)",
            r"likely[:\s]+([A-Za-z\s]+)",
        ]

        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                return match.group(1).strip()

        # 패턴 매칭 실패 시 응답 첫 줄 반환
        first_line = response.split('\n')[0]
        return first_line[:100] if len(first_line) > 100 else first_line

    def chat(self, prompt: str, image_paths: Optional[List[str]] = None) -> str:
        """
        일반 채팅 인터페이스

        Args:
            prompt: 사용자 프롬프트
            image_paths: 이미지 경로 리스트 (선택)

        Returns:
            모델 응답
        """
        if image_paths:
            return self.analyze_image(image_paths, prompt)

        # 텍스트만 있는 경우
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt}
        ]

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error: {str(e)}"


# 테스트 코드
if __name__ == "__main__":
    import sys

    # API 키 확인
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY 환경변수를 설정하세요.")
        sys.exit(1)

    # 래퍼 초기화
    vlm = GPT4oWrapper(api_key=api_key, use_labels_prompt=False)

    # 간단한 테스트
    print("VLM Wrapper 초기화 성공!")
    print(f"Model: {vlm.model}")
    print(f"Max tokens: {vlm.max_tokens}")
