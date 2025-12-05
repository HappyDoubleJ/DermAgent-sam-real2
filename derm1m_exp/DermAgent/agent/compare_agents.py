"""
DermatologyAgent vs ReActDermatologyAgent 비교 스크립트

두 에이전트의 진단 결과를 비교하고 평가합니다.
"""

import os
import sys
import csv
import json
import random
import base64
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List

# 경로 설정
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(PROJECT_DIR / "eval"))

from dermatology_agent import DermatologyAgent
from react_agent import ReActDermatologyAgent
from evaluation_metrics import HierarchicalEvaluator


def setup_logging(output_dir: Path, agent_name: str) -> logging.Logger:
    """로깅 설정 (파일 전용, 터미널 출력 없음)"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = output_dir / f"{agent_name}_{timestamp}.log"

    logger = logging.getLogger(agent_name)
    logger.setLevel(logging.INFO)

    # 기존 핸들러 제거
    logger.handlers.clear()

    # 파일 핸들러만 사용 (터미널 출력 제거)
    fh = logging.FileHandler(log_file, encoding='utf-8')
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(fh)

    return logger, log_file


class GPT4oVLM:
    """GPT-4o Vision 모델"""

    def __init__(self, api_key: str, logger: Optional[logging.Logger] = None):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key)
        self.model = "gpt-4o"
        self.logger = logger

    def _encode_image(self, image_path: str) -> str:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def _log(self, message: str):
        if self.logger:
            self.logger.info(message)
        else:
            print(message)

    def chat_img(self, prompt: str, image_paths: list, max_tokens: int = 1024) -> str:
        image_path = image_paths[0] if image_paths else None

        if not image_path or not os.path.exists(image_path):
            self._log(f"Warning: Image not found: {image_path}")
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.7
            )
            return response.choices[0].message.content

        base64_image = self._encode_image(image_path)

        self._log(f"\n{'='*60}")
        self._log(f"GPT-4o Request")
        self._log(f"{'='*60}")
        self._log(f"Prompt (first 300 chars): {prompt[:300]}...")
        self._log(f"Image: {image_path}")
        self._log(f"{'='*60}\n")

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                }],
                max_tokens=max_tokens,
                temperature=0.7
            )

            answer = response.choices[0].message.content

            self._log(f"\n{'='*60}")
            self._log(f"GPT-4o Response")
            self._log(f"{'='*60}")
            self._log(answer)
            self._log(f"{'='*60}\n")

            return answer

        except Exception as e:
            self._log(f"GPT-4o Error: {e}")
            return "{}"


def load_random_sample(csv_path: str, seed: Optional[int] = None) -> Dict[str, Any]:
    """CSV에서 랜덤 샘플 로드"""
    if seed is not None:
        random.seed(seed)

    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    sample = random.choice(rows)
    return sample


def run_dermatology_agent(
    image_path: str,
    vlm: Any,
    logger: logging.Logger,
    output_dir: Path,
    model_name: str = "unknown"
) -> Dict[str, Any]:
    """DermatologyAgent 실행"""
    logger.info("\n" + "="*80)
    logger.info("RUNNING: DermatologyAgent (Fixed 5-Step Pipeline)")
    logger.info(f"Model: {model_name}")
    logger.info("="*80 + "\n")

    agent = DermatologyAgent(
        ontology_path=None,
        vlm_model=vlm,
        verbose=True
    )

    # verbose 출력을 로거로 리다이렉트
    original_log = agent._log
    agent._log = lambda msg: logger.info(f"[DermatologyAgent] {msg}")

    try:
        result = agent.diagnose(str(image_path), max_depth=4)

        # 메타데이터 추가
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_with_meta = {
            "metadata": {
                "model": model_name,
                "timestamp": timestamp,
                "agent": "DermatologyAgent",
                "image_path": str(image_path)
            },
            "result": result
        }

        # JSON 저장
        json_file = output_dir / f"dermatology_agent_{model_name}_{timestamp}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(result_with_meta, f, indent=2, ensure_ascii=False)

        logger.info(f"\nResult saved to: {json_file}")
        return result, json_file

    except Exception as e:
        logger.error(f"DermatologyAgent Error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {"error": str(e)}, None


def run_react_agent(
    image_path: str,
    vlm: Any,
    logger: logging.Logger,
    output_dir: Path,
    model_name: str = "unknown"
) -> Dict[str, Any]:
    """ReActDermatologyAgent 실행"""
    logger.info("\n" + "="*80)
    logger.info("RUNNING: ReActDermatologyAgent (Dynamic ReAct Pattern)")
    logger.info(f"Model: {model_name}")
    logger.info("="*80 + "\n")

    agent = ReActDermatologyAgent(
        ontology_path=None,
        vlm_model=vlm,
        max_steps=8,
        verbose=True
    )

    # verbose 출력을 로거로 리다이렉트
    original_log = agent._log
    agent._log = lambda msg, level="info": logger.info(f"[ReActAgent] {msg}")

    try:
        result = agent.diagnose(str(image_path))
        result_dict = result.to_dict()

        # 메타데이터 추가
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_with_meta = {
            "metadata": {
                "model": model_name,
                "timestamp": timestamp,
                "agent": "ReActDermatologyAgent",
                "image_path": str(image_path)
            },
            "result": result_dict
        }

        # JSON 저장
        json_file = output_dir / f"react_agent_{model_name}_{timestamp}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(result_with_meta, f, indent=2, ensure_ascii=False)

        logger.info(f"\nResult saved to: {json_file}")
        return result_dict, json_file

    except Exception as e:
        logger.error(f"ReActAgent Error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {"error": str(e)}, None


def format_path_from_ontology(path: List[str]) -> List[str]:
    """
    get_path_to_root() 결과를 포맷팅: root 제거 + 역순 (큰 범위 → 작은 범위)

    Args:
        path: get_path_to_root() 반환값 (예: ["노드", "parent", ..., "root"])

    Returns:
        포맷팅된 path (예: ["큰범위", ..., "노드"])
    """
    if not path:
        return []
    # root 제거
    filtered = [p for p in path if p != 'root']
    # 역순 (큰 범위 → 작은 범위)
    return list(reversed(filtered))


def format_path_remove_root(path: List[str]) -> List[str]:
    """
    이미 올바른 순서인 path에서 root만 제거

    Args:
        path: 이미 [큰 범위 → 작은 범위] 순서인 path

    Returns:
        root가 제거된 path
    """
    if not path:
        return []
    return [p for p in path if p != 'root']


def evaluate_results(
    gt_label: str,
    derm_result: Dict,
    react_result: Dict,
    logger: logging.Logger,
    model_name: str = "unknown"
) -> Dict[str, Any]:
    """평가 수행"""
    logger.info("\n" + "="*80)
    logger.info("EVALUATION RESULTS")
    logger.info(f"Model: {model_name}")
    logger.info("="*80 + "\n")

    evaluator = HierarchicalEvaluator()

    # Ground Truth 파싱
    gt_labels = [l.strip() for l in gt_label.split(',')]

    # Ground Truth 경로 계산
    gt_paths = {}
    for gt in gt_labels:
        path = evaluator.tree.get_path_to_root(gt)
        if path:
            gt_paths[gt] = format_path_from_ontology(path)

    # DermatologyAgent 예측 추출
    derm_pred = derm_result.get('final_diagnosis', [])
    if isinstance(derm_pred, str):
        derm_pred = [derm_pred]

    # ReActAgent 예측 추출
    react_pred = react_result.get('primary_diagnosis', '')
    react_differentials = react_result.get('differential_diagnoses', [])
    if react_pred:
        react_preds = [react_pred] + react_differentials
    else:
        react_preds = []

    # 평가
    derm_eval = evaluator.evaluate_single(gt_labels, derm_pred)
    react_eval = evaluator.evaluate_single(gt_labels, react_preds)

    logger.info(f"Ground Truth: {gt_labels}")
    for gt, path in gt_paths.items():
        logger.info(f"  GT Path ({gt}): {' → '.join(path)}")
    logger.info(f"\n--- DermatologyAgent ---")
    logger.info(f"Prediction: {derm_pred}")
    logger.info(f"Diagnosis Path: {derm_result.get('diagnosis_path', [])}")
    for key, value in derm_eval.items():
        if key != 'valid':
            if isinstance(value, float):
                logger.info(f"  {key}: {value:.4f}")
            else:
                logger.info(f"  {key}: {value}")

    logger.info(f"\n--- ReActDermatologyAgent ---")
    logger.info(f"Prediction: {react_preds}")
    logger.info(f"Ontology Path: {react_result.get('ontology_path', [])}")
    logger.info(f"Confidence: {react_result.get('confidence', 'N/A')}")
    for key, value in react_eval.items():
        if key != 'valid':
            if isinstance(value, float):
                logger.info(f"  {key}: {value:.4f}")
            else:
                logger.info(f"  {key}: {value}")

    # 비교
    logger.info(f"\n--- Comparison ---")
    if derm_eval.get('valid') and react_eval.get('valid'):
        derm_f1 = derm_eval.get('hierarchical_f1', 0)
        react_f1 = react_eval.get('hierarchical_f1', 0)

        if derm_f1 > react_f1:
            logger.info(f"Winner: DermatologyAgent (F1: {derm_f1:.4f} vs {react_f1:.4f})")
        elif react_f1 > derm_f1:
            logger.info(f"Winner: ReActAgent (F1: {react_f1:.4f} vs {derm_f1:.4f})")
        else:
            logger.info(f"Tie (F1: {derm_f1:.4f})")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return {
        'metadata': {
            'model': model_name,
            'timestamp': timestamp
        },
        'ground_truth': gt_labels,
        'ground_truth_paths': {gt: path for gt, path in gt_paths.items()},
        'dermatology_agent': {
            'prediction': derm_pred,
            'path': format_path_remove_root(derm_result.get('diagnosis_path', [])),
            'evaluation': derm_eval
        },
        'react_agent': {
            'prediction': react_preds,
            'path': format_path_from_ontology(react_result.get('ontology_path', [])),
            'evaluation': react_eval
        }
    }


def main():
    parser = argparse.ArgumentParser(description='Compare DermatologyAgent vs ReActAgent')
    parser.add_argument('--csv', type=str,
                       default='/home/work/wonjun/DermAgent/dataset/Derm1M/Derm1M_v2_pretrain_ontology_sampled_100.csv',
                       help='CSV file path')
    parser.add_argument('--image_dir', type=str,
                       default='/home/work/wonjun/DermAgent/dataset/Derm1M',
                       help='Image directory')
    parser.add_argument('--output_dir', type=str,
                       default=None,
                       help='Output directory (default: agent/results/)')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for sample selection')
    parser.add_argument('--row', type=int, default=None,
                       help='Specific row number to use (0-indexed)')
    args = parser.parse_args()

    # 출력 디렉터리 설정
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = SCRIPT_DIR / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 메인 로거 설정 (파일 전용, 터미널 출력 없음)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = "gpt-4o"
    main_log_file = output_dir / f"comparison_{model_name}_{timestamp}.log"

    # 터미널에 시작 메시지만 출력
    print(f"[{timestamp}] Starting comparison with model: {model_name}")
    print(f"Log file: {main_log_file}")

    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
        handlers=[
            logging.FileHandler(main_log_file, encoding='utf-8')
        ]
    )
    logger = logging.getLogger('main')

    logger.info("="*80)
    logger.info("AGENT COMPARISON: DermatologyAgent vs ReActDermatologyAgent")
    logger.info("="*80)
    logger.info(f"Timestamp: {timestamp}")
    logger.info(f"Model: {model_name}")
    logger.info(f"CSV: {args.csv}")
    logger.info(f"Output Dir: {output_dir}")
    logger.info(f"Seed: {args.seed}")
    logger.info("="*80 + "\n")

    # 샘플 로드
    if args.row is not None:
        with open(args.csv, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        sample = rows[args.row]
        logger.info(f"Using specified row: {args.row}")
    else:
        sample = load_random_sample(args.csv, args.seed)
        logger.info(f"Random sample selected (seed: {args.seed})")

    logger.info("\n" + "="*80)
    logger.info("SAMPLE INFO")
    logger.info("="*80)
    logger.info(f"Filename: {sample['filename']}")
    logger.info(f"Disease Label (GT): {sample.get('disease_label', 'N/A')}")
    logger.info(f"Hierarchical Label: {sample.get('hierarchical_disease_label', 'N/A')}")

    if 'caption' in sample and sample['caption']:
        caption = sample['caption'][:200] + "..." if len(sample['caption']) > 200 else sample['caption']
        logger.info(f"Caption: {caption}")

    logger.info("="*80 + "\n")

    # 이미지 경로
    image_path = Path(args.image_dir) / sample['filename']
    if not image_path.exists():
        logger.error(f"Image not found: {image_path}")
        sys.exit(1)
    logger.info(f"Image Path: {image_path}")
    logger.info(f"Image Exists: {image_path.exists()}\n")

    # VLM 초기화
    # API 키 로드
    env_path = Path(__file__).resolve().parent.parent.parent.parent / ".env"
    if env_path.exists():
        from dotenv import load_dotenv
        load_dotenv(env_path)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY not found.")
        sys.exit(1)

    logger.info(f"API Key loaded: {api_key[:4]}...")  # 앞자리만 로깅
    vlm = GPT4oVLM(api_key, logger)

    # 에이전트 실행
    derm_result, derm_json = run_dermatology_agent(image_path, vlm, logger, output_dir, model_name)
    react_result, react_json = run_react_agent(image_path, vlm, logger, output_dir, model_name)

    # 평가
    gt_label = sample.get('disease_label', '')
    if gt_label and gt_label != 'no definitive diagnosis':
        eval_results = evaluate_results(gt_label, derm_result, react_result, logger, model_name)

        # 평가 결과 저장
        eval_json = output_dir / f"evaluation_{model_name}_{timestamp}.json"
        with open(eval_json, 'w', encoding='utf-8') as f:
            json.dump(eval_results, f, indent=2, ensure_ascii=False, default=str)
        logger.info(f"\nEvaluation saved to: {eval_json}")
    else:
        logger.info(f"\nSkipping evaluation: GT is '{gt_label}'")

    # 요약
    logger.info("\n" + "="*80)
    logger.info("SUMMARY")
    logger.info("="*80)
    logger.info(f"Main Log: {main_log_file}")
    if derm_json:
        logger.info(f"DermatologyAgent JSON: {derm_json}")
    if react_json:
        logger.info(f"ReActAgent JSON: {react_json}")
    logger.info("="*80)
    logger.info("Comparison Complete!")

    # 터미널에 완료 메시지 출력
    print(f"[{timestamp}] Comparison complete!")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
