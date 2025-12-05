"""
Dermatology Agent Runner

에이전트를 실제 VLM 모델과 함께 실행하는 스크립트
"""

import os
import sys
import json
import csv
import argparse
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm

# 경로 설정 - agent 및 eval 폴더의 모듈을 import하기 위해
SCRIPT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(SCRIPT_DIR / "agent"))
sys.path.insert(0, str(SCRIPT_DIR / "eval"))

from dermatology_agent import DermatologyAgent, DiagnosisState
from ontology_utils import OntologyTree
from evaluation_metrics import HierarchicalEvaluator
from pipeline import GPT4oVLM, QwenVLM, InternVLM


def create_vlm(model_type: str, api_key: str = None, model_path: str = None):
    """VLM 인스턴스 생성"""
    if model_type == "gpt":
        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise ValueError("GPT 모델 사용 시 --api_key 또는 OPENAI_API_KEY 환경 변수가 필요합니다.")
        return GPT4oVLM(api_key=key)
    if model_type == "qwen":
        if not model_path:
            raise ValueError("Qwen 모델 사용 시 --model_path를 지정해야 합니다.")
        return QwenVLM(model_path=model_path)
    if model_type == "internvl":
        if not model_path:
            raise ValueError("InternVL 모델 사용 시 --model_path를 지정해야 합니다.")
        return InternVLM(model_path=model_path)
    raise ValueError(f"지원하지 않는 모델 타입입니다: {model_type}")


def run_agent_diagnosis(
    agent: DermatologyAgent,
    image_paths: List[str],
    output_path: str = None,
    max_depth: int = 4
) -> List[Dict]:
    """에이전트로 진단 실행"""
    
    results = []
    
    for image_path in tqdm(image_paths, desc="Diagnosing"):
        try:
            result = agent.diagnose(image_path, max_depth=max_depth)
            results.append(result)
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            results.append({
                "image_path": image_path,
                "error": str(e),
                "final_diagnosis": []
            })
    
    # 결과 저장
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {output_path}")
    
    return results


def evaluate_results(
    results: List[Dict],
    ground_truths: List[List[str]],
    ontology_path: str
) -> Dict:
    """결과 평가"""
    
    evaluator = HierarchicalEvaluator(ontology_path)
    
    predictions = [r.get("final_diagnosis", []) for r in results]
    
    eval_result = evaluator.evaluate_batch(ground_truths, predictions)
    evaluator.print_evaluation_report(eval_result)
    
    return eval_result


def load_csv_data(csv_path: str) -> tuple:
    """CSV에서 데이터 로드"""
    image_paths = []
    ground_truths = []
    
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = row.get('filename', '')
            disease_label = row.get('disease_label', '')
            
            image_paths.append(filename)
            
            # 쉼표로 구분된 라벨 파싱
            if disease_label:
                labels = [l.strip() for l in disease_label.split(',')]
            else:
                labels = []
            ground_truths.append(labels)
    
    return image_paths, ground_truths


def main():
    parser = argparse.ArgumentParser(description="Run Dermatology Diagnosis Agent")
    parser.add_argument('--ontology', type=str, default=None,
                        help='Path to ontology.json (auto-detect if not specified)')
    parser.add_argument('--input_csv', type=str, required=False,
                        help='Input CSV with image paths and ground truth')
    parser.add_argument('--image_dir', type=str, default='',
                        help='Base directory for images')
    parser.add_argument('--output', type=str, default='agent_results.json',
                        help='Output JSON file path')
    parser.add_argument('--model', type=str, choices=['gpt', 'qwen', 'internvl'],
                        default='gpt', help='VLM model to use')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to model (for qwen/internvl)')
    parser.add_argument('--api_key', type=str, default=None,
                        help='API key (for gpt)')
    parser.add_argument('--max_depth', type=int, default=4,
                        help='Maximum ontology traversal depth')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose output')
    
    args = parser.parse_args()
    
    # VLM 모델 초기화
    try:
        vlm = create_vlm(args.model, api_key=args.api_key, model_path=args.model_path)
    except ValueError as e:
        print(f"Error initializing VLM: {e}")
        return
    
    # 에이전트 생성
    try:
        agent = DermatologyAgent(
            ontology_path=args.ontology,
            vlm_model=vlm,
            verbose=args.verbose
        )
        if args.ontology is None:
            print(f"✓ Ontology auto-detected\n")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\n해결 방법:")
        print("  1. 프로젝트 루트에 ontology.json이 존재하는지 확인하세요.")
        print("  2. 수동 경로 지정: python run_agent.py --ontology /path/to/ontology.json ...")
        return
    
    # 데이터 로드
    if args.input_csv:
        image_paths, ground_truths = load_csv_data(args.input_csv)
        
        # 이미지 경로에 base directory 추가
        if args.image_dir:
            image_paths = [os.path.join(args.image_dir, p) for p in image_paths]
        
        print(f"Loaded {len(image_paths)} samples from {args.input_csv}")
        
        # 진단 실행
        results = run_agent_diagnosis(
            agent, 
            image_paths, 
            output_path=args.output,
            max_depth=args.max_depth
        )
        
        # 평가
        if ground_truths:
            print("\n=== Evaluation ===")
            evaluate_results(results, ground_truths, args.ontology)
    else:
        print("No input CSV provided. Specify --input_csv for batch processing.")


if __name__ == "__main__":
    main()


"""
Usage Examples:

# GPT-4o로 실행
python run_agent.py \
    --input_csv /path/to/sampled_data.csv \
    --image_dir /path/to/images \
    --output gpt_results.json \
    --model gpt \
    --api_key YOUR_API_KEY \
    --max_depth 4

# Qwen으로 실행
CUDA_VISIBLE_DEVICES=0,1 python run_agent.py \
    --input_csv /path/to/sampled_data.csv \
    --image_dir /path/to/images \
    --output qwen_results.json \
    --model qwen \
    --model_path Qwen/Qwen2-VL-7B-Instruct

# InternVL로 실행
python run_agent.py \
    --input_csv /path/to/sampled_data.csv \
    --image_dir /path/to/images \
    --output internvl_results.json \
    --model internvl \
    --model_path OpenGVLab/InternVL2-8B
"""
