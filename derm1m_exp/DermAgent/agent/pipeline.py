"""
Derm1M Agent Pipeline Runner

ReAct 에이전트 + 계층적 평가 통합 파이프라인
"""

import os
import sys
import json
import csv
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

# 경로 설정 - eval 및 agent 폴더의 모듈을 import하기 위해
SCRIPT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(SCRIPT_DIR / "eval"))
sys.path.insert(0, str(SCRIPT_DIR / "agent"))

# 로컬 모듈
from ontology_utils import OntologyTree
from evaluation_metrics import HierarchicalEvaluator
from react_agent import ReActDermatologyAgent, DiagnosisResult


# ============ VLM 모델 래퍼 ============

class VLMFactory:
    """VLM 모델 팩토리"""
    
    @staticmethod
    def create(model_type: str, **kwargs):
        """모델 생성"""
        if model_type == "gpt":
            api_key = kwargs.get("api_key") or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("API key is required for GPT model")
            return GPT4oVLM(api_key=api_key)
        elif model_type == "qwen":
            model_path = kwargs.get("model_path")
            if not model_path:
                raise ValueError("model_path is required for Qwen model")
            return QwenVLM(model_path=model_path)
        elif model_type == "internvl":
            model_path = kwargs.get("model_path")
            if not model_path:
                raise ValueError("model_path is required for InternVL model")
            return InternVLM(model_path=model_path)
        else:
            raise ValueError(f"Unknown model type: {model_type}")


class GPT4oVLM:
    """GPT-4o 래퍼"""
    
    def __init__(self, api_key: str, model: str = "gpt-4o"):
        import openai
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
    
    def chat_img(self, prompt: str, image_paths: List[str], max_tokens: int = 1024) -> str:
        import base64
        
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        
        for path in image_paths:
            if os.path.exists(path):
                with open(path, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode()
                messages[0]["content"].append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
                })
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            return json.dumps({"error": str(e)})


class QwenVLM:
    """Qwen-VL 래퍼"""
    
    def __init__(self, model_path: str):
        import torch
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
        from qwen_vl_utils import process_vision_info
        
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.process_vision_info = process_vision_info
    
    def chat_img(self, prompt: str, image_paths: List[str], max_tokens: int = 1024) -> str:
        messages = [{"role": "user", "content": []}]
        
        for path in image_paths:
            if os.path.exists(path):
                messages[0]["content"].append({"type": "image", "image": f"file://{path}"})
        
        messages[0]["content"].append({"type": "text", "text": prompt})
        
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = self.process_vision_info(messages)
        
        inputs = self.processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt"
        ).to(self.model.device)
        
        outputs = self.model.generate(**inputs, max_new_tokens=max_tokens)
        generated = outputs[0][inputs.input_ids.shape[1]:]
        
        return self.processor.decode(generated, skip_special_tokens=True)


class InternVLM:
    """InternVL 래퍼"""
    
    def __init__(self, model_path: str):
        import torch
        from transformers import AutoTokenizer, AutoModel
        
        self.model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map='auto'
        ).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    def chat_img(self, prompt: str, image_paths: List[str], max_tokens: int = 1024) -> str:
        import torch
        from PIL import Image
        import torchvision.transforms as T
        from torchvision.transforms.functional import InterpolationMode
        
        MEAN = (0.485, 0.456, 0.406)
        STD = (0.229, 0.224, 0.225)
        
        transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB')),
            T.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD)
        ])
        
        images = []
        for path in image_paths:
            if os.path.exists(path):
                img = Image.open(path)
                images.append(transform(img))
        
        if not images:
            return json.dumps({"error": "No valid images"})
        
        pixel_values = torch.stack(images).to(dtype=torch.bfloat16, device=self.model.device)
        
        question = "<image>\n" * len(images) + prompt
        
        return self.model.chat(
            self.tokenizer,
            pixel_values=pixel_values,
            question=question,
            generation_config=dict(max_new_tokens=max_tokens, do_sample=False)
        )


# ============ 파이프라인 ============

@dataclass
class PipelineConfig:
    """파이프라인 설정"""
    ontology_path: Optional[str] = None  # None이면 자동 탐색
    model_type: str = "gpt"
    model_path: Optional[str] = None
    api_key: Optional[str] = None
    max_steps: int = 8
    max_depth: int = 4
    verbose: bool = True
    save_reasoning: bool = True


class DiagnosisPipeline:
    """진단 파이프라인"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        
        # VLM 초기화
        model_kwargs = {}
        if config.api_key:
            model_kwargs["api_key"] = config.api_key
        if config.model_path:
            model_kwargs["model_path"] = config.model_path
        
        self.vlm = VLMFactory.create(config.model_type, **model_kwargs)
        
        # 에이전트 초기화
        self.agent = ReActDermatologyAgent(
            ontology_path=config.ontology_path,
            vlm_model=self.vlm,
            max_steps=config.max_steps,
            verbose=config.verbose
        )
        
        # 평가기 초기화
        self.evaluator = HierarchicalEvaluator(config.ontology_path)
        
        print(f"Pipeline initialized with {config.model_type} model")
    
    def diagnose_single(self, image_path: str) -> Dict[str, Any]:
        """단일 이미지 진단"""
        result = self.agent.diagnose(image_path)
        
        output = {
            "image_path": image_path,
            "primary_diagnosis": result.primary_diagnosis,
            "differential_diagnoses": result.differential_diagnoses,
            "confidence": result.confidence,
            "ontology_path": result.ontology_path,
            "verification_passed": result.verification_passed,
            "warnings": result.warnings
        }
        
        if result.observations:
            output["observations"] = result.observations.to_dict()
        
        if self.config.save_reasoning:
            output["reasoning_chain"] = [s.to_dict() for s in result.reasoning_chain]
        
        return output
    
    def diagnose_batch(
        self,
        data: List[Dict[str, Any]],
        image_base_dir: str = ""
    ) -> List[Dict[str, Any]]:
        """배치 진단"""
        results = []
        
        try:
            from tqdm import tqdm
            iterator = tqdm(data, desc="Diagnosing")
        except ImportError:
            iterator = data
        
        for item in iterator:
            filename = item.get("filename", item.get("image_path", ""))
            image_path = os.path.join(image_base_dir, filename) if image_base_dir else filename
            
            try:
                result = self.diagnose_single(image_path)
                result["ground_truth"] = item.get("disease_label", item.get("label", ""))
            except Exception as e:
                result = {
                    "image_path": image_path,
                    "error": str(e),
                    "primary_diagnosis": "",
                    "ground_truth": item.get("disease_label", "")
                }
            
            results.append(result)
        
        return results
    
    def evaluate(
        self,
        results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """결과 평가"""
        ground_truths = []
        predictions = []
        
        for r in results:
            gt = r.get("ground_truth", "")
            if isinstance(gt, str):
                gt_labels = [l.strip() for l in gt.split(",") if l.strip()]
            else:
                gt_labels = gt if isinstance(gt, list) else []
            
            pred_labels = [r.get("primary_diagnosis", "")] if r.get("primary_diagnosis") else []
            pred_labels.extend(r.get("differential_diagnoses", [])[:2])
            
            ground_truths.append(gt_labels)
            predictions.append(pred_labels)
        
        eval_result = self.evaluator.evaluate_batch(ground_truths, predictions)
        
        return {
            "exact_match": eval_result.exact_match,
            "partial_match": eval_result.partial_match,
            "hierarchical_precision": eval_result.hierarchical_precision,
            "hierarchical_recall": eval_result.hierarchical_recall,
            "hierarchical_f1": eval_result.hierarchical_f1,
            "avg_hierarchical_distance": eval_result.avg_hierarchical_distance,
            "level_accuracy": eval_result.level_accuracy,
            "avg_partial_credit": eval_result.avg_partial_credit,
            "total_samples": eval_result.total_samples,
            "valid_samples": eval_result.valid_samples
        }
    
    def run(
        self,
        input_path: str,
        output_path: str,
        image_base_dir: str = "",
        limit: Optional[int] = None
    ) -> Dict[str, Any]:
        """전체 파이프라인 실행"""
        
        # 데이터 로드
        print(f"\nLoading data from {input_path}...")
        
        if input_path.endswith(".csv"):
            with open(input_path, 'r', encoding='utf-8-sig') as f:
                reader = csv.DictReader(f)
                data = list(reader)
        elif input_path.endswith(".json"):
            with open(input_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            raise ValueError(f"Unsupported file format: {input_path}")
        
        if limit:
            data = data[:limit]
        
        print(f"Loaded {len(data)} samples")
        
        # 진단 실행
        print("\nRunning diagnosis...")
        results = self.diagnose_batch(data, image_base_dir)
        
        # 평가
        print("\nEvaluating results...")
        evaluation = self.evaluate(results)
        
        # 결과 저장
        output = {
            "config": {
                "model_type": self.config.model_type,
                "max_steps": self.config.max_steps,
                "total_samples": len(results)
            },
            "evaluation": evaluation,
            "results": results
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        print(f"\nResults saved to {output_path}")
        
        # 평가 결과 출력
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        print(f"Total Samples: {evaluation['total_samples']}")
        print(f"Valid Samples: {evaluation['valid_samples']}")
        print(f"\nExact Match: {evaluation['exact_match']:.4f}")
        print(f"Partial Match: {evaluation['partial_match']:.4f}")
        print(f"Hierarchical F1: {evaluation['hierarchical_f1']:.4f}")
        print(f"Avg Distance: {evaluation['avg_hierarchical_distance']:.4f}")
        print(f"Partial Credit: {evaluation['avg_partial_credit']:.4f}")
        print("\nLevel Accuracy:")
        for level, acc in sorted(evaluation['level_accuracy'].items()):
            print(f"  Level {level}: {acc:.4f}")
        print("="*60)
        
        return output


# ============ CLI ============

def main():
    parser = argparse.ArgumentParser(
        description="Derm1M Diagnosis Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with GPT-4o
  python pipeline.py \\
      --input data.csv \\
      --output results.json \\
      --image_dir /path/to/images \\
      --model gpt \\
      --api_key YOUR_API_KEY
  
  # Run with Qwen-VL
  CUDA_VISIBLE_DEVICES=0,1 python pipeline.py \\
      --input data.csv \\
      --output results.json \\
      --model qwen \\
      --model_path Qwen/Qwen2-VL-7B-Instruct

  # Run with InternVL
  python pipeline.py \\
      --input data.csv \\
      --output results.json \\
      --model internvl \\
      --model_path OpenGVLab/InternVL2-8B
        """
    )
    
    parser.add_argument('--input', type=str, help='Input CSV or JSON file')
    parser.add_argument('--output', type=str, default='results.json', help='Output file')
    parser.add_argument('--image_dir', type=str, default='', help='Base image directory')
    parser.add_argument('--ontology', type=str, default=None, help='Ontology path (auto-detect if not specified)')
    parser.add_argument('--model', type=str, choices=['gpt', 'qwen', 'internvl'], default='gpt')
    parser.add_argument('--model_path', type=str, help='Model path (for local models)')
    parser.add_argument('--api_key', type=str, help='API key (for GPT)')
    parser.add_argument('--max_steps', type=int, default=8, help='Max reasoning steps')
    parser.add_argument('--limit', type=int, help='Limit samples')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--no_reasoning', action='store_true', help='Do not save reasoning chain')
    
    args = parser.parse_args()

    # 입력 파일 확인
    if not args.input:
        parser.print_help()
        print("\nError: --input is required")
        return
    
    # 설정
    config = PipelineConfig(
        ontology_path=args.ontology,
        model_type=args.model,
        model_path=args.model_path,
        api_key=args.api_key or os.getenv("OPENAI_API_KEY"),
        max_steps=args.max_steps,
        verbose=args.verbose,
        save_reasoning=not args.no_reasoning
    )
    
    # 파이프라인 실행
    pipeline = DiagnosisPipeline(config)
    pipeline.run(
        input_path=args.input,
        output_path=args.output,
        image_base_dir=args.image_dir,
        limit=args.limit
    )


if __name__ == "__main__":
    main()
