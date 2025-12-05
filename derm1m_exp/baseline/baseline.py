import os
import sys
import argparse
import csv
import json
from pathlib import Path
from tqdm import tqdm
from typing import Optional

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from project_path import SAMPLED_DATA_CSV, OUTPUTS_ROOT, DERM1M_ROOT
from model import QwenVL, GPT4o, InternVL


def load_openai_key(arg_key: Optional[str] = None) -> Optional[str]:
    """Load OPENAI_API_KEY from arg, env, or project .env."""
    if arg_key:
        return arg_key
    env_key = os.environ.get("OPENAI_API_KEY")
    if env_key:
        return env_key
    env_path = Path(__file__).resolve().parents[2] / ".env"  # /home/work/wonjun/DermAgent/.env
    if env_path.exists():
        try:
            for line in env_path.read_text(encoding="utf-8").splitlines():
                if line.strip().startswith("OPENAI_API_KEY"):
                    _, val = line.split("=", 1)
                    return val.strip().strip('"').strip("'")
        except Exception:
            pass
    return None


class DermDiagnosisModel:
    """Wrapper class for dermatology diagnosis models"""

    def __init__(self, base_model):
        self.base_model = base_model
        # Use the base model's instruction (which includes disease_labels from model.py)
        # No need to override it here
        
    def analyze_image(self, image_path):
        """Analyze a single image and extract disease_label, body_location, and caption"""
        prompt = """Please analyze this dermatological image and provide the following information in JSON format:

{
    "disease_label": "The specific skin disease visible in the image (or 'no definitive diagnosis' if no clear lesion is visible)",
    "body_location": "The body part or location where the condition appears",
    "caption": "A detailed description of the skin condition visible in the image"
}

IMPORTANT: If you cannot identify any clear skin lesion or disease in the image, set disease_label to "no definitive diagnosis".

Provide ONLY the JSON output without any additional text."""

        try:
            # Check if image exists
            if not os.path.exists(image_path):
                return {
                    "disease_label": "Image not found",
                    "body_location": "N/A",
                    "caption": "Image file does not exist"
                }

            # Call the base model
            response = self.base_model.chat_img(prompt, [image_path], max_tokens=512)

            # Try to parse JSON response
            try:
                # Extract JSON from response (handle cases where model adds extra text)
                json_start = response.find('{')
                json_end = response.rfind('}') + 1
                if json_start != -1 and json_end > json_start:
                    json_str = response[json_start:json_end]
                    result = json.loads(json_str)

                    # Validate required fields
                    if all(key in result for key in ["disease_label", "body_location", "caption"]):
                        return result
                    else:
                        # Fill missing fields
                        return {
                            "disease_label": result.get("disease_label", "Unknown"),
                            "body_location": result.get("body_location", "Unknown"),
                            "caption": result.get("caption", response)
                        }
                else:
                    # No JSON found, use raw response
                    return {
                        "disease_label": "Parse error",
                        "body_location": "Parse error",
                        "caption": response
                    }
            except json.JSONDecodeError:
                # JSON parsing failed, return raw response in caption
                return {
                    "disease_label": "Parse error",
                    "body_location": "Parse error",
                    "caption": response
                }

        except Exception as e:
            return {
                "disease_label": "Error",
                "body_location": "Error",
                "caption": f"Error processing image: {str(e)}"
            }


def process_csv(input_csv, output_csv, image_base_folder, model):
    """Process all images in the CSV and generate predictions"""

    # Wrap the model
    derm_model = DermDiagnosisModel(model)

    # Read input CSV
    print(f"Reading CSV file: {input_csv}")
    with open(input_csv, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"Total images to process: {len(rows)}")

    # Process each image
    results = []
    for row in tqdm(rows, desc="Processing images"):
        filename = row.get('filename', '')

        # Construct full image path
        image_path = os.path.join(image_base_folder, filename)

        # Analyze image
        analysis = derm_model.analyze_image(image_path)

        # Store result
        result = {
            'filename': filename,
            'predicted_disease_label': analysis['disease_label'],
            'predicted_body_location': analysis['body_location'],
            'predicted_caption': analysis['caption'],
            'ground_truth_disease_label': row.get('disease_label', ''),
            'ground_truth_body_location': row.get('body_location', ''),
            'ground_truth_caption': row.get('caption', '')
        }
        results.append(result)

    # Write output CSV
    print(f"\nWriting results to: {output_csv}")
    fieldnames = [
        'filename',
        'predicted_disease_label',
        'predicted_body_location',
        'predicted_caption',
        'ground_truth_disease_label',
        'ground_truth_body_location',
        'ground_truth_caption'
    ]

    with open(output_csv, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"Processing complete! Results saved to {output_csv}")


def main():
    parser = argparse.ArgumentParser(description="Dermatology diagnosis using vision-language models")
    parser.add_argument('--model', choices=['qwen', 'gpt', 'internvl'], required=True,
                        help='The model to use')
    parser.add_argument('--input_csv', default=SAMPLED_DATA_CSV,
                        help=f'Input CSV file containing filenames (default: {SAMPLED_DATA_CSV})')
    parser.add_argument('--output_csv', default=None,
                        help='Output CSV file to store predictions (default: auto-generated in outputs/)')
    parser.add_argument('--image_base_folder', default=DERM1M_ROOT,
                        help=f'Base folder containing images (default: {DERM1M_ROOT})')
    parser.add_argument('--api_key', default=None,
                        help='API key for GPT model (if applicable)')
    parser.add_argument('--model_path', default=None,
                        help='Model path for Qwen/InternVL models')
    parser.add_argument('--no_labels_prompt', action='store_true',
                        help='Use label-agnostic prompt (Qwen/GPT)')

    args = parser.parse_args()
    
    # Auto-generate output_csv if not provided
    if args.output_csv is None:
        model_name = args.model
        args.output_csv = os.path.join(OUTPUTS_ROOT, f"{model_name}_predictions.csv")

    # Initialize model
    if args.model == 'qwen':
        if args.model_path is None:
            raise ValueError("Model path must be provided for Qwen model.")
        agent = QwenVL(model_path=args.model_path, use_labels_prompt=not args.no_labels_prompt)
    elif args.model == 'gpt':
        key = load_openai_key(args.api_key)
        if key is None:
            raise ValueError("API key must be provided for GPT model (set --api_key or OPENAI_API_KEY or .env).")
        agent = GPT4o(api_key=key, use_labels_prompt=not args.no_labels_prompt)
    elif args.model == 'internvl':
        if args.model_path is None:
            raise ValueError("Model path must be provided for InternVL model.")
        agent = InternVL(model_path=args.model_path)

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output_csv)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Process CSV
    process_csv(args.input_csv, args.output_csv, args.image_base_folder, agent)


if __name__ == "__main__":
    main()


"""
Usage examples:

# Using Qwen model
CUDA_VISIBLE_DEVICES=1,2,3 python /home/heodnjswns/burnskin/derm1m_exp/baseline/baseline.py --model qwen \
    --input_csv /home/heodnjswns/burnskin/dataset/Derm1M/random_samples_100/sampled_data.csv \
    --output_csv /home/heodnjswns/burnskin/derm1m_exp/baseline/outputs/qwen3vl32b_predictions.csv \
    --image_base_folder /home/heodnjswns/burnskin/dataset/Derm1M \
    --model_path Qwen/Qwen3-VL-32B-Instruct

# Using GPT-4o
python /home/heodnjswns/burnskin/derm1m_exp/baseline/baseline.py --model gpt \
    --input_csv /home/heodnjswns/burnskin/dataset/Derm1M/random_samples_100/sampled_data.csv \
    --output_csv /home/heodnjswns/burnskin/derm1m_exp/baseline/outputs/gpt4o_predictions.csv \
    --image_base_folder /home/heodnjswns/burnskin/dataset/Derm1M \
    --api_key YOUR_API_KEY

# Using InternVL model
CUDA_VISIBLE_DEVICES=4,5 python /home/heodnjswns/burnskin/derm1m_exp/baseline/baseline.py --model internvl \
    --input_csv /home/heodnjswns/burnskin/dataset/Derm1M/random_samples_100/sampled_data.csv \
    --output_csv /home/heodnjswns/burnskin/derm1m_exp/baseline/outputs/internvl3_14b_predictions.csv \
    --image_base_folder /home/heodnjswns/burnskin/dataset/Derm1M \
    --model_path OpenGVLab/InternVL3-14B
"""
