import os
import torch
from transformers import AutoProcessor, AutoModelForCausalLM, AutoTokenizer, AutoModel, AutoModelForVision2Seq
import json
import csv
from tqdm import tqdm
import pandas as pd
from qwen_vl_utils import process_vision_info
import openai
from openai import OpenAI
from utils import encode_image, encode_video, count_csv_rows, sort_files_by_number_in_name, compress_video, compress_image, videolist2imglist, sample_video_frames
import random
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

class BaselineModel:
    def __init__(self):
        pass

    def test_qa(self, json_file, csv_file):
        raise NotImplementedError


class QwenVL(BaselineModel):
    def __init__(self, model_path, use_labels_prompt: bool = True):
        # Load disease labels only when the label-guided prompt is desired
        if use_labels_prompt:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            disease_labels_path = os.path.join(script_dir, "extracted_node_names.txt")
            with open(disease_labels_path, "r") as f:
                disease_labels = [line.strip().split("→")[1] if "→" in line else line.strip() for line in f.readlines() if line.strip()]
            disease_labels_str = ", ".join(disease_labels)
            self.instruction = f"""
            You are a dermatology expert. You are provided with a skin image and a question about it.
            Please analyze the image carefully and provide a detailed diagnosis or answer based on your expertise.
            Focus on identifying skin conditions, lesions, or abnormalities visible in the image.

            When identifying skin conditions, the disease_label should be one of the following: {disease_labels_str}

            Provide a clear and professional response.
            """
        else:
            # Label-agnostic instruction
            self.instruction = """
            You are a dermatology expert. You are provided with a skin image and a question about it.
            Analyze the image carefully and provide a detailed, professional answer about visible skin findings.
            Do NOT assume or reference any predefined disease label list. If uncertain, state the likely differentials.
            """

        # Try to use flash attention if available, otherwise fall back to eager attention
        load_kwargs = dict(
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

        # First try with flash attention
        try:
            import flash_attn
            load_kwargs["attn_implementation"] = "flash_attention_2"
            print("Using Flash Attention 2")
        except ImportError:
            print("Flash Attention not available, using eager attention")

        try:
            # Qwen-VL models use AutoModelForVision2Seq for generation capability
            self.model = AutoModelForVision2Seq.from_pretrained(
                model_path,
                **load_kwargs,
            )
        except (TypeError, ValueError, ImportError) as exc:
            # Flash attention 관련 에러 시 제거 후 재시도
            if "flash" in str(exc).lower() or "attn_implementation" in str(exc):
                print(f"Flash Attention error, retrying without it: {exc}")
                load_kwargs.pop("attn_implementation", None)
                self.model = AutoModelForVision2Seq.from_pretrained(
                    model_path,
                    **load_kwargs,
                )
            else:
                raise
        # Use fast processor explicitly to avoid warning
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, use_fast=True)

    def chat_video(self, input_text, video_path, max_tokens=512):
        messages = [
            {"role": "system", "content": self.instruction},
            {
                "role": "user",
                "content": []
            },
        ]
        for i in range(len(video_path)):
            messages[1]["content"].append(
                {
                    "type": "video",
                    "video": "file://" + video_path[i],
                    "max_pixels": 360 * 640,
                    "fps": 1.0,
                }
            )
        messages[1]["content"].append(
            {"type": "text", "text": input_text}
        )

        # Use chat API instead of generate
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)

        # Generate output using model's generate method from the base model
        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        return output_text

    def chat_img(self, input_text, image_path, max_tokens=512):
        messages = [
            {"role": "system", "content": self.instruction},
            {
                "role": "user",
                "content": []
            },
        ]
        for i in range(len(image_path)):
            messages[1]["content"].append(
                {"type": "image", "image": "file://" + image_path[i]}
            )
        messages[1]["content"].append(
            {"type": "text", "text": input_text}
        )

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)

        generated_ids = self.model.generate(**inputs, max_new_tokens=max_tokens)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        return output_text

    def test_qa(self, input_file, output_file, materials_folder):
        with open(input_file, "r") as f:
            data = json.load(f)
        answers = []
        for item in tqdm(data):
            id_num = item["id"]
            question = item["Q"]
            prompt = f"Q: {question}"

            old_material_path = item["materials"]
            material_path = []
            try:
                if old_material_path:
                    # 모든 materials 항목을 순회하며 파일/디렉토리 처리
                    for path in old_material_path:
                        full_path = os.path.join(materials_folder, path)
                        if os.path.isfile(full_path):
                            # 파일이면 그대로 추가
                            material_path.append(full_path)
                        elif os.path.isdir(full_path):
                            # 디렉토리면 내부 파일들을 정렬해서 추가
                            dir_files = sort_files_by_number_in_name(full_path)
                            material_path.extend(dir_files)
                    if material_path:
                        image_exts = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff', '.tif', '.heic'}
                        video_exts = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm', '.mpeg', '.mpg'}
                        if os.path.splitext(material_path[0])[1].lower() in image_exts:
                            reply = self.chat_img(prompt, material_path)
                        elif os.path.splitext(material_path[0])[1].lower() in video_exts:
                            reply = self.chat_video(prompt, material_path)
                        else:
                            reply = self.chat_img(prompt, [])
                    else:
                        reply = self.chat_img(prompt, [])
                else:
                    reply = self.chat_img(prompt, [])
                answer = reply
            except Exception as e:
                print(f"Error processing item id {id_num}: {e}")
                answer = "Error: Unable to process this image."

            answers.append({"id": id_num, "Answer": answer})
        with open(output_file, "w", encoding='utf-8') as f:
            json.dump(answers, f, ensure_ascii=False, indent=4)
        return


class InternVL(BaselineModel):
    def __init__(self, model_path):
        # Load disease labels - use relative path from script location
        script_dir = os.path.dirname(os.path.abspath(__file__))
        disease_labels_path = os.path.join(script_dir, "extracted_node_names.txt")
        with open(disease_labels_path, "r") as f:
            disease_labels = [line.strip().split("→")[1] if "→" in line else line.strip() for line in f.readlines() if line.strip()]
        disease_labels_str = ", ".join(disease_labels)

        self.instruction = f"""
        You are a dermatology expert. You are provided with a skin image and a question about it.
        Please analyze the image carefully and provide a detailed diagnosis or answer based on your expertise.
        Focus on identifying skin conditions, lesions, or abnormalities visible in the image.

        When identifying skin conditions, the disease_label should be one of the following: {disease_labels_str}

        Provide a clear and professional response.
        """

        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load model and tokenizer
        # Use trust_remote_code to ensure the model loads with the correct architecture
        self.model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map='auto'  # Let transformers handle device placement
        ).eval()

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )

    def load_image(self, image_path, max_num=12):
        """Load and preprocess image"""
        image = Image.open(image_path).convert('RGB')
        transform = self._build_transform(input_size=448)
        images = self._dynamic_preprocess(image, image_size=448, max_num=max_num)
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values).to(dtype=torch.bfloat16, device=self.device)
        return pixel_values

    def _build_transform(self, input_size):
        """Build image transform"""
        IMAGENET_MEAN = (0.485, 0.456, 0.406)
        IMAGENET_STD = (0.229, 0.224, 0.225)

        transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
        return transform

    def _find_closest_aspect_ratio(self, aspect_ratio, target_ratios, width, height, image_size):
        """Find the closest aspect ratio"""
        best_ratio_diff = float('inf')
        best_ratio = (1, 1)
        area = width * height
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
        return best_ratio

    def _dynamic_preprocess(self, image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
        """Dynamically preprocess image"""
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height

        # Calculate target ratios
        target_ratios = set(
            (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
            i * j <= max_num and i * j >= min_num)
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

        # Find the closest aspect ratio
        target_aspect_ratio = self._find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, image_size)

        # Calculate target width and height
        target_width = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

        # Resize the image
        resized_img = image.resize((target_width, target_height))
        processed_images = []
        for i in range(blocks):
            box = (
                (i % (target_width // image_size)) * image_size,
                (i // (target_width // image_size)) * image_size,
                ((i % (target_width // image_size)) + 1) * image_size,
                ((i // (target_width // image_size)) + 1) * image_size
            )
            split_img = resized_img.crop(box)
            processed_images.append(split_img)

        if use_thumbnail and len(processed_images) != 1:
            thumbnail_img = image.resize((image_size, image_size))
            processed_images.append(thumbnail_img)

        return processed_images

    def chat_img(self, input_text, image_path, max_tokens=512):
        """Chat with images"""
        if not image_path:
            # No image case
            question = f"{self.instruction}\n\n{input_text}"
            try:
                response = self.model.chat(
                    self.tokenizer,
                    pixel_values=None,
                    question=question,
                    generation_config=dict(max_new_tokens=max_tokens, do_sample=False)
                )
            except AttributeError:
                # If chat method doesn't exist, try alternative approach
                response = "Error: Model does not support chat without images"
            return response

        # Load images
        pixel_values_list = []
        num_patches_list = []

        for img_path in image_path:
            pixel_values = self.load_image(img_path, max_num=12)
            pixel_values_list.append(pixel_values)
            num_patches_list.append(pixel_values.size(0))

        # Concatenate all images
        pixel_values = torch.cat(pixel_values_list, dim=0)

        # Build question with image placeholders
        question = self.instruction + "\n\n"
        for i in range(len(image_path)):
            question += f"<image>\n"
        question += input_text

        # Generate response - try with history parameter
        try:
            response = self.model.chat(
                self.tokenizer,
                pixel_values=pixel_values,
                question=question,
                generation_config=dict(max_new_tokens=max_tokens, do_sample=False),
                num_patches_list=num_patches_list,
                history=None,
                return_history=False
            )
        except TypeError:
            # If history parameter is not supported, try without it
            response = self.model.chat(
                self.tokenizer,
                pixel_values=pixel_values,
                question=question,
                generation_config=dict(max_new_tokens=max_tokens, do_sample=False),
                num_patches_list=num_patches_list
            )

        return response

    def chat_video(self, input_text, video_path, max_tokens=512, max_frames=24):
        """Chat with videos by sampling frames"""
        # Sample frames from video - returns list of image file paths
        frame_paths = []

        for vid_path in video_path:
            # Sample frames uniformly from the video
            frames = sample_video_frames(vid_path, max_frames=max_frames)
            frame_paths.extend(frames)

        try:
            # Use chat_img to process video frames
            response = self.chat_img(input_text, frame_paths, max_tokens=max_tokens)
        finally:
            # Clean up temporary files created by sample_video_frames
            for frame_path in frame_paths:
                try:
                    os.remove(frame_path)
                except:
                    pass

        return response

    def test_qa(self, input_file, output_file, materials_folder):
        """Test QA on the dataset"""
        with open(input_file, "r") as f:
            data = json.load(f)

        answers = []
        for item in tqdm(data):
            id_num = item["id"]
            question = item["Q"]
            prompt = f"Q: {question}"

            old_material_path = item["materials"]
            material_path = []

            try:
                if old_material_path:
                    # Process all materials (files/directories)
                    for path in old_material_path:
                        full_path = os.path.join(materials_folder, path)
                        if os.path.isfile(full_path):
                            # Add file directly
                            material_path.append(full_path)
                        elif os.path.isdir(full_path):
                            # Add sorted files from directory
                            dir_files = sort_files_by_number_in_name(full_path)
                            material_path.extend(dir_files)

                    if material_path:
                        image_exts = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff', '.tif', '.heic'}
                        video_exts = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm', '.mpeg', '.mpg'}

                        if os.path.splitext(material_path[0])[1].lower() in image_exts:
                            reply = self.chat_img(prompt, material_path)
                        elif os.path.splitext(material_path[0])[1].lower() in video_exts:
                            reply = self.chat_video(prompt, material_path)
                        else:
                            reply = self.chat_img(prompt, [])
                    else:
                        reply = self.chat_img(prompt, [])
                else:
                    reply = self.chat_img(prompt, [])

                answer = reply
            except Exception as e:
                print(f"Error processing item id {id_num}: {e}")
                answer = "Error: Unable to process this image."

            answers.append({"id": id_num, "Answer": answer})

        with open(output_file, "w", encoding='utf-8') as f:
            json.dump(answers, f, ensure_ascii=False, indent=4)

        return


class GPT4o(BaselineModel):
    def __init__(self, api_key, use_labels_prompt: bool = True):
        self.client = OpenAI(api_key=api_key)

        if use_labels_prompt:
            # Load disease labels - use relative path from script location
            script_dir = os.path.dirname(os.path.abspath(__file__))
            disease_labels_path = os.path.join(script_dir, "extracted_node_names.txt")
            with open(disease_labels_path, "r") as f:
                disease_labels = [line.strip().split("→")[1] if "→" in line else line.strip() for line in f.readlines() if line.strip()]
            disease_labels_str = ", ".join(disease_labels)

            self.instruction = f"""
            You are a dermatology expert. You are provided with a skin image and a question about it.
            Please analyze the image carefully and provide a detailed diagnosis or answer based on your expertise.
            Focus on identifying skin conditions, lesions, or abnormalities visible in the image.

            When identifying skin conditions, the disease_label should be one of the following: {disease_labels_str}

            Provide a clear and professional response.
            """
        else:
            # Label-agnostic instruction
            self.instruction = """
            You are a dermatology expert. You are provided with a skin image and a question about it.
            Analyze the image carefully and provide a detailed, professional answer about visible skin findings.
            Do NOT assume or reference any predefined disease label list. If uncertain, state the likely differentials.
            """
        self.model = "gpt-4o"

    def chat_img(self, input_text, image_path, max_tokens=512):
        try:
            base64_images = []
            for image in image_path:
                base64_image = encode_image(image)
                base64_images.append(base64_image)

            messages = [
                {"role": "system", "content": self.instruction},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": input_text},
                    ]
                }
            ]
            for image in base64_images:
                messages[1]["content"].append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{image}"
                    }
                })
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens
            )
            return completion.choices[0].message.content
        except openai.APIStatusError as e:
            if "413" in str(e):
                print("Image too large, compressing and retrying...")
                compressed_images = []
                i = 0
                for image in image_path:
                    i += 1
                    compressed_image_path = compress_image(image, f"tmp_file/compressed_image{i}.jpg", quality=50)
                    compressed_images.append(compressed_image_path)
                return self.chat_img(input_text, compressed_images, max_tokens)
            else:
                raise e

    def chat_video(self, input_text, video_path, max_tokens=512):
        try:
            base64_videos = []
            base64_videos = videolist2imglist(video_path, 25)

            messages = [
                {"role": "system", "content": self.instruction},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": input_text},
                    ]
                }
            ]
            i = 0
            for video in base64_videos:
                intro = f"These images are uniformly captured from the {i}th video in chronological order. There are a total of {len(video)} pictures."
                messages[1]["content"].append({
                    "type": "text",
                    "text": intro
                })
                for image in video:
                    messages[1]["content"].append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image}"
                        }
                    })
                i += 1
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens
            )
            return completion.choices[0].message.content
        except openai.APIStatusError as e:
            if "413" in str(e):
                print("video too large, compressing and retrying...")
                compressed_videos = []
                i = 0
                for video in video_path:
                    i += 1
                    compressed_video_path = compress_video(video, f"tmp_file/compressed_video{i}.mp4")
                    compressed_videos.append(compressed_video_path)
                return self.chat_video(input_text, compressed_videos, max_tokens)
            else:
                raise e

    def test_qa(self, input_file, output_file, materials_folder):
        with open(input_file, "r") as f:
            data = json.load(f)
        answers = []
        for item in tqdm(data):
            id_num = item["id"]
            question = item["Q"]
            prompt = f"Q: {question}"

            old_material_path = item["materials"]
            material_path = []
            try:
                if old_material_path:
                    # 모든 materials 항목을 순회하며 파일/디렉토리 처리
                    for path in old_material_path:
                        full_path = os.path.join(materials_folder, path)
                        if os.path.isfile(full_path):
                            # 파일이면 그대로 추가
                            material_path.append(full_path)
                        elif os.path.isdir(full_path):
                            # 디렉토리면 내부 파일들을 정렬해서 추가
                            dir_files = sort_files_by_number_in_name(full_path)
                            material_path.extend(dir_files)
                    if material_path:
                        image_exts = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff', '.tif', '.heic'}
                        video_exts = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm', '.mpeg', '.mpg'}
                        if os.path.splitext(material_path[0])[1].lower() in image_exts:
                            reply = self.chat_img(prompt, material_path)
                        elif os.path.splitext(material_path[0])[1].lower() in video_exts:
                            reply = self.chat_video(prompt, material_path)
                        else:
                            reply = self.chat_img(prompt, [])
                    else:
                        reply = self.chat_img(prompt, [])
                else:
                    reply = self.chat_img(prompt, [])
                answer = reply
            except Exception as e:
                print(f"Error processing item id {id_num}: {e}")
                answer = "Error: Unable to process this image."

            answers.append({"id": id_num, "Answer": answer})
        with open(output_file, "w", encoding='utf-8') as f:
            json.dump(answers, f, ensure_ascii=False, indent=4)
        return
