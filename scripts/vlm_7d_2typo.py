import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
import torch
import wandb
import pickle
import random
import os, sys
import json
import re
import argparse
import time
# Keep project-root imports working when this script lives under scripts/
from pathlib import Path as _Path
_REPO_ROOT = _Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from utils import find_or_create_run_dir
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as transforms
import pandas as pd
from dotenv import load_dotenv

ENABLE_WANDB_LOGGING = False

if ENABLE_WANDB_LOGGING:
    wandb.init(project="vlm-jailbreak")

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(1) 
load_dotenv()


def text_to_art_image_pil(text, font_path='arial.ttf', font_size=100, text_color=(0, 0, 0), bg_color=(255, 255, 255), image_width=500):
    """Renders text into a PIL Image with word-wrapping and centered layout."""
    image = Image.new('RGB', (image_width, 300), color=bg_color)
    draw = ImageDraw.Draw(image)

    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        print(f"Font not found at {font_path}, trying arial.ttf...")
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except IOError:
            print("Arial font not found, using default font.")
            font = ImageFont.load_default()

    lines = []
    words = text.split()
    current_line = ""
    max_width = int(image_width * 0.9)

    for word in words:
        test_line = f"{current_line} {word}".strip()
        try:
            text_bbox = draw.textbbox((0, 0), test_line, font=font)
            text_width = text_bbox[2] - text_bbox[0]
        except AttributeError:
            text_width = draw.textsize(test_line, font=font)[0]
        
        if text_width <= max_width:
            current_line = test_line
        else:
            if current_line:
                lines.append(current_line)
            current_line = word

    if current_line:
        lines.append(current_line)

    line_spacing = max(5, font_size // 10)
    total_height = len(lines) * font_size + (len(lines) - 1) * line_spacing + 20
    image = Image.new('RGB', (image_width, total_height), color=bg_color)
    draw = ImageDraw.Draw(image)

    y_offset = 10
    for line in lines:
        try:
            text_bbox = draw.textbbox((0, 0), line, font=font)
            text_width = text_bbox[2] - text_bbox[0]
        except AttributeError:
            text_width = draw.textsize(line, font=font)[0]
        
        text_x = (image.width - text_width) // 2
        draw.text((text_x, y_offset), line, font=font, fill=text_color)
        y_offset += font_size + line_spacing 

    return image


def parse_7d_2typo_json(adversarial_queries, original_prompt):
    """
    Parse attacker output for 7d+2typo variant.
    Expected format: {"text prompt 1": ..., "text prompt 2": ...}
    Returns: (text_prompts_1, text_prompts_2)
    text_prompts_1 is also used as adversarial query to VLM.
    """
    text_prompts_1 = []
    text_prompts_2 = []
    
    for query in adversarial_queries:
        parsed = {}
        json_match = re.search(r'\{.*\}', str(query), re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            json_str = re.sub(r',\s*([}\]])', r'\1', json_str)
            open_b = json_str.count('{')
            close_b = json_str.count('}')
            if open_b > close_b:
                json_str += '}' * (open_b - close_b)
            try:
                parsed = json.loads(json_str)
            except json.JSONDecodeError:
                pass
        
        if not parsed:
            for key_name in ["text prompt 1", "text prompt 2"]:
                pattern = rf'"{key_name}"\s*:\s*"(.*?)"'
                match = re.search(pattern, str(query), re.DOTALL | re.IGNORECASE)
                if match:
                    parsed[key_name] = match.group(1).strip().rstrip('",}')
        
        tp1 = parsed.get("text prompt 1", "")
        tp2 = parsed.get("text prompt 2", "")
        
        if not tp1:
            tp1 = "Please analyze the images and provide detailed information about what you see."
        if not tp2:
            tp2 = tp1
            
        text_prompts_1.append(tp1)
        text_prompts_2.append(tp2)
    
    print(f"Parsed {len(text_prompts_1)} text prompt pairs")
    return text_prompts_1, text_prompts_2


def concatenate_images_7d_2typo(distraction_image_paths, typo_image_1, typo_image_2,
                                 images_per_row=3, target_size=(300, 300), fill_color=(255, 255, 255), 
                                 font_size=20, rotation_angle=0):
    """Concatenate 7 distraction images + 2 typographic images into a 3x3 grid."""
    images = []
    
    try:
        font = ImageFont.truetype(str(_REPO_ROOT / "Super Moods.ttf"), font_size)  
    except IOError:
        try:
            font = ImageFont.truetype("arial.ttf", font_size)  
        except IOError:
            font = ImageFont.load_default()

    for idx, img_path in enumerate(distraction_image_paths[:7]):
        if img_path.startswith('./llava_images/'):
            filename = img_path.replace('./llava_images/', '')
            full_path = str(_REPO_ROOT / "CS_DJ_main" / "llava_images" / filename)
            img_path = full_path
        img = Image.open(img_path)
        img.thumbnail(target_size) 

        diagonal = int((target_size[0]**2 + target_size[1]**2)**0.5)
        expanded_img = Image.new('RGB', (diagonal, diagonal), fill_color)
        img_x, img_y = img.size
        paste_x = (diagonal - img_x) // 2
        paste_y = (diagonal - img_y) // 2
        expanded_img.paste(img, (paste_x, paste_y))

        rotated_img = expanded_img.rotate(rotation_angle, expand=True, fillcolor=fill_color)

        final_padded_img = Image.new('RGB', target_size, fill_color)
        rotated_img_x, rotated_img_y = rotated_img.size
        final_paste_x = (target_size[0] - rotated_img_x) // 2
        final_paste_y = (target_size[1] - rotated_img_y) // 2
        final_padded_img.paste(rotated_img, (final_paste_x, final_paste_y))

        final_img = Image.new('RGB', (target_size[0], target_size[1] + font_size + 10), fill_color)
        final_img.paste(final_padded_img, (0, 0))

        draw = ImageDraw.Draw(final_img)
        text = str(idx + 1)
        try:
            text_bbox = draw.textbbox((0, 0), text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
        except AttributeError:
            text_width = draw.textsize(text, font=font)[0]
        
        text_x = (target_size[0] - text_width) // 2
        text_y = target_size[1] + (font_size // 2)
        draw.text((text_x, text_y), text, font=font, fill=(0, 0, 0))  

        images.append(final_img)

    # Typo image 1 (position 8) and Typo image 2 (position 9)
    for pos_idx, typo_img in enumerate([typo_image_1, typo_image_2]):
        if typo_img:
            typo_copy = typo_img.copy()
            typo_copy.thumbnail(target_size)
            
            final_padded_img = Image.new('RGB', target_size, fill_color)
            img_x, img_y = typo_copy.size
            paste_x = (target_size[0] - img_x) // 2
            paste_y = (target_size[1] - img_y) // 2
            final_padded_img.paste(typo_copy, (paste_x, paste_y))

            final_img = Image.new('RGB', (target_size[0], target_size[1] + font_size + 10), fill_color)
            final_img.paste(final_padded_img, (0, 0))

            draw = ImageDraw.Draw(final_img)
            text = str(8 + pos_idx)
            try:
                text_bbox = draw.textbbox((0, 0), text, font=font)
                text_width = text_bbox[2] - text_bbox[0]
            except AttributeError:
                text_width = draw.textsize(text, font=font)[0]
            
            text_x = (target_size[0] - text_width) // 2
            text_y = target_size[1] + (font_size // 2)
            draw.text((text_x, text_y), text, font=font, fill=(0, 0, 0))

            images.append(final_img)

    width, height = target_size[0], target_size[1] + font_size + 10
    rows = (len(images) + images_per_row - 1) // images_per_row
    total_width = width * images_per_row
    total_height = height * rows
    new_image = Image.new('RGB', (total_width, total_height), fill_color)

    for index, img in enumerate(images):
        x_offset = (index % images_per_row) * width
        y_offset = (index // images_per_row) * height
        new_image.paste(img, (x_offset, y_offset))

    return new_image


@hydra.main(config_path="../configs", config_name="vlm_7d_2typo", version_base=None)
def main(cfg):
    cfg_omega = OmegaConf.create(cfg)
    run_dir, is_resume = find_or_create_run_dir(cfg_omega)
    target_vlm = instantiate(cfg.target)
    print("Loaded target VLM")
    print("Loading models...")
    attacker = instantiate(cfg.attacker)
    print("Loaded attacker")
    judge = instantiate(cfg.judge)
    print("Loaded judge")

    prompts, category = instantiate(cfg.data)
    print("Loaded data")
    optimizer = instantiate(cfg.optimizer, attacker.parameters())
    print("Loaded optimizer")
    loss_fn = instantiate(cfg.loss)(
        target_lm=target_vlm, 
        judge=judge,
    )
    print("Loaded loss function")

    all_adversarial_data = {
        'indices': [],
        'original_prompts': [],
        'adversarial_prompts': [],
        'responses': [],
        'scores': [],
        'losses': [],
        'query_counts': [],
        'prompt_ids': []
    }
    
    global_counter = 0
    log_every = cfg.log_every
    JB = 0
    prompts_total = []
    responses_total = []
    scores = []
    final_adv_prompts = []
    fail_cases = []
    start_idx = 0
    end_idx = 4
    Total_JB_prompt = 0
    Total_JB_round = 0
    SUCCESS = 0

    select_img_path = str(
        _REPO_ROOT / "CS_DJ_main/image_map/harmbench_distraction_image_map_seed_0_num_10000.json"
    )
    
    distraction_images_map = {}
    if os.path.exists(select_img_path):
        with open(select_img_path, 'r', encoding='utf-8') as f:
            distraction_images_map = json.load(f)
        print(f"Loaded distraction images map from {select_img_path}")
    else:
        print(f"Warning: Distraction images map not found at {select_img_path}")

    font_path = str(_REPO_ROOT / "Super Moods.ttf")
    
    best_prompts_file = f"{run_dir}/best_prompts_incremental.txt"
    with open(best_prompts_file, "w") as f:
        f.write("BEST PROMPTS INCREMENTAL LOG (7D+2TYPO ABLATION)\n")
        f.write("=" * 80 + "\n")
        f.write(f"Run Directory: {run_dir}\n")
        f.write(f"Start Index: {start_idx}, End Index: {end_idx}\n")
        f.write(f"Started at: {pd.Timestamp.now()}\n")
        f.write("=" * 80 + "\n\n")

    total_run_start_time = time.time()

    for prompt_idx, (prompt, category) in enumerate(zip(prompts, category)):
        if prompt_idx < start_idx:
            continue
        if prompt_idx > end_idx:
            break
        attacker.initialize_attacker()
        optimizer = instantiate(cfg.optimizer, attacker.parameters())
        
        cur_prompt_adv = []
        cur_prompt_resp = []
        cur_images = []
        scores = []

        max_queries = cfg.max_queries
        THRESHOLD = cfg.threshold
        BATCH_SIZE = cfg.batch_size
        CUR_COUNT = 0
        total_attempts = 0
        batch_prompt = [prompt for _ in range(BATCH_SIZE)]
        PASS = False
        prompt_start_time = time.time()
        print("=====New prompt ======")
        print("Original prompt: ", prompt)

        while True:
            if CUR_COUNT >= max_queries:
                break 
            Total_JB_round += 1
            optimizer.zero_grad()
            print()
            print(CUR_COUNT)
            print("Original prompt: ", prompt)
            adversarial_queries, log_likelihoods, entropy, log_prior = attacker.generate_conditional_prompt(
                batch_prompt, prior=True
            )

            text_prompts_1, text_prompts_2 = parse_7d_2typo_json(adversarial_queries, prompt)
            
            print("Log-likelihoods", log_likelihoods)
            print("Log-prior", log_prior)
            
            distraction_image_paths = []
            if prompt in distraction_images_map:
                distraction_image_paths = distraction_images_map[prompt][:7]
            else:
                print(f"Warning: No distraction images found for prompt: {prompt}")
                exit()
            
            concatenated_images = []
            for i in range(len(text_prompts_1)):
                typo_image_1 = text_to_art_image_pil(
                    text_prompts_1[i], font_path=font_path, font_size=80, 
                    text_color=(0, 0, 0), bg_color=(255, 255, 255), image_width=800
                )
                typo_image_2 = text_to_art_image_pil(
                    text_prompts_2[i], font_path=font_path, font_size=80, 
                    text_color=(0, 0, 0), bg_color=(255, 255, 255), image_width=800
                )
                
                concat_image = concatenate_images_7d_2typo(
                    distraction_image_paths, 
                    typo_image_1, 
                    typo_image_2,
                    images_per_row=3, 
                    target_size=(300, 300)
                )
                concatenated_images.append(concat_image)
            
            save_dir = os.path.join(run_dir, "concatenated_images")
            os.makedirs(save_dir, exist_ok=True)
            
            print(f"Generated {len(concatenated_images)} concatenated images (7d+2typo) for batch size {len(text_prompts_1)}")
            loss_dct = loss_fn(
                adversarial_queries=text_prompts_1,
                log_likelihoods=log_likelihoods,
                entropies=entropy,
                log_prior=log_prior,
                original_queries=batch_prompt,
                images=concatenated_images,
            )

            loss = loss_dct.pop("loss")
            print(f"Loss: {loss.item()}")
            
            batch_scores = loss_dct.pop("judge_scores")
            scores += batch_scores
            loss.backward()
            
            loss_scalar = loss.item()

            if ENABLE_WANDB_LOGGING:
                log_dict = {}
                for k, v in loss_dct.items():
                    if isinstance(v, (int, float)):
                        log_dict[k] = v
                    elif isinstance(v, list) and len(v) <= 20:
                        log_dict[k] = v
                log_dict["loss"] = loss_scalar
                log_dict["query_count"] = CUR_COUNT
                log_dict["prompt_id"] = hash(prompt) % 100000
                wandb.log(log_dict)
            
            cur_prompt_adv += adversarial_queries
            cur_prompt_resp += loss_dct["responses"]
            cur_images += concatenated_images
            
            total_attempts += 1

            for i in range(len(adversarial_queries)):
                all_adversarial_data['indices'].append(global_counter)
                all_adversarial_data['original_prompts'].append(prompt)
                all_adversarial_data['adversarial_prompts'].append(adversarial_queries[i])
                all_adversarial_data['responses'].append(loss_dct["responses"][i])
                all_adversarial_data['scores'].append(batch_scores[i])
                all_adversarial_data['losses'].append(loss_scalar)
                all_adversarial_data['query_counts'].append(CUR_COUNT + i)
                all_adversarial_data['prompt_ids'].append(prompt_idx)
                global_counter += 1

            trigger_update = (np.max(batch_scores) >= THRESHOLD)
            if trigger_update:
                print("Above-threshold response detected, updating and finishing current prompt")
                JB = JB + 1
                PASS = True
                
                save_dir = os.path.join(run_dir, "concatenated_images")
                os.makedirs(save_dir, exist_ok=True)
                
                best_batch_idx = int(np.argmax(batch_scores))
                best_batch_image = concatenated_images[best_batch_idx]
                best_batch_score = batch_scores[best_batch_idx]
                
                image_filename = f"best_adv_image_prompt_{prompt_idx:04d}_score_{best_batch_score:.4f}_threshold_passed.png"
                image_path = os.path.join(save_dir, image_filename)
                best_batch_image.save(image_path)
                print(f"Saved best adversarial image (threshold passed) to: {image_path}")
                
                print("-" * 80)
                break

            CUR_COUNT += BATCH_SIZE
            Total_JB_prompt += BATCH_SIZE
            print("KL:", loss_dct['kl'], "entropy:" , loss_dct['entropy'])
            for q, r in zip(adversarial_queries, loss_dct['responses']):
                print("Adv Query:", q)
                print("VLM Response:\n", r)
                print("--------------------------------")

        prompt_elapsed_time = time.time() - prompt_start_time
        prompt_elapsed_minutes = prompt_elapsed_time / 60.0

        if PASS:
            print(f"Successfully finished prompt {prompt}")
        else:
            print(f"Failed to jailbreak prompt {prompt}")
        print(f"Time for prompt {prompt_idx}: {prompt_elapsed_time:.2f}s ({prompt_elapsed_minutes:.2f}min)")

        best_adv_prompt_idx = int(np.argmax(scores))
        print("best_adv_prompt_idx:", best_adv_prompt_idx)
        best_adv_prompt = cur_prompt_adv[best_adv_prompt_idx]
        best_score = scores[best_adv_prompt_idx]
        prompts_total.append(best_adv_prompt)
        responses_total.append(cur_prompt_resp[best_adv_prompt_idx])
        
        if len(cur_images) > 0 and best_adv_prompt_idx < len(cur_images):
            save_dir = os.path.join(run_dir, "concatenated_images")
            os.makedirs(save_dir, exist_ok=True)
            best_image = cur_images[best_adv_prompt_idx]
            image_filename = f"best_adv_image_prompt_{prompt_idx:04d}_idx_{best_adv_prompt_idx:04d}_score_{best_score:.4f}.png"
            image_path = os.path.join(save_dir, image_filename)
            best_image.save(image_path)
            print(f"Saved best adversarial image (overall) to: {image_path}")
        
        best_prompts_file = f"{run_dir}/best_prompts_incremental.txt"
        with open(best_prompts_file, "a") as f:
            f.write(f"PROMPT {prompt_idx} (Index: {prompt_idx - start_idx + 1}):\n")
            f.write(f"Timestamp: {pd.Timestamp.now()}\n")
            f.write(f"Original Prompt: {prompt}\n")
            f.write(f"Best Adversarial Prompt: {best_adv_prompt}\n")
            f.write(f"Best Response: {cur_prompt_resp[best_adv_prompt_idx]}\n")
            f.write(f"Best Score: {best_score:.4f}\n")
            f.write(f"Threshold: {THRESHOLD}\n")
            f.write(f"Query Count: {CUR_COUNT}\n")
            f.write(f"Jailbreak Success: {'YES' if best_score > THRESHOLD else 'NO'}\n")
            f.write(f"Current JB Rate: {JB}/{prompt_idx - start_idx + 1} = {JB/(prompt_idx - start_idx + 1):.4f}\n")
            f.write(f"Time Elapsed: {prompt_elapsed_time:.2f}s ({prompt_elapsed_minutes:.2f}min)\n")
            f.write("=" * 80 + "\n\n")
        
        print(f"Best prompt saved to {best_prompts_file}")
        
        if best_score <= THRESHOLD and PASS == False:
            fail_cases.append({
                'prompt_idx': prompt_idx,
                'original_prompt': prompt,
                'best_adv_prompt': best_adv_prompt,
                'best_response': cur_prompt_resp[best_adv_prompt_idx],
                'best_score': best_score,
                'threshold': THRESHOLD
            })
        
        print(f"Response: {cur_prompt_resp[best_adv_prompt_idx]}")
        print()
        print(f"Adv prompt: {best_adv_prompt}")
        print(f"Best prompt id: {best_adv_prompt_idx}")
        print(f"Scores: {scores[best_adv_prompt_idx]}, Total JB: {JB}")
        print(f"AVG JB prompts per round: {Total_JB_prompt/Total_JB_round if Total_JB_round > 0 else 0:.2f}")
        print(f"Total JB rounds: {Total_JB_round}")

        avg_jb_rate = JB / (prompt_idx - start_idx + 1) if (prompt_idx - start_idx + 1) > 0 else 0
        print(f"Average JB rate: {avg_jb_rate:.4f} ({JB}/{prompt_idx - start_idx + 1})")
        print("-" * 80)

        final_adv_prompts.append(best_adv_prompt)

    total_run_elapsed = time.time() - total_run_start_time
    total_run_minutes = total_run_elapsed / 60.0

    # Save adversarial data summary
    with open(f"{run_dir}/adversarial_data_summary.txt", "w") as f:
        f.write("Adversarial Data Summary (7D+2Typo Ablation)\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("ALL ADVERSARIAL ATTEMPTS:\n")
        f.write("-" * 30 + "\n")
        for i in range(len(all_adversarial_data['indices'])):
            f.write(f"Index: {all_adversarial_data['indices'][i]}\n")
            f.write(f"Original Prompt: {all_adversarial_data['original_prompts'][i]}\n")
            f.write(f"Adversarial Prompt: {all_adversarial_data['adversarial_prompts'][i]}\n")
            f.write(f"Response: {all_adversarial_data['responses'][i]}\n")
            f.write(f"Score: {all_adversarial_data['scores'][i]}\n")
            f.write(f"Loss: {all_adversarial_data['losses'][i]}\n")
            f.write(f"Query Count: {all_adversarial_data['query_counts'][i]}\n")
            f.write(f"Prompt ID: {all_adversarial_data['prompt_ids'][i]}\n")
            f.write("-" * 30 + "\n\n")
        
        f.write("\n" + "=" * 50 + "\n")
        f.write("BEST ADVERSARIAL PROMPTS BY ORIGINAL PROMPT:\n")
        f.write("=" * 50 + "\n\n")
        
        prompt_groups = {}
        for i in range(len(all_adversarial_data['indices'])):
            prompt_id = all_adversarial_data['prompt_ids'][i]
            if prompt_id not in prompt_groups:
                prompt_groups[prompt_id] = []
            prompt_groups[prompt_id].append(i)
        
        for prompt_id in sorted(prompt_groups.keys()):
            group_indices = prompt_groups[prompt_id]
            best_idx = max(group_indices, key=lambda x: all_adversarial_data['scores'][x])
            
            f.write(f"Original Prompt ID: {prompt_id}\n")
            f.write(f"Original Prompt: {all_adversarial_data['original_prompts'][best_idx]}\n")
            f.write(f"Best Adversarial Prompt: {all_adversarial_data['adversarial_prompts'][best_idx]}\n")
            f.write(f"Best Response: {all_adversarial_data['responses'][best_idx]}\n")
            f.write(f"Best Score: {all_adversarial_data['scores'][best_idx]}\n")
            f.write("-" * 30 + "\n\n")
    
    print(f"Summary saved to {run_dir}/adversarial_data_summary.txt")
    
    if fail_cases:
        print("\n" + "=" * 80)
        print("BEST PROMPTS FOR FAIL CASES (JAILBREAKING UNSUCCESSFUL)")
        print("=" * 80)
        print(f"Total fail cases: {len(fail_cases)}")
        print(f"Threshold used: {THRESHOLD}")
        print()
        
        for i, fail_case in enumerate(fail_cases):
            print(f"FAIL CASE {i+1}:")
            print(f"Prompt Index: {fail_case['prompt_idx']}")
            print(f"Original Prompt: {fail_case['original_prompt']}")
            print(f"Best Adversarial Prompt: {fail_case['best_adv_prompt']}")
            print(f"Best Response: {fail_case['best_response']}")
            print(f"Best Score: {fail_case['best_score']:.4f} (Threshold: {fail_case['threshold']})")
            print("-" * 80)
        
        with open(f"{run_dir}/fail_cases_summary.txt", "w") as f:
            f.write("BEST PROMPTS FOR FAIL CASES (7D+2Typo ABLATION)\n")
            f.write("=" * 80 + "\n")
            f.write(f"Total fail cases: {len(fail_cases)}\n")
            f.write(f"Threshold used: {THRESHOLD}\n\n")
            
            for i, fail_case in enumerate(fail_cases):
                f.write(f"FAIL CASE {i+1}:\n")
                f.write(f"Prompt Index: {fail_case['prompt_idx']}\n")
                f.write(f"Original Prompt: {fail_case['original_prompt']}\n")
                f.write(f"Best Adversarial Prompt: {fail_case['best_adv_prompt']}\n")
                f.write(f"Best Response: {fail_case['best_response']}\n")
                f.write(f"Best Score: {fail_case['best_score']:.4f} (Threshold: {fail_case['threshold']})\n")
                f.write("-" * 80 + "\n\n")
        
        print(f"Fail cases summary saved to {run_dir}/fail_cases_summary.txt")
    else:
        print("\n" + "=" * 80)
        print("NO FAIL CASES - ALL PROMPTS SUCCESSFULLY JAILBROKEN!")
        print("=" * 80)

    # Save best adversarial prompts for each datapoint
    print("\n" + "=" * 80)
    print("SAVING BEST ADVERSARIAL PROMPTS FOR EACH DATAPOINT")
    print("=" * 80)
    
    best_adv_prompts_by_datapoint = []
    
    prompt_groups = {}
    for i in range(len(all_adversarial_data['indices'])):
        prompt_id = all_adversarial_data['prompt_ids'][i]
        if prompt_id not in prompt_groups:
            prompt_groups[prompt_id] = []
        prompt_groups[prompt_id].append(i)
    
    for prompt_id in sorted(prompt_groups.keys()):
        group_indices = prompt_groups[prompt_id]
        best_idx = max(group_indices, key=lambda x: all_adversarial_data['scores'][x])
        
        best_adv_prompts_by_datapoint.append({
            'datapoint_id': prompt_id,
            'original_prompt': all_adversarial_data['original_prompts'][best_idx],
            'best_adversarial_prompt': all_adversarial_data['adversarial_prompts'][best_idx],
            'best_response': all_adversarial_data['responses'][best_idx],
            'best_score': all_adversarial_data['scores'][best_idx],
            'query_count': all_adversarial_data['query_counts'][best_idx],
            'loss': all_adversarial_data['losses'][best_idx]
        })
    
    best_prompts_file = f"{run_dir}/best_adv_prompts_by_datapoint.txt"
    with open(best_prompts_file, "w") as f:
        f.write("BEST ADVERSARIAL PROMPTS FOR EACH DATAPOINT (7D+2Typo ABLATION)\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Total datapoints processed: {len(best_adv_prompts_by_datapoint)}\n")
        f.write(f"Threshold used: {THRESHOLD}\n\n")
        
        for i, datapoint_data in enumerate(best_adv_prompts_by_datapoint):
            f.write(f"DATAPOINT {i+1}:\n")
            f.write(f"Datapoint ID: {datapoint_data['datapoint_id']}\n")
            f.write(f"Original Prompt: {datapoint_data['original_prompt']}\n")
            f.write(f"Best Adversarial Prompt: {datapoint_data['best_adversarial_prompt']}\n")
            f.write(f"Best Response: {datapoint_data['best_response']}\n")
            f.write(f"Best Score: {datapoint_data['best_score']:.4f}\n")
            f.write(f"Query Count: {datapoint_data['query_count']}\n")
            f.write(f"Loss: {datapoint_data['loss']:.4f}\n")
            f.write("-" * 80 + "\n\n")
    
    print(f"Best adversarial prompts saved to {best_prompts_file}")
    print(f"Total datapoints with best prompts: {len(best_adv_prompts_by_datapoint)}")

    # Final statistics
    print("\n" + "=" * 80)
    print("FINAL OVERALL STATISTICS (7D+2Typo ABLATION)")
    print("=" * 80)
    prompts_processed = end_idx - start_idx + 1 if end_idx >= start_idx else 0
    overall_jb_rate = JB / prompts_processed if prompts_processed > 0 else 0
    avg_prompts_per_round = Total_JB_prompt / Total_JB_round if Total_JB_round > 0 else 0
    avg_rounds_per_prompt = Total_JB_round / prompts_processed if prompts_processed > 0 else 0
    
    print(f"Total prompts processed: {prompts_processed}")
    print(f"Total successful jailbreaks: {JB}")
    print(f"Overall jailbreak success rate: {overall_jb_rate:.4f} ({JB}/{prompts_processed})")
    print(f"Total rounds executed: {Total_JB_round}")
    print(f"Total prompts generated: {Total_JB_prompt}")
    print(f"Average prompts per round: {avg_prompts_per_round:.2f}")
    print(f"Average rounds per prompt: {avg_rounds_per_prompt:.2f}")
    print(f"Total run time: {total_run_elapsed:.2f}s ({total_run_minutes:.2f}min)")

    # Save CSV
    print("\n" + "=" * 80)
    print("SAVING CSV FILE WITH ORIGINAL PROMPTS AND BEST ADVERSARIAL RESPONSES")
    print("=" * 80)
    
    csv_data = []
    for datapoint_data in best_adv_prompts_by_datapoint:
        csv_data.append({
            'original_prompt': datapoint_data['original_prompt'],
            'best_adversarial_response': datapoint_data['best_response']
        })
    
    csv_df = pd.DataFrame(csv_data)
    csv_file_path = f"{run_dir}/original_prompts_vs_best_responses.csv"
    csv_df.to_csv(csv_file_path, index=False)
    
    print(f"CSV file saved to: {csv_file_path}")
    print(f"CSV contains {len(csv_data)} rows with original prompts and best adversarial responses")

if __name__ == "__main__":
    main()
