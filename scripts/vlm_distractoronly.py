import hydra
import numpy as np
from omegaconf import OmegaConf
from hydra.utils import instantiate
import torch
import random
import os
import sys
import json
import time
# Keep project-root imports working when this script lives under scripts/
from pathlib import Path as _Path
_REPO_ROOT = _Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from utils import find_or_create_run_dir
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
from dotenv import load_dotenv


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(1)
load_dotenv()


def concatenate_distractor_images(distraction_image_paths, images_per_row=3,
                                   target_size=(300, 300), fill_color=(255, 255, 255),
                                   font_size=20, rotation_angle=0):
    """Concatenate up to 9 distractor images into a numbered grid (3x3)."""
    images = []

    try:
        font = ImageFont.truetype(
            str(_REPO_ROOT / "Super Moods.ttf"), font_size)
    except IOError:
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except IOError:
            font = ImageFont.load_default()

    for idx, img_path in enumerate(distraction_image_paths[:9]):
        if img_path.startswith('./llava_images/'):
            filename = img_path.replace('./llava_images/', '')
            img_path = str(_REPO_ROOT / "CS_DJ_main" / "llava_images" / filename)

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


@hydra.main(config_path="../configs", config_name="vlm_distractoronly", version_base=None)
def main(cfg):
    cfg_omega = OmegaConf.create(cfg)
    run_dir, is_resume = find_or_create_run_dir(cfg_omega)

    target_vlm = instantiate(cfg.target)
    print("Loaded target VLM")

    judge = instantiate(cfg.judge)
    print("Loaded judge")

    prompts, categories = instantiate(cfg.data)
    print("Loaded data")

    THRESHOLD = cfg.threshold
    start_idx = cfg.get("start_idx", 0)
    end_idx = cfg.get("end_idx", 4)

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

    all_results = []
    JB = 0
    fail_cases = []
    total_time = 0.0

    log_file = f"{run_dir}/results_log.txt"
    with open(log_file, "w") as f:
        f.write("DISTRACTOR-ONLY ABLATION RESULTS\n")
        f.write("=" * 80 + "\n")
        f.write(f"Run Directory: {run_dir}\n")
        f.write(f"Threshold: {THRESHOLD}\n")
        f.write(f"Start Index: {start_idx}, End Index: {end_idx}\n")
        f.write(f"Started at: {pd.Timestamp.now()}\n")
        f.write("=" * 80 + "\n\n")

    save_dir = os.path.join(run_dir, "concatenated_images")
    os.makedirs(save_dir, exist_ok=True)

    for prompt_idx, (prompt, category) in enumerate(zip(prompts, categories)):
        if prompt_idx < start_idx:
            continue
        if prompt_idx > end_idx:
            break

        print("=" * 60)
        print(f"Prompt {prompt_idx}: {prompt}")
        prompt_start = time.time()

        if prompt in distraction_images_map:
            distraction_image_paths = distraction_images_map[prompt][:9]
        else:
            print(f"Warning: No distraction images found for prompt: {prompt}")
            exit()

        concatenated_image = concatenate_distractor_images(
            distraction_image_paths,
            images_per_row=3,
            target_size=(300, 300),
        )

        responses = target_vlm.batch_respond([prompt], [concatenated_image])
        response = responses[0]

        with torch.no_grad():
            judge_scores = judge.batch_judge_train([prompt], [response])
        score = judge_scores.exp().cpu().item()

        PASS = score >= THRESHOLD
        if PASS:
            JB += 1

        image_filename = f"distractor_image_prompt_{prompt_idx:04d}_score_{score:.4f}.png"
        image_path = os.path.join(save_dir, image_filename)
        concatenated_image.save(image_path)

        prompt_elapsed = time.time() - prompt_start
        total_time += prompt_elapsed
        processed = prompt_idx - start_idx + 1
        avg_time = total_time / processed

        result = {
            'prompt_idx': prompt_idx,
            'original_prompt': prompt,
            'category': category,
            'response': response,
            'score': score,
            'jailbreak_success': PASS,
            'time_seconds': prompt_elapsed,
        }
        all_results.append(result)

        with open(log_file, "a") as f:
            f.write(f"PROMPT {prompt_idx}:\n")
            f.write(f"Original Prompt: {prompt}\n")
            f.write(f"Response: {response}\n")
            f.write(f"Score: {score:.4f}\n")
            f.write(f"Jailbreak: {'YES' if PASS else 'NO'}\n")
            f.write(f"Running JB Rate: {JB}/{processed} = {JB/processed:.4f}\n")
            f.write(f"Time: {prompt_elapsed:.2f}s | Total: {total_time:.2f}s | Avg: {avg_time:.2f}s\n")
            f.write("=" * 80 + "\n\n")

        if not PASS:
            fail_cases.append(result)

        print(f"Response: {response}")
        print(f"Score: {score:.4f}, JB: {'YES' if PASS else 'NO'}")
        print(f"Running JB rate: {JB/processed:.4f} ({JB}/{processed})")
        print(f"Time: {prompt_elapsed:.2f}s | Total: {total_time:.2f}s | Avg: {avg_time:.2f}s")
        print("-" * 60)

    # ---- final statistics ----
    prompts_processed = end_idx - start_idx + 1 if end_idx >= start_idx else 0
    overall_jb_rate = JB / prompts_processed if prompts_processed > 0 else 0

    final_avg_time = total_time / prompts_processed if prompts_processed > 0 else 0

    print("\n" + "=" * 80)
    print("FINAL STATISTICS (DISTRACTOR-ONLY ABLATION)")
    print("=" * 80)
    print(f"Total prompts processed: {prompts_processed}")
    print(f"Total successful jailbreaks: {JB}")
    print(f"Overall JB rate: {overall_jb_rate:.4f} ({JB}/{prompts_processed})")
    print(f"Total time: {total_time:.2f}s | Avg per prompt: {final_avg_time:.2f}s")

    if fail_cases:
        with open(f"{run_dir}/fail_cases.txt", "w") as f:
            f.write("FAIL CASES (DISTRACTOR-ONLY ABLATION)\n")
            f.write("=" * 80 + "\n")
            f.write(f"Total fail cases: {len(fail_cases)}\n")
            f.write(f"Threshold: {THRESHOLD}\n\n")
            for i, fc in enumerate(fail_cases):
                f.write(f"FAIL {i+1}: Prompt {fc['prompt_idx']}\n")
                f.write(f"Prompt: {fc['original_prompt']}\n")
                f.write(f"Response: {fc['response']}\n")
                f.write(f"Score: {fc['score']:.4f}\n")
                f.write("-" * 80 + "\n\n")
        print(f"Fail cases saved to {run_dir}/fail_cases.txt")
    else:
        print("NO FAIL CASES - ALL PROMPTS SUCCESSFULLY JAILBROKEN!")

    csv_data = [{
        'original_prompt': r['original_prompt'],
        'response': r['response'],
        'score': r['score'],
        'jailbreak': r['jailbreak_success'],
        'time_seconds': r['time_seconds'],
    } for r in all_results]
    csv_df = pd.DataFrame(csv_data)
    csv_path = f"{run_dir}/distractor_only_results.csv"
    csv_df.to_csv(csv_path, index=False)
    print(f"CSV saved to: {csv_path}")
    print(f"CSV contains {len(csv_data)} rows")


if __name__ == "__main__":
    main()
