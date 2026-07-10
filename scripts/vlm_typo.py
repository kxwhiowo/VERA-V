import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
import torch
import wandb
import random
import os, sys
import json
import argparse
import time
# Keep project-root imports working when this script lives under scripts/
from pathlib import Path as _Path
_REPO_ROOT = _Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from utils import find_or_create_run_dir
from utils.json_parser import parse_adversarial_json_output, validate_prompts
from utils.image_utils import text_to_art_image_pil, concatenate_images_with_distraction
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

@hydra.main(config_path="../configs", config_name="vlm_dual", version_base=None)
def main(cfg):
    # # Parse CS-DJ configuration
    # cs_dj_parser = CS_DJ_parser()
    # cs_dj_args = cs_dj_parser.parse_args([])
    cfg_omega = OmegaConf.create(cfg)
    run_dir, is_resume = find_or_create_run_dir(cfg_omega)
    target_vlm = instantiate(cfg.target)
    print("Loaded target VLM")
    print("Loading models...")
    attacker = instantiate(cfg.attacker)
    print("Loaded attacker")
    judge = instantiate(cfg.judge)
    print("Loaded judge")
    # ext_judge = instantiate(cfg.ext_judge)
    # print("Loaded ext_judge")
    diffusion_model = instantiate(cfg.diffusion)
    print("Loaded diffusion model")

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
    
    # Training loop
    log_every = cfg.log_every
    JB = 0
    prompts_total = []
    responses_total = []
    scores = []
    final_adv_prompts = []
    # fail_cases = []
    start_idx = 2
    end_idx = 3
    Total_JB_prompt = 0
    Total_JB_round = 0

    SUCCESS = 0
    # Change this to the distraction image map you created (see README).
    select_img_path = str(
        _REPO_ROOT / "CS_DJ_main/image_map/harmbench_distraction_image_map_seed_0_num_10000.json"
    )
    
    # Load distraction images mapping
    distraction_images_map = {}
    if os.path.exists(select_img_path):
        with open(select_img_path, 'r', encoding='utf-8') as f:
            distraction_images_map = json.load(f)
        print(f"Loaded distraction images map from {select_img_path}")
    else:
        print(f"Warning: Distraction images map not found at {select_img_path}")
    
    # Initialize best prompts file (clear it for fresh run)
    best_prompts_file = f"{run_dir}/best_prompts_incremental.txt"
    with open(best_prompts_file, "w") as f:
        f.write("BEST PROMPTS INCREMENTAL LOG\n")
        f.write("=" * 80 + "\n")
        f.write(f"Run Directory: {run_dir}\n")
        f.write(f"Start Index: {start_idx}, End Index: {end_idx}\n")
        f.write(f"Started at: {pd.Timestamp.now()}\n")
        f.write("=" * 80 + "\n\n")
        
    for prompt_idx, (prompt, category) in enumerate(zip(prompts, category)):

        if prompt_idx < start_idx:
            continue
        if prompt_idx > end_idx:
            break
        attacker.initialize_attacker()
        optimizer = instantiate(cfg.optimizer, attacker.parameters())
        
        cur_prompt_adv = []
        cur_prompt_resp = []
        cur_images = []  # Store all concatenated images
        scores = []

        max_queries = cfg.max_queries
        THRESHOLD = cfg.threshold
        BATCH_SIZE = cfg.batch_size
        CUR_COUNT = 0
        total_attempts = 0
        refinement_limit = 1.5 * max_queries
        batch_prompt = [prompt for _ in range(BATCH_SIZE)]
        NON_UPDATE_THRESHOLD = 0.4
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

            image_prompts, text_prompts = parse_adversarial_json_output(adversarial_queries, prompt)
            
            print("Log-likelihoods", log_likelihoods)
            print("Log-prior", log_prior)
            diffusion_images = diffusion_model.generate_batch(image_prompts)
            
            distraction_image_paths = []
            if prompt in distraction_images_map:
                distraction_image_paths = distraction_images_map[prompt][:7]  # Take first 7 images
            else:
                print(f"Warning: No distraction images found for prompt: {prompt}")
                exit()
                # Use empty list if no distraction images found
                distraction_image_paths = []
            
            # Generate concatenated images for each batch item
            font_path = str(_REPO_ROOT / "Super Moods.ttf")
            concatenated_images = []
            
            for i in range(len(text_prompts)):
                # Use corresponding diffusion image or None if not available
                diffusion_image = diffusion_images[i] if i < len(diffusion_images) else None
                
                # Generate typo image for this specific text prompt
                typo_image = text_to_art_image_pil(text_prompts[i] if i < len(text_prompts) else "", 
                                              font_path=font_path, font_size=80, 
                                              text_color=(0, 0, 0), bg_color=(255, 255, 255), 
                                              image_width=800)

                cancatenate_image = concatenate_images_with_distraction(
                    distraction_image_paths, 
                    diffusion_image, 
                    typo_image,
                    images_per_row=3, 
                    target_size=(300, 300)
                )
                concatenated_images.append(cancatenate_image)
            
            # Ensure save directory exists for concatenated images
            # save_dir = os.path.join(run_dir, "concatenated_images")
            # os.makedirs(save_dir, exist_ok=True)
            
            print(f"Generated {len(concatenated_images)} concatenated images for batch size {len(text_prompts)}")
            loss_dct = loss_fn(
                adversarial_queries=text_prompts, 
                log_likelihoods=log_likelihoods,
                entropies=entropy,
                log_prior=log_prior,
                original_queries=batch_prompt,
                images=concatenated_images,  # Pass the full list of concatenated images to match batch size
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
            cur_images += concatenated_images  # Store concatenated images
            
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

            # Optimizer step and PASS if any score crosses THRESHOLD or any response is unsafe
            trigger_update = (np.max(batch_scores) >= THRESHOLD)
            if trigger_update:
                print("Above-threshold response detected, updating and finishing current prompt")
                # optimizer.step()
                JB = JB + 1
                PASS = True

                print("------------------------------------------------------------------------------------")
                break

            CUR_COUNT += BATCH_SIZE
            # optimizer.step()
            Total_JB_prompt += BATCH_SIZE
            # print(f"Fixed textual prompt: {fixed_textual_prompts[prompt_idx]}")
            # print(f"Parsed text prompt: {text_prompts[0] if text_prompts else 'None'}")
            print("KL:", loss_dct['kl'], "entropy:" , loss_dct['entropy'])
            for q, r in zip(adversarial_queries, loss_dct['responses']):
                print("Adv Query:", q)
                print("VLM Response:\n", r)
                print("--------------------------------")


        prompt_elapsed_time = time.time() - prompt_start_time
        prompt_elapsed_minutes = prompt_elapsed_time / 60.0

        if PASS == True:
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

        # Save the best prompt immediately to a file
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
        
        print(f"Response: {cur_prompt_resp[best_adv_prompt_idx]}")
        print()
        print(f"Adv prompt: {best_adv_prompt}")
        print(f"Best prompt id: {best_adv_prompt_idx}")
        print(f"Scores: {scores[best_adv_prompt_idx]}, Total JB: {JB}")
        print(f"AVG JB prompts per round: {Total_JB_prompt/Total_JB_round if Total_JB_round > 0 else 0:.2f}")
        print(f"Total JB rounds: {Total_JB_round}")

        avg_jb_rate = JB / (prompt_idx - start_idx + 1) if (prompt_idx - start_idx + 1) > 0 else 0
        print(f"Average JB rate: {avg_jb_rate:.4f} ({JB}/{prompt_idx - start_idx + 1})")
        print("------------------------------------------------------------------------------------")

        final_adv_prompts.append(best_adv_prompt)
        # if torch.cuda.is_available():
        #     torch.cuda.empty_cache()
  
    
    # Save adversarial data summary (keeping the log functionality)
    with open(f"{run_dir}/adversarial_data_summary.txt", "w") as f:
        f.write("Adversarial Data Summary\n")
        f.write("=" * 50 + "\n\n")
        
        # First, write all individual adversarial attempts
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
            # f.write(f"Image File: image_{all_adversarial_data['indices'][i]:06d}.png\n")
            f.write("-" * 30 + "\n\n")
        
        # Then, write the best adversarial prompt for each original prompt
        f.write("\n" + "=" * 50 + "\n")
        f.write("BEST ADVERSARIAL PROMPTS BY ORIGINAL PROMPT:\n")
        f.write("=" * 50 + "\n\n")
        
        # Group by prompt_id and find the best score for each
        prompt_groups = {}
        for i in range(len(all_adversarial_data['indices'])):
            prompt_id = all_adversarial_data['prompt_ids'][i]
            if prompt_id not in prompt_groups:
                prompt_groups[prompt_id] = []
            prompt_groups[prompt_id].append(i)
        
        # For each prompt group, find the best adversarial prompt (by highest score)
        for prompt_id in sorted(prompt_groups.keys()):
            group_indices = prompt_groups[prompt_id]
            best_idx = max(group_indices, key=lambda x: all_adversarial_data['scores'][x])
            
            f.write(f"Original Prompt ID: {prompt_id}\n")
            f.write(f"Original Prompt: {all_adversarial_data['original_prompts'][best_idx]}\n")
            f.write(f"Best Adversarial Prompt: {all_adversarial_data['adversarial_prompts'][best_idx]}\n")
            f.write(f"Best Response: {all_adversarial_data['responses'][best_idx]}\n")
            f.write(f"Best Score: {all_adversarial_data['scores'][best_idx]}\n")
            # f.write(f"Image File: image_{all_adversarial_data['indices'][best_idx]:06d}.png\n")
            f.write("-" * 30 + "\n\n")
    

    print(f"Summary saved to {run_dir}/adversarial_data_summary.txt")
    
   

    # Print final overall statistics
    print("\n" + "=" * 80)
    print("FINAL OVERALL STATISTICS")
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


if __name__ == "__main__":
    main()