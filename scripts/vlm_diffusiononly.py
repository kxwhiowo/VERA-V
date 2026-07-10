import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
import torch
import wandb
import random
import os
import sys
import json
# Keep project-root imports working when this script lives under scripts/
from pathlib import Path as _Path
_REPO_ROOT = _Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from utils import find_or_create_run_dir
from PIL import Image
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


@hydra.main(config_path="../configs", config_name="vlm_diffusiononly", version_base=None)
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

    log_every = cfg.log_every
    JB = 0
    prompts_total = []
    responses_total = []
    scores = []
    final_adv_prompts = []
    fail_cases = []
    start_idx = 0
    end_idx = 49
    Total_JB_prompt = 0
    Total_JB_round = 0
    SUCCESS = 0

    best_prompts_file = f"{run_dir}/best_prompts_incremental.txt"
    with open(best_prompts_file, "w") as f:
        f.write("BEST PROMPTS INCREMENTAL LOG (DIFFUSION-ONLY ABLATION)\n")
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
        cur_images = []
        scores = []

        max_queries = cfg.max_queries
        THRESHOLD = cfg.threshold
        BATCH_SIZE = cfg.batch_size
        CUR_COUNT = 0
        total_attempts = 0
        batch_prompt = [prompt for _ in range(BATCH_SIZE)]
        PASS = False

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

            print("Log-likelihoods", log_likelihoods)
            print("Log-prior", log_prior)

            # VLMAttacker outputs image prompts directly — feed them to diffusion
            diffusion_images = diffusion_model.generate_batch(adversarial_queries)

            save_dir = os.path.join(run_dir, "diffusion_images")
            os.makedirs(save_dir, exist_ok=True)

            print(f"Generated {len(diffusion_images)} diffusion images for batch size {BATCH_SIZE}")

            # Pass diffusion images directly to the target VLM (no concatenation)
            loss_dct = loss_fn(
                adversarial_queries=adversarial_queries,
                log_likelihoods=log_likelihoods,
                entropies=entropy,
                log_prior=log_prior,
                original_queries=batch_prompt,
                images=diffusion_images,
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
            cur_images += diffusion_images

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

                best_batch_idx = int(np.argmax(batch_scores))
                best_batch_image = diffusion_images[best_batch_idx]
                best_batch_score = batch_scores[best_batch_idx]

                image_filename = f"best_diffusion_image_prompt_{prompt_idx:04d}_score_{best_batch_score:.4f}_threshold_passed.png"
                image_path = os.path.join(save_dir, image_filename)
                best_batch_image.save(image_path)
                print(f"Saved best diffusion image (threshold passed) to: {image_path}")

                print("-" * 80)
                break

            CUR_COUNT += BATCH_SIZE
            Total_JB_prompt += BATCH_SIZE
            print("KL:", loss_dct['kl'], "entropy:", loss_dct['entropy'])
            for q, r in zip(adversarial_queries, loss_dct['responses']):
                print("Image Prompt:", q)
                print("VLM Response:\n", r)
                print("--------------------------------")

        if PASS:
            print(f"Successfully finished prompt {prompt}")
        else:
            print(f"Failed to jailbreak prompt {prompt}")

        best_adv_prompt_idx = int(np.argmax(scores))
        print("best_adv_prompt_idx:", best_adv_prompt_idx)
        best_adv_prompt = cur_prompt_adv[best_adv_prompt_idx]
        best_score = scores[best_adv_prompt_idx]
        prompts_total.append(best_adv_prompt)
        responses_total.append(cur_prompt_resp[best_adv_prompt_idx])

        if len(cur_images) > 0 and best_adv_prompt_idx < len(cur_images):
            save_dir = os.path.join(run_dir, "diffusion_images")
            os.makedirs(save_dir, exist_ok=True)

            best_image = cur_images[best_adv_prompt_idx]
            image_filename = f"best_diffusion_image_prompt_{prompt_idx:04d}_idx_{best_adv_prompt_idx:04d}_score_{best_score:.4f}.png"
            image_path = os.path.join(save_dir, image_filename)
            best_image.save(image_path)
            print(f"Saved best diffusion image (overall) to: {image_path}")

        best_prompts_file = f"{run_dir}/best_prompts_incremental.txt"
        with open(best_prompts_file, "a") as f:
            f.write(f"PROMPT {prompt_idx} (Index: {prompt_idx - start_idx + 1}):\n")
            f.write(f"Timestamp: {pd.Timestamp.now()}\n")
            f.write(f"Original Prompt: {prompt}\n")
            f.write(f"Best Image Prompt: {best_adv_prompt}\n")
            f.write(f"Best Response: {cur_prompt_resp[best_adv_prompt_idx]}\n")
            f.write(f"Best Score: {best_score:.4f}\n")
            f.write(f"Threshold: {THRESHOLD}\n")
            f.write(f"Query Count: {CUR_COUNT}\n")
            f.write(f"Jailbreak Success: {'YES' if best_score > THRESHOLD else 'NO'}\n")
            f.write(f"Current JB Rate: {JB}/{prompt_idx - start_idx + 1} = {JB/(prompt_idx - start_idx + 1):.4f}\n")
            f.write("=" * 80 + "\n\n")

        print(f"Best prompt saved to {best_prompts_file}")

        if best_score <= THRESHOLD and not PASS:
            fail_cases.append({
                'prompt_idx': prompt_idx,
                'original_prompt': prompt,
                'best_image_prompt': best_adv_prompt,
                'best_response': cur_prompt_resp[best_adv_prompt_idx],
                'best_score': best_score,
                'threshold': THRESHOLD
            })

        print(f"Response: {cur_prompt_resp[best_adv_prompt_idx]}")
        print()
        print(f"Image Prompt: {best_adv_prompt}")
        print(f"Best prompt id: {best_adv_prompt_idx}")
        print(f"Scores: {scores[best_adv_prompt_idx]}, Total JB: {JB}")
        print(f"AVG JB prompts per round: {Total_JB_prompt/Total_JB_round if Total_JB_round > 0 else 0:.2f}")
        print(f"Total JB rounds: {Total_JB_round}")

        avg_jb_rate = JB / (prompt_idx - start_idx + 1) if (prompt_idx - start_idx + 1) > 0 else 0
        print(f"Average JB rate: {avg_jb_rate:.4f} ({JB}/{prompt_idx - start_idx + 1})")
        print("-" * 80)

        final_adv_prompts.append(best_adv_prompt)

    # Save adversarial data summary
    with open(f"{run_dir}/adversarial_data_summary.txt", "w") as f:
        f.write("Adversarial Data Summary (Diffusion-Only Ablation)\n")
        f.write("=" * 50 + "\n\n")

        f.write("ALL ADVERSARIAL ATTEMPTS:\n")
        f.write("-" * 30 + "\n")
        for i in range(len(all_adversarial_data['indices'])):
            f.write(f"Index: {all_adversarial_data['indices'][i]}\n")
            f.write(f"Original Prompt: {all_adversarial_data['original_prompts'][i]}\n")
            f.write(f"Image Prompt: {all_adversarial_data['adversarial_prompts'][i]}\n")
            f.write(f"Response: {all_adversarial_data['responses'][i]}\n")
            f.write(f"Score: {all_adversarial_data['scores'][i]}\n")
            f.write(f"Loss: {all_adversarial_data['losses'][i]}\n")
            f.write(f"Query Count: {all_adversarial_data['query_counts'][i]}\n")
            f.write(f"Prompt ID: {all_adversarial_data['prompt_ids'][i]}\n")
            f.write("-" * 30 + "\n\n")

        f.write("\n" + "=" * 50 + "\n")
        f.write("BEST IMAGE PROMPTS BY ORIGINAL PROMPT:\n")
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
            f.write(f"Best Image Prompt: {all_adversarial_data['adversarial_prompts'][best_idx]}\n")
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
            print(f"Best Image Prompt: {fail_case['best_image_prompt']}")
            print(f"Best Response: {fail_case['best_response']}")
            print(f"Best Score: {fail_case['best_score']:.4f} (Threshold: {fail_case['threshold']})")
            print("-" * 80)

        with open(f"{run_dir}/fail_cases_summary.txt", "w") as f:
            f.write("BEST PROMPTS FOR FAIL CASES (DIFFUSION-ONLY ABLATION)\n")
            f.write("=" * 80 + "\n")
            f.write(f"Total fail cases: {len(fail_cases)}\n")
            f.write(f"Threshold used: {THRESHOLD}\n\n")

            for i, fail_case in enumerate(fail_cases):
                f.write(f"FAIL CASE {i+1}:\n")
                f.write(f"Prompt Index: {fail_case['prompt_idx']}\n")
                f.write(f"Original Prompt: {fail_case['original_prompt']}\n")
                f.write(f"Best Image Prompt: {fail_case['best_image_prompt']}\n")
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
    print("SAVING BEST IMAGE PROMPTS FOR EACH DATAPOINT")
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
            'best_image_prompt': all_adversarial_data['adversarial_prompts'][best_idx],
            'best_response': all_adversarial_data['responses'][best_idx],
            'best_score': all_adversarial_data['scores'][best_idx],
            'query_count': all_adversarial_data['query_counts'][best_idx],
            'loss': all_adversarial_data['losses'][best_idx]
        })

    best_prompts_file = f"{run_dir}/best_image_prompts_by_datapoint.txt"
    with open(best_prompts_file, "w") as f:
        f.write("BEST IMAGE PROMPTS FOR EACH DATAPOINT (DIFFUSION-ONLY ABLATION)\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Total datapoints processed: {len(best_adv_prompts_by_datapoint)}\n")
        f.write(f"Threshold used: {THRESHOLD}\n\n")

        for i, datapoint_data in enumerate(best_adv_prompts_by_datapoint):
            f.write(f"DATAPOINT {i+1}:\n")
            f.write(f"Datapoint ID: {datapoint_data['datapoint_id']}\n")
            f.write(f"Original Prompt: {datapoint_data['original_prompt']}\n")
            f.write(f"Best Image Prompt: {datapoint_data['best_image_prompt']}\n")
            f.write(f"Best Response: {datapoint_data['best_response']}\n")
            f.write(f"Best Score: {datapoint_data['best_score']:.4f}\n")
            f.write(f"Query Count: {datapoint_data['query_count']}\n")
            f.write(f"Loss: {datapoint_data['loss']:.4f}\n")
            f.write("-" * 80 + "\n\n")

    print(f"Best image prompts saved to {best_prompts_file}")
    print(f"Total datapoints with best prompts: {len(best_adv_prompts_by_datapoint)}")

    # Final overall statistics
    print("\n" + "=" * 80)
    print("FINAL OVERALL STATISTICS (DIFFUSION-ONLY ABLATION)")
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
