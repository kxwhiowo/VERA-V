# VERA-V \[ICML 2026\]

## Abstract

Vision-Language Models (VLMs) extend large language models with visual reasoning, but their multimodal design also introduces new, underexplored vulnerabilities. Existing multimodal red-teaming methods largely rely on brittle templates, focus on single-attack settings, and expose only a narrow subset of vulnerabilities. To address these limitations, we introduce VERA-V, a variational inference framework that recasts multimodal jailbreak discovery as learning a joint posterior distribution over paired text-image prompts. This probabilistic view captures complex cross-modal interactions, enabling stealthy, coordinated adversarial inputs that bypass model guardrails. We train a lightweight attacker to approximate the posterior, allowing efficient sampling of diverse jailbreaks and providing distributional insights into vulnerabilities. VERA-V further integrates three complementary strategies: (i) typography-based text prompts that embed harmful cues, (ii) diffusion-based image synthesis that introduces adversarial signals, and (iii) structured distractors to fragment VLM attention. Experiments on HarmBench and HADES benchmarks show that VERA-V consistently outperforms state-of-the-art baselines on both open-source and frontier VLMs, achieving up to 53.75% higher attack success rate (ASR) over the best baseline on GPT-4o.

## Setup



### Environment

Create and activate the conda environment from the provided spec:

```bash
conda env create -f verav.yml
conda activate veravlm
```

Run all commands from the project root (`VERA-V/`).

### API keys

If you use API-based targets or judges (e.g. `gpt4o`, `gpt_judge`), copy the example env file and fill in your key:

```bash
cp .env.example .env
# then edit .env:
OPENAI_API_KEY=your_key_here
```

The scripts call `load_dotenv()` at startup. Do not commit `.env`.

### Data and assets

- **HarmBench prompts** are included at `data/behavior_datasets/harmbench_behaviors_text_all.csv`.
- **Typo font** is referenced in `utils/image_utils.py` (`Super Moods.ttf` at the project root). Place the font there, or update `font_path` if it lives elsewhere.
- **Distraction image map** (required by scripts that use CS-DJ distractors) is expected at:
`CS_DJ_main/image_map/harmbench_distraction_image_map_seed_0_num_10000.json`

See [Creating the distraction image map](#creating-the-distraction-image-map) below to download the image library and build this map.

## Creating the distraction image map

VERA-V reuses the visual-enhanced distraction pipeline from [CS-DJ](https://github.com/TeamPigeonLab/CS-DJ#preparation). Follow these steps once before running attack scripts that need distractors (`vlm_typo.py`, `vlm_distractoronly.py`, `vlm_7d_2typo.py`, `vlm_7d_2diff.py`, etc.).

### 1. Download the image library

As in CS-DJ preparation, download the [LLaVA-CC3M-Pretrain-595K](https://huggingface.co/datasets/liuhaotian/LLaVA-CC3M-Pretrain-595K) dataset and store the images under:

```text
CS_DJ_main/llava_images/
```

Example layout after download:

```text
CS_DJ_main/llava_images/
├── GCC_train_000000001.jpg
├── GCC_train_000000002.jpg
└── ...
```



### 2. Required models

CS-DJ uses the following models for map construction (already covered by the project environment / Hugging Face downloads):

- Embedding model: [clip-ViT-L-14](https://huggingface.co/sentence-transformers/clip-ViT-L-14)
- Auxiliary model (CS-DJ structured distraction): [Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)



### 3. Build CLIP image embeddings

From `CS_DJ_main/`, sample images and encode them with CLIP:

```bash
cd CS_DJ_main
python script/image_embeding.py \
  --src_dir ./llava_images \
  --save_embeding_path ./image_embedding \
  --seed 0 \
  --num_images 10000
```

This writes:

```text
CS_DJ_main/image_embedding/map_seed_0_num_10000.json
```



### 4. Build the distraction image map

Map each harmful behavior to a set of contrasting (maximally dissimilar) subimages:

```bash
python script/Visual-Enhanced_Distraction.py \
  --seed 0 \
  --num_images 10000 \
  --save_map_path ./image_map
```

This writes:

```text
CS_DJ_main/image_map/distraction_image_map_seed_0_num_10000.json
```

For HarmBench experiments in this repo, the expected filename is:

```text
CS_DJ_main/image_map/harmbench_distraction_image_map_seed_0_num_10000.json
```

If your script produced `distraction_image_map_seed_0_num_10000.json`, rename or copy it to the HarmBench name above, or point `select_img_path` in the attack script to your file.

### 5. Point attack scripts at the map (and image root)

In each attack script that uses distractors, set `select_img_path` to your map, for example in `scripts/vlm_typo.py`:

```python
select_img_path = "CS_DJ_main/image_map/harmbench_distraction_image_map_seed_0_num_10000.json"
```

Distraction image paths inside the JSON are typically relative (`./llava_images/...`). Ensure `utils/image_utils.py` (and any script-local path rewriting) resolves those paths to your local `CS_DJ_main/llava_images/` directory.

## Running `scripts/vlm_typo.py`



### Basic usage

```bash
conda activate veravlm
python scripts/vlm_typo.py
```

This loads `configs/vlm_dual.yaml` and starts the attack loop.

## Configuration (`configs/vlm_dual.yaml`)

The top-level YAML composes smaller config groups via the `defaults` list. Each group maps to a file under `configs/<group>/`.


| Group       | Default                 | Description                                |
| ----------- | ----------------------- | ------------------------------------------ |
| `judge`     | `hb_judge`              | HarmBench classifier for scoring responses |
| `ext_judge` | `combined_judge`        | External safety judge (MD + Llama Guard 2) |
| `attacker`  | `dual_vlm`              | Dual VLM attacker (Vicuna + LoRA)          |
| `loss`      | `vlm_loss`              | ELBO-style VLM loss                        |
| `optimizer` | `adamw`                 | AdamW optimizer                            |
| `target`    | `qwen2.5`               | Target VLM                                 |
| `data`      | `harmbench`             | HarmBench behavior dataset                 |
| `diffusion` | `stable-diffusion-v1-5` | Stable Diffusion version                   |




### Top-level parameters


| Parameter                              | Default             | Meaning                                               |
| -------------------------------------- | ------------------- | ----------------------------------------------------- |
| `batch_size`                           | `8`                 | Adversarial prompts per optimization step             |
| `max_queries`                          | `40`                | Max query budget per harmful prompt                   |
| `threshold`                            | `0.90`              | Judge score above which a jailbreak counts as success |
| `attacker_device` / `attacker_device2` |  | Devices for the attackers                         |
| `judge_device`                         |             | Device for the HarmBench judge                        |
| `diffusion_device`                     |             | Device for Stable Diffusion                           |


## Project layout

```
VERA-V/
├── scripts/                 # Experiment entry points
│   ├── vlm_typo.py          # Main entry (dual VLM + typo + diffusion)
│   ├── vlm_typoonly.py
│   ├── vlm_diffusiononly.py
│   ├── vlm_distractoronly.py
│   ├── vlm_1diff_1typo.py
│   ├── vlm_7d_2typo.py
│   ├── vlm_7d_2diff.py
│   └── vlm_textonly.py
├── configs/                 # Hydra configs
│   ├── vlm_dual.yaml        # Default top-level config
│   ├── attacker/            # Attacker model configs
│   ├── target/              # Target VLM configs
│   ├── judge/               # Judge configs
│   ├── data/                # Dataset configs
│   ├── diffusion/           # Diffusion model configs
│   ├── loss/                # Loss function configs
│   └── optimizer/           # Optimizer configs
├── attacker/                # Attacker implementations
├── target/                  # Target VLM wrappers
├── judge/                   # Judge implementations
├── diffusion/               # Diffusion backends
├── training/                # Loss functions
├── utils/                   # Data loading, image utils, run dirs
├── data/                    # Datasets
└── CS_DJ_main/              # CS-DJ distraction image pipeline & maps
    ├── llava_images/        # LLaVA-CC3M images (download required)
    ├── image_embedding/     # CLIP embeddings (generated)
    ├── image_map/           # Distraction image maps (generated)
    └── script/              # image_embeding.py, Visual-Enhanced_Distraction.py, ...
```

## Evaluation

To evaluate the output, use `harmfulness_harmbench.py` to test the attack success rate (ASR).

```
python harmfulness_harmbench.py --input_file xxx
```

## Citation

If you like our work, please cite:

```bibtex
@inproceedings{
liao2026verav,
title={{VERA}-V: Variational Inference Framework for Jailbreaking Vision-Language Models},
author={Qilin Liao and Anamika Lochab and Ruqi Zhang},
year={2026},
booktitle={ICML},
url={https://openreview.net/forum?id=tj9EwfaxHm}
}
```

