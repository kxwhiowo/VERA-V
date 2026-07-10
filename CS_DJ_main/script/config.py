import argparse
import os
from pathlib import Path

_CS_DJ_DIR = Path(__file__).resolve().parents[1]
_REPO_ROOT = _CS_DJ_DIR.parent


def CS_DJ_parser():
    
    parser = argparse.ArgumentParser(description="CS-DJ Configuration")
    parser.add_argument("--src_dir", type=str, default=str(_CS_DJ_DIR / "llava_images"), help="Image directory")
    parser.add_argument("--save_embeding_path", type=str, default=str(_CS_DJ_DIR / "image_embedding"), help="Path to save the embedding.")
    parser.add_argument("--save_map_path", type=str, default=str(_CS_DJ_DIR / "image_map"), help="Path to save the jailbreak image map.")
    parser.add_argument("--jailbreak_folder_path", type=str, default=str(_REPO_ROOT / "harmbench_behaviors_first_200.json"), help="Jailbreak raw query set.")
    parser.add_argument("--jailbreak_response_save_path", type=str, default=str(_CS_DJ_DIR / "results"), help="Path to save the jailbreak response.")
    parser.add_argument("--distraction_image_save_path", type=str, default=str(_CS_DJ_DIR / "distraction_images"), help="Path to save the jailbreak image.")
    parser.add_argument("--strategy_name", type=str, default='CS-DJ_best_method', help="Strategy name.")
    
    # main set up
    parser.add_argument("--seed", type=int, default=0, help="Random seed for selecting images.")
    parser.add_argument("--num_images", type=int, default=10000, help="Number of images to select.")
    parser.add_argument("--object_model", type=str, default='qwen2.5-vl-7b', choices=['gpt-4o-mini', 'gpt-4o', 'gpt-4-vision-preview', 'qwen2.5-vl-7b', 'qwen2.5-vl-14b', 'qwen2.5-vl-32b', 'qwen2.5-vl-72b', 'internvl3-8b', 'internvl3-26b', 'internvl2-8b', 'internvl2-26b'], help="Jailbreak object model.")
    parser.add_argument("--api_key", type=str, default=os.getenv("OPENAI_API_KEY", ""), help="OpenAI API key (or set OPENAI_API_KEY).")
    
    # VLM model specific arguments
    parser.add_argument("--use_vlm", action='store_true', help="Use VLM model instead of OpenAI API.")
    parser.add_argument("--vlm_device", type=str, default='auto', help="Device for VLM model.")
    parser.add_argument("--load_in_4bit", action='store_true', help="Load VLM model in 4-bit precision.")
    parser.add_argument("--load_in_8bit", action='store_true', help="Load VLM model in 8-bit precision.")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Maximum new tokens for VLM generation.")
    parser.add_argument("--temperature", type=float, default=0, help="Temperature for VLM generation.")
      
    return parser
