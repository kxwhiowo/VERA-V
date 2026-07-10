import os
import json
import pandas as pd
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util

from config import CS_DJ_parser
from pathlib import Path as _Path
_CS_DJ_DIR = _Path(__file__).resolve().parents[1]
_REPO_ROOT = _CS_DJ_DIR.parent


def main(args):
    # Force use of CUDA:1 and set environment variables
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # Now cuda:0 refers to physical GPU 1
    print(f"Using device: {device} (physical GPU 1)")
    
    # Check GPU memory before starting
    if torch.cuda.is_available():
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB total")
        print(f"GPU memory allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
        print(f"GPU memory reserved: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")

    # Load HarmBench behaviors (list of strings)
    harmbench_path = str(_REPO_ROOT / 'harmbench_behaviors_first_200.json')
    with open(harmbench_path, 'r', encoding='utf-8') as f:
        jailbreak_question_list = json.load(f)
    
    # Verify it's a list of strings
    if not isinstance(jailbreak_question_list, list) or not all(isinstance(item, str) for item in jailbreak_question_list):
        raise ValueError("Expected harmbench_behaviors_first_200.json to contain a list of strings")
    
    print(f"First behavior: {jailbreak_question_list[0]}")
    print(f"Loaded {len(jailbreak_question_list)} jailbreak behaviors from harmbench_behaviors_first_200.json")

    # Load image embeddings
    embedding_path = str(_CS_DJ_DIR / 'image_embedding' / 'map_seed_0_num_10000.json')
    with open(embedding_path, 'r', encoding='utf-8') as f:
        embedding_data = json.load(f)

    image_embeddings = [item['img_emb'] for item in embedding_data]
    image_paths = [item['img_path'] for item in embedding_data]

    print(f"Loaded {len(image_embeddings)} image embeddings")
    
    # Convert to tensors with explicit device placement
    print("Loading image embeddings to GPU...")
    image_embeddings = torch.tensor(image_embeddings, device=device)
    
    # Initialize model with explicit device
    print("Loading CLIP model...")
    model = SentenceTransformer('clip-ViT-L-14', device=device)
    
    # Clear cache after model loading
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"After model loading - GPU memory allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")

    results = {}
    for behavior in tqdm(jailbreak_question_list, desc="Processing behaviors"):
        max_distance_embedding_list = []
        selected_image_list = []

        # Encode the behavior text
        text_emb = model.encode(behavior, convert_to_tensor=True).to(device)
        max_distance_embedding_list.append(text_emb)

        # Select 15 most dissimilar images
        for _ in range(15):
            combined_emb = torch.vstack(max_distance_embedding_list)
            cos_scores = util.cos_sim(combined_emb, image_embeddings).mean(dim=0) 

            min_score, min_index = torch.min(cos_scores, dim=0)
            selected_image_list.append(image_paths[int(min_index)])
            max_distance_embedding_list.append(image_embeddings[min_index])

        results[behavior] = selected_image_list

    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cs_dj_dir = os.path.dirname(script_dir)  # Go up one level to CS_DJ_main
    
    # Resolve save path relative to CS_DJ_main directory
    if args.save_map_path.startswith('./'):
        save_map_path = os.path.join(cs_dj_dir, args.save_map_path[2:])
    else:
        save_map_path = args.save_map_path
    
    if not os.path.exists(save_map_path):
        os.makedirs(save_map_path)
    
    output_file = os.path.join(save_map_path, f'distraction_image_map_seed_{args.seed}_num_{args.num_images}.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    print(f"Results saved to {output_file}")

if __name__ == "__main__":

    parser = CS_DJ_parser()
    args = parser.parse_args()

    main(args)
