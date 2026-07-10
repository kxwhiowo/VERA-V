from omegaconf import OmegaConf
import os


def config_matches(current_cfg, config_path):
    """Compare the current configuration with the one at `config_path`."""
    existing_cfg = OmegaConf.load(config_path)
    return current_cfg == existing_cfg


def find_or_create_run_dir(cfg):
    # Build the checkpoint base directory

    base_dir = f"{cfg.save_dir}"
    os.makedirs(base_dir, exist_ok=True)

    # Iterate over existing run directories
    run_number = 0
    resume_dir = None
    for dir_name in sorted(os.listdir(base_dir)):
        if dir_name.startswith("run_"):
            run_path = os.path.join(base_dir, dir_name)
            config_path = os.path.join(run_path, "config.yaml")
            if os.path.isfile(config_path) and config_matches(cfg, config_path):
                print(f"Matching configuration found in {run_path}. Resuming...")
                resume_dir = run_path
                break
            run_number += 1

    # If no matching config found, create a new run directory
    if resume_dir is None:
        run_dir = os.path.join(base_dir, f"run_{run_number}")
        os.makedirs(run_dir, exist_ok=True)
        OmegaConf.save(cfg, os.path.join(run_dir, "config.yaml"))
        print(f"No matching configuration found. Created new directory: {run_dir}")
        return run_dir, False
    else:
        return resume_dir, True
