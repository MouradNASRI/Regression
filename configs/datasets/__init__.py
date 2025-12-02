from pathlib import Path
import yaml

CONFIG_DIR = Path(__file__).resolve().parent

def load_dataset_config(name: str) -> dict:
    path = CONFIG_DIR / f"{name}.yml"
    if not path.exists():
        raise ValueError(f"Unknown dataset '{name}'. Expected config at {path}")
    with open(path, "r") as f:
        return yaml.safe_load(f)
