# src/model_utils.py

import os
import yaml
from typing import Dict, Any

from src.model_registry import get_model_group


def load_model_config(task:str, model_name: str) -> Dict[str, Any]:
    """
    Load a model config YAML from configs/models/<group>/<model_name>.yaml.

    Example:
        model_name = "lasso" (in group "linear")
        -> configs/models/linear/lasso.yaml

    Falls back to old flat layout configs/models/<model_name>.yaml
    if the grouped file is not found (for backward compatibility).

    Parameters
    ----------
    task : the task required to be done, regression or classification ...
    model_name : str
        Model identifier passed via CLI (e.g. --model ols).

    Returns
    -------
    cfg : Dict[str, Any]
        Parsed YAML configuration.
    """
    # Adjust base_dir if your project structure differs.
    base_dir = os.path.dirname(os.path.dirname(__file__))  # project root
    group = get_model_group(task ,model_name)
    model_cfg_path = None
    
    # 1) Try grouped layout: configs/models/<group>/<model>.yaml
    if group is not None:
        grouped_path = os.path.join(
            base_dir, "configs", "models", task, group, f"{model_name}.yml"
        )
        if os.path.exists(grouped_path):
            model_cfg_path = grouped_path
    print("********************************************************")
    print("********************************************************")
    print("group " + group)
    print("base_dir " + base_dir)
    print("model_cfg_path " + model_cfg_path)
    print("********************************************************")
    print("********************************************************")
    if model_cfg_path is None:
        raise FileNotFoundError(
            f"Model config for '{model_name}' not found."
        )

    with open(model_cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if cfg is None:
        raise ValueError(f"Model config {model_cfg_path} is empty or invalid YAML")

    return cfg