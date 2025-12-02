# src/hparam_search.py

from __future__ import annotations

import os
from pathlib import Path
import yaml
from typing import Any, Dict, Iterable, Optional
from itertools import product
from .training_utils import train_eval_log

from src.model_registry import get_model_group

BASE_DIR = os.path.dirname(os.path.dirname(__file__))


def load_hparam_search_config(
    name: str,
    model_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Load a hyperparameter search config.

    Supports grouped layout:
        configs/hparam_search/<group>/<name>.yaml
    and flat layout:
        configs/hparam_search/<name>.yaml

    If model_name is provided, we use its group as default folder.
    """
    search_cfg_path = None

    # 1) If model_name is provided, we can infer group
    if model_name is not None:
        group = get_model_group(model_name)
        if group is not None:
            grouped_path = os.path.join(
                BASE_DIR, "configs", "hparam_search", group, f"{name}.yml"
            )
            if os.path.exists(grouped_path):
                search_cfg_path = grouped_path

    # 2) Fallback: flat layout
    if search_cfg_path is None:
        flat_path = os.path.join(
            BASE_DIR, "configs", "hparam_search", f"{name}.yml"
        )
        if os.path.exists(flat_path):
            search_cfg_path = flat_path

    if search_cfg_path is None:
        raise FileNotFoundError(
            f"Hparam search config '{name}' not found (model={model_name})."
        )

    with open(search_cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if cfg is None:
        raise ValueError(f"Hparam search config {search_cfg_path} is empty or invalid YAML")

    return cfg
    
def _generate_grid(search_space: Dict[str, Iterable[Any]]):
    """
    Generate all combinations of a simple grid search space.

    Parameters
    ----------
    search_space : dict[str, Iterable[Any]]
        Example:
            {"alpha": [0.1, 1.0], "normalize": [True, False]}

    Yields
    ------
    dict[str, Any]
        One dictionary per combination, e.g.:
            {"alpha": 0.1, "normalize": True}
            {"alpha": 0.1, "normalize": False}
            ...
    """
    if not search_space:
        # No search space → single "empty" combination
        yield {}
        return

    keys = list(search_space.keys())
    values_list = [list(search_space[k]) for k in keys]

    for combo in product(*values_list):
        yield dict(zip(keys, combo))


def run_hparam_search(
    search_cfg: Dict[str, Any],
    X_train,
    y_train,
    X_test,
    y_test,
    pre=None,
    cli_model_params: Dict[str, Any] | None = None,
    cli_model_name: str | None = None,
    mlflow_experiment: str | None = None,
) -> Dict[str, Any]:
    """
    Run a hyperparameter grid search on top of `train_eval_log`.

    Parameters
    ----------
    search_cfg : dict
        Parsed YAML config for the search. Expected keys:
            - "name": str, human-readable search name (optional, default "hparam_search")
            - "model": str, model name used by train_eval_log (unless overridden by CLI)
            - "metric": str, metric key returned by train_eval_log()["metrics"]
            - "mode": "min" or "max" (default: "min")
            - "strategy": currently only "grid" is supported
            - "base_model_params": dict of fixed/default params (optional)
            - "params": dict[str, list[Any]] of values to explore (optional)
    X_train, y_train, X_test, y_test :
        Training and test data passed directly to `train_eval_log`.
    pre :
        Whatever preprocessor object/flag your `train_eval_log` expects. Can be None.
    cli_model_params : dict, optional
        Parsed `--model_params` from CLI. These are treated as FIXED overrides:
        - they override base_model_params,
        - they are *not* tuned by the search.
    cli_model_name : str, optional
        Model name from CLI (`--model`). If provided, it overrides `search_cfg["model"]`.
    mlflow_experiment : str, optional
        Experiment name to pass to `train_eval_log` (or None to use its default).

    Returns
    -------
    dict
        Description of the best trial:

            {
                "search_name": str,
                "metric_name": str,
                "mode": "min" | "max",
                "best_params": dict[str, Any],
                "best_metrics": dict[str, float],
                "best_metric_value": float,
                "num_trials": int,
            }
    """

    # ----------------------------------------------------------------------
    # 1. Resolve basic config
    # ----------------------------------------------------------------------
    search_name = search_cfg.get("name", "hparam_search")
    metric_name = search_cfg["metric"]
    mode = search_cfg.get("mode", "min").lower()
    strategy = search_cfg.get("strategy", "grid").lower()

    if mode not in {"min", "max"}:
        raise ValueError(f"Invalid mode '{mode}'. Expected 'min' or 'max'.")

    # Decide which model name to use
    cfg_model_name = search_cfg.get("model")
    if cli_model_name is not None:
        model_name = cli_model_name
    else:
        if cfg_model_name is None:
            raise ValueError(
                "No model specified in search config and no CLI model name given."
            )
        model_name = cfg_model_name

    # ----------------------------------------------------------------------
    # 2. Build base model params (config + CLI overrides)
    # ----------------------------------------------------------------------
    base_model_params: Dict[str, Any] = dict(search_cfg.get("base_model_params", {}))

    # CLI overrides config defaults and is considered FIXED
    if cli_model_params:
        base_model_params.update(cli_model_params)

    # ----------------------------------------------------------------------
    # 3. Build search space and ensure we don't override CLI-fixed params
    # ----------------------------------------------------------------------
    raw_space: Dict[str, Iterable[Any]] = dict(search_cfg.get("params", {}))

    if cli_model_params:
        # Remove any keys that the user fixed via CLI:
        # we don't want the search to change those.
        search_space = {
            name: values
            for name, values in raw_space.items()
            if name not in cli_model_params
        }
    else:
        search_space = raw_space

    if strategy != "grid":
        raise NotImplementedError(
            f"Search strategy '{strategy}' is not implemented. Only 'grid' is supported."
        )

    # ----------------------------------------------------------------------
    # 4. Iterate over all combinations and call train_eval_log
    # ----------------------------------------------------------------------
    best_trial: Dict[str, Any] | None = None
    num_trials = 0

    for trial_idx, hp_update in enumerate(_generate_grid(search_space)):
        num_trials += 1

        # Merged params for this trial:
        # - base_model_params: config + CLI (fixed)
        # - hp_update: only params that are allowed to be tuned
        trial_params = {**base_model_params, **hp_update}

        trial_result = train_eval_log(
            model_name=model_name,
            model_params=trial_params,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            pre=pre,
            mlflow_experiment=mlflow_experiment,
        )

        # We assume train_eval_log returns {"metrics": {...}, ...}
        trial_metrics = trial_result["metrics"]
        if metric_name not in trial_metrics:
            raise KeyError(
                f"Metric '{metric_name}' not found in trial metrics. "
                f"Available: {list(trial_metrics.keys())}"
            )

        metric_value = trial_metrics[metric_name]

        # Check if this trial is better than the current best
        if best_trial is None:
            is_better = True
        elif mode == "min":
            is_better = metric_value < best_trial["metric_value"]
        else:  # mode == "max"
            is_better = metric_value > best_trial["metric_value"]

        if is_better:
            best_trial = {
                "metric_value": metric_value,
                "params": trial_params,
                "metrics": trial_metrics,
            }

    if best_trial is None:
        raise RuntimeError("Hyperparameter search produced no trials.")

    # ----------------------------------------------------------------------
    # 5. Return a compact summary
    # ----------------------------------------------------------------------
    return {
        "search_name": search_name,
        "metric_name": metric_name,
        "mode": mode,
        "best_params": best_trial["params"],
        "best_metrics": best_trial["metrics"],
        "best_metric_value": best_trial["metric_value"],
        "num_trials": num_trials,
    }
