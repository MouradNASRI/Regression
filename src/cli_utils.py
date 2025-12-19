"""
CLI utilities for the project:
 - build_parser(): constructs the full argparse CLI interface
 - parse_kv(): parses key=value CLI arguments into a dict

These are separated from main.py so that:
  * main.py stays clean and contains only orchestration logic
  * CLI parsing logic is reusable (e.g. for notebooks, scripts, future CLIs)
"""

import argparse
import ast
import os


def parse_kv(pairs):
    """
    Parse key=value strings into a dict, with literal_eval casting.
    Example: ["alpha=0.1", "hidden_layer_sizes='(100,50)'"]
      -> {"alpha": 0.1, "hidden_layer_sizes": (100, 50)}
    """
    cfg = {}
    for item in pairs:
        k, v = item.split("=", 1)
        try:
            v = ast.literal_eval(v)
        except Exception:
            pass
        cfg[k] = v
    return cfg


def build_parser():
    """
    Construct and return the main CLI argument parser.

    This function defines ALL command-line options supported by the training script:
      * Dataset selection (e.g. --dataset bike)
      * Input data path
      * Test split ratio
      * Model selection
      * Model hyperparameters (via key=value)
      * MLflow profile + optional experiment override
      * Utility flags such as --list_models

    Keeping this in a dedicated utility module avoids cluttering main.py
    and makes the CLI definition reusable (e.g., for automation scripts,
    interactive notebooks, or future training entrypoints).
    """
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, required=True, help="path to input data")
    p.add_argument("--dataset", type=str, required=True, help="Dataset key (e.g. 'bike', 'housing')")
    p.add_argument("--mlflow_profile", type=str, default="local", help="MLflow profile name (config in configs/mlflow/<name>.yml)")

    p.add_argument("--test_train_ratio", type=float, default=0.25)

    # Model selection & discovery
    p.add_argument("--model", default="linear", help="model key; use --list_models to see options")
    p.add_argument("--list_models", action="store_true", help="print available models and exit")

    # Model hyperparameters as key=value
    p.add_argument("--model_params", nargs="*", default=[],
                   help='e.g. alpha=0.3 n_estimators=200 hidden_layer_sizes="(100,50)"')

    # MLflow / experiment controls
    p.add_argument("--mlflow_uri", type=str, default=os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    p.add_argument("--experiment", type=str, default=os.getenv("MLFLOW_EXPERIMENT_NAME"))
    p.add_argument("--autolog", action="store_true", help="use mlflow.sklearn.autolog (metrics/params only)")
    p.add_argument("--registered_model_name", type=str, default=None, help="optional registry name (not used here unless you add model registry calls)")
    p.add_argument("--hparam_search", type=str, default=None, help="Name of a hyperparameter search config under configs/hparam_search/")
    p.add_argument("--target", type=str, default=None, help="Override target column name (otherwise use dataset config default).",)
    p.add_argument("--task", type=str, choices=["regression", "classification"], required=True, help="Type of ML task (e.g. regression, classification).",)


    return p


def list_available_models(task):
    """
    Print all registered models grouped by category and exit.

    This is a helper for the --list_models CLI flag.

    It queries the central MODEL_REGISTRY (via list_models(by_group=True)),
    formats the output nicely, prints it to stdout, and terminates the process.

    Keeping this logic here avoids cluttering main.py with utility behavior.
    """
    from src.model_registry import list_models
    import sys

    print("Available models by group:")
    groups = list_models(task=task, by_group=True)

    for group_name, registry in groups.items():
        print(f"  [{group_name}] {', '.join(sorted(registry.keys()))}")

    sys.exit(0)
