# This program is linux compatible version of the YOLO-RayTune hyperparameter optimization program using the yolo11 pretrained model.
# It uses Ray Tune's internal OptunaSearch algorithm along with ASHA scheduler, and yolo11 in-built AdamW loss function for hyperparameter optimizations.
# The metric for evaluation is the mAP50 metric, commonly used for evaluation of YOLO based models.
# Although this program does not make use of the K-folds method for train-val split, for lesser runtime.
# The dataset used for testing this program is: https://www.kaggle.com/datasets/redzapdos123/aerial-view-of-vehicles-and-humans-dataset-yolo

import os

# Increased Ray startup timeout and relax Tune metric checking.
os.environ["RAY_NODE_STARTUP_TIMEOUT"] = "120"
os.environ["TUNE_DISABLE_STRICT_METRIC_CHECKING"] = "1"

# Use home directory instead of /mnt/big_disk.
HOME_DIR     = os.path.expanduser("~")
TMP_BASE     = os.path.join(HOME_DIR, "big_disk_tmp")
RAY_TMP      = os.path.join(HOME_DIR, "ray_tmp")
TUNE_RESULTS = os.path.join(HOME_DIR, "tune_results")
RAY_RESULTS  = os.path.join(HOME_DIR, "ray_results")

# Add the missing RESULT_BASE variable.
RESULT_BASE = os.path.join(HOME_DIR, "results")

for path in (TMP_BASE, RAY_TMP, TUNE_RESULTS, RAY_RESULTS, RESULT_BASE):
    os.makedirs(path, exist_ok=True)

import yaml
import tempfile
import shutil
import torch
from ultralytics import YOLO
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch
import matplotlib.pyplot as plt

# Initialize Ray with a custom temp directory.
ray.init(_temp_dir=RAY_TMP, ignore_reinit_error=True)

# The folder and file path configurations.
DATA_YAML  = "/home/test/datasets/StratifiedRoadVehiclesDatasetProYolo/data.yaml"
YOLO_MODEL = "yolo11n.pt"
if not os.path.exists(YOLO_MODEL):
    raise FileNotFoundError(f"Model file not found: {YOLO_MODEL}")

# Function to extract the mAP50 values.
def extract_mAP50(metrics_dict):
    # Ultralytics stores mAP50 under different keys depending on version. The YOLO version used is YOLOv11.
    if "metrics/mAP50(B)" in metrics_dict:
        return metrics_dict["metrics/mAP50(B)"]
    for k, v in metrics_dict.items():
        if "mAP50" in k:
            return v
    return 0.0

# Function for running a training trial for YOLO.
def train_yolo_trial(config, checkpoint_dir=None):
    lr0        = config["lr0"]       # learning rate
    batch_size = config["batch_size"]
    epochs     = config["epochs"]
    imgsz      = config.get("imgsz", 640)

    model = YOLO(YOLO_MODEL)
    results = model.train(
        data=DATA_YAML,
        epochs=epochs,
        batch=batch_size,
        imgsz=imgsz,
        optimizer="AdamW",
        lr0=lr0,
        weight_decay=config.get("weight_decay", 0.0005),
        verbose=False,
        project=os.path.join(RESULT_BASE, "results", "tune"),
        name="yolo_exp"
    )

    # Extract mAP50
    m_val = 0.0
    if isinstance(results, list) and results and hasattr(results[0], "metrics"):
        m_val = extract_mAP50(results[0].metrics)
    elif hasattr(results, "metrics"):
        m_val = extract_mAP50(results.metrics)

    # Save weights to checkpoint
    ckpt_dir = tempfile.mkdtemp(dir=TMP_BASE)
    weight_folder = os.path.join(os.getcwd(), "runs", "detect", "train", "exp", "weights")
    if os.path.isdir(weight_folder):
        for fname in ("best.pt", "last.pt"):
            src = os.path.join(weight_folder, fname)
            if os.path.exists(src):
                shutil.copy(src, os.path.join(ckpt_dir, fname))

    tune.report(
        **{"metrics/mAP50(B)": m_val},
        checkpoint=tune.Checkpoint.from_directory(ckpt_dir)
    )

# Function to save the results files.
def save_results_files(analysis):
    out_dir     = os.path.join(RESULT_BASE, "results", "tune")
    weights_dir = os.path.join(out_dir, "weights")
    os.makedirs(weights_dir, exist_ok=True)

    # Save best hyperparameters
    best_cfg = analysis.get_best_config(metric="metrics/mAP50(B)", mode="max")
    with open(os.path.join(out_dir, "best_hyperparameters.yaml"), "w") as f:
        yaml.dump(best_cfg, f)

    # Save results CSV
    df = analysis.results_df
    df.to_csv(os.path.join(out_dir, "tune_results.csv"), index=False)

    # Plot best fitness curve
    bests = df.sort_values("metrics/mAP50(B)", ascending=False)["metrics/mAP50(B)"]
    plt.figure()
    plt.plot(bests.values, marker="o")
    plt.xlabel("Trial (sorted)")
    plt.ylabel("mAP50")
    plt.title("Best Fitness Across Trials")
    plt.savefig(os.path.join(out_dir, "best_fitness.png"))
    plt.close()

    # Plot hyperparameter vs mAP50 scatter
    hyper_keys = [c for c in df.columns if c.startswith("config.")]
    if hyper_keys:
        plt.figure(figsize=(8, 4 * len(hyper_keys)))
        for i, key in enumerate(hyper_keys, start=1):
            plt.subplot(len(hyper_keys), 1, i)
            plt.scatter(df[key], df["metrics/mAP50(B)"], alpha=0.7)
            plt.xlabel(key.replace("config.", ""))
            plt.ylabel("mAP50")
            plt.title(f"{key.replace('config.', '')} vs mAP50")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "tune_scatter_plots.png"))
        plt.close()

    # Copy best.pt and last.pt to final folder
    best_trial = analysis.get_best_trial(metric="metrics/mAP50(B)", mode="max")
    td = best_trial.local_path
    for wf in ("best.pt", "last.pt"):
        for root, _, files in os.walk(td):
            if wf in files:
                shutil.copy(os.path.join(root, wf), os.path.join(weights_dir, wf))
                break

def main():
    # Optimized starting points to seed Optuna.
    starting_points = [
        {"lr0": 1e-3, "batch_size": 16, "weight_decay": 5e-4},
        {"lr0": 3e-4, "batch_size": 32, "weight_decay": 1e-4},
    ]

    # Ray Tune configuration.
    config = {
        "lr0": tune.loguniform(1e-5, 1e-3),
        "batch_size": tune.choice([16, 32]),
        "epochs": 150,
        "weight_decay": tune.loguniform(1e-5, 1e-3),
        "imgsz": 640,
    }

    # ASHA scheduler.
    scheduler = ASHAScheduler(
        metric="metrics/mAP50(B)",
        mode="max",
        max_t=150,
        grace_period=15,
        reduction_factor=4
    )

    # Optuna search algorithm.
    search_alg = OptunaSearch(
        metric="metrics/mAP50(B)",
        mode="max",
        points_to_evaluate=starting_points
    )

    # Run hyperparameter tuning.
    analysis = tune.run(
        train_yolo_trial,
        resources_per_trial={"cpu": 12, "gpu": 1 if torch.cuda.is_available() else 0},
        config=config,
        num_samples=70,
        scheduler=scheduler,
        search_alg=search_alg,
        storage_path=RAY_RESULTS,
        name="yolo11_standard_hpo",
        trial_dirname_creator=lambda t: f"trial_{t.trial_id}"
    )

    # Print and save results.
    print("Best hyperparameters:", analysis.get_best_config(metric="metrics/mAP50(B)", mode="max"))
    save_results_files(analysis)

if __name__ == "__main__":
    main()
