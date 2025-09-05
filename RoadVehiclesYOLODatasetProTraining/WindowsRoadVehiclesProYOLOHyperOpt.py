import os
import tempfile
import shutil
import torch
from ultralytics import YOLO
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch

os.environ["RAY_NODE_STARTUP_TIMEOUT"] = "120"
os.environ["TUNE_DISABLE_STRICT_METRIC_CHECKING"] = "1"

HOME_DIR = os.path.expanduser("~")
TMP_BASE = os.path.join(HOME_DIR, "big_disk_tmp")
RAY_TMP = os.path.join(HOME_DIR, "ray_tmp")
RAY_RESULTS = os.path.join(HOME_DIR, "ray_results")
RESULT_BASE = os.path.join(HOME_DIR, "results")

for path in (TMP_BASE, RAY_TMP, RAY_RESULTS, RESULT_BASE):
    os.makedirs(path, exist_ok=True)

ray.init(_temp_dir=RAY_TMP, ignore_reinit_error=True)

DATA_PATH = r"C:\Users\Xeron\OneDrive\Desktop\StratifiedRoadVehiclesDatasetProYolo\data.yaml"
MODEL_PATH = "yolo11n.pt"
RESULTS_DIR = r"C:\Users\Xeron\RoadVehiclesDatasetHyperOpt"
NUM_SAMPLES = 60
MAX_T = 25
GRACE_PERIOD = 5

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

def extract_mAP50(metrics_dict):
    if "metrics/mAP50(B)" in metrics_dict:
        return metrics_dict["metrics/mAP50(B)"]
    for k, v in metrics_dict.items():
        if "mAP50" in k:
            return v
    return 0.0

def train_yolo(config, checkpoint_dir=None):
    model = YOLO(MODEL_PATH)
    
    if checkpoint_dir:
        for candidate in ("best.pt", "last.pt"):
            src = os.path.join(checkpoint_dir, candidate)
            if os.path.exists(src):
                try:
                    model = YOLO(src)
                    break
                except Exception:
                    pass
    
    project_dir = os.path.join(RESULT_BASE, "results", "tune")
    name = "yolo_exp"
    epochs = int(config.get("epochs", 25))
    
    results = model.train(
        data=DATA_PATH,
        epochs=epochs,
        batch=int(config["batch_size"]),
        imgsz=int(config["imgsz"]),
        lr0=float(config["lr0"]),
        lrf=float(config["lrf"]),
        optimizer=config["optimizer"],
        momentum=float(config.get("momentum", 0.9)),
        weight_decay=float(config.get("weight_decay", 0.0)),
        warmup_epochs=float(config.get("warmup_epochs", 0.0)),
        hsv_h=float(config.get("hsv_h", 0.0)),
        hsv_s=float(config.get("hsv_s", 0.0)),
        hsv_v=float(config.get("hsv_v", 0.0)),
        degrees=float(config.get("degrees", 0.0)),
        translate=float(config.get("translate", 0.0)),
        scale=float(config.get("scale", 0.5)),
        shear=float(config.get("shear", 0.0)),
        perspective=float(config.get("perspective", 0.0)),
        flipud=float(config.get("flipud", 0.0)),
        fliplr=float(config.get("fliplr", 0.0)),
        mosaic=float(config.get("mosaic", 0.0)),
        mixup=float(config.get("mixup", 0.0)),
        copy_paste=float(config.get("copy_paste", 0.0)),
        box=float(config.get("box", 0.05)),
        cls=float(config.get("cls", 0.5)),
        dfl=float(config.get("dfl", 0.5)),
        verbose=False,
        project=project_dir,
        name=name,
        exist_ok=True,
        save=True
    )
    
    m_val = 0.0
    if isinstance(results, list) and results and hasattr(results[0], "metrics"):
        m_val = extract_mAP50(results[0].metrics)
    elif hasattr(results, "metrics"):
        m_val = extract_mAP50(results.metrics)
    
    weight_folder = os.path.join(
        os.getcwd(), "runs", "detect", "train", "exp", "weights"
    )
    
    ckpt_dir = tempfile.mkdtemp(dir=TMP_BASE)
    if weight_folder and os.path.isdir(weight_folder):
        for fname in ("best.pt", "last.pt"):
            src = os.path.join(weight_folder, fname)
            if os.path.exists(src):
                shutil.copy(src, os.path.join(ckpt_dir, fname))
    
    tune.report(
        **{"metrics/mAP50(B)": m_val},
        checkpoint=tune.Checkpoint.from_directory(ckpt_dir)
    )

search_space = {
    "optimizer": tune.choice(["AdamW"]),
    "lr0": tune.loguniform(1e-6, 1e-3),
    "lrf": tune.loguniform(0.02, 0.15),
    "batch_size": tune.choice([8, 16]),
    "imgsz": tune.choice([512, 640]),
    "epochs": MAX_T,
    "weight_decay": tune.loguniform(1e-6, 2e-4),
    "momentum": 0.9,
    "warmup_epochs": tune.uniform(0.0, 2.0),
    "mosaic": tune.choice([0.0, 0.5]),
    "mixup": tune.uniform(0.0, 0.15),
    "hsv_h": tune.uniform(0.0, 0.02),
    "hsv_s": tune.uniform(0.0, 0.4),
    "hsv_v": tune.uniform(0.0, 0.4),
    "fliplr": tune.choice([0.0, 0.5]),
    "flipud": tune.choice([0.0, 0.15]),
    "copy_paste": tune.choice([0.0, 0.5]),
    "box": tune.uniform(0.02, 0.08),
    "cls": tune.uniform(0.5, 1.2),
    "dfl": tune.uniform(0.2, 0.8)
}

scheduler = ASHAScheduler(
    metric="metrics/mAP50(B)",
    mode="max",
    max_t=MAX_T,
    grace_period=GRACE_PERIOD,
    reduction_factor=2
)

search_alg = OptunaSearch(
    metric="metrics/mAP50(B)",
    mode="max"
)

analysis = tune.run(
    train_yolo,
    resources_per_trial={"cpu": 8, "gpu": 1 if torch.cuda.is_available() else 0},
    config=search_space,
    num_samples=NUM_SAMPLES,
    scheduler=scheduler,
    search_alg=search_alg,
    storage_path=RAY_RESULTS,
    name="yolo11_road_vehicles_hpo",
    trial_dirname_creator=lambda t: f"trial_{t.trial_id}"
)

best_config = analysis.get_best_config(metric="metrics/mAP50(B)", mode="max")
best_trial = analysis.get_best_trial(metric="metrics/mAP50(B)", mode="max")

print("Best Config:", best_config)
print("Best mAP50:", best_trial.last_result.get("metrics/mAP50(B)"))

ray.shutdown()