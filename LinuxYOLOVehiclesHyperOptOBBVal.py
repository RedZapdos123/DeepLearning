import os
import yaml
import tempfile
import shutil
import torch
import matplotlib.pyplot as plt
from ultralytics import YOLO
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch

#―――――――――― Setup dirs and env vars ――――――――――
HOME_DIR     = os.path.expanduser("~")
TMP_BASE     = os.path.join(HOME_DIR, "big_disk_tmp")
RAY_TMP      = os.path.join(HOME_DIR, "ray_tmp")
TUNE_RESULTS = os.path.join(HOME_DIR, "tune_results")
RAY_RESULTS  = os.path.join(HOME_DIR, "ray_results")
RESULT_BASE  = os.path.join(HOME_DIR, "results")
for path in (TMP_BASE, RAY_TMP, TUNE_RESULTS, RAY_RESULTS, RESULT_BASE):
    os.makedirs(path, exist_ok=True)
os.environ["TUNE_DISABLE_STRICT_METRIC_CHECKING"] = "1"
os.environ["RAY_NODE_STARTUP_TIMEOUT"]    = "120"
ray.init(_temp_dir=RAY_TMP, ignore_reinit_error=True)

#―――――――――― Config ―――――――――――――――――――――
DATA_YAML      = "/home/test/datasets/StratifiedDroneVehiclesOBB/data.yaml"
YOLO_MODEL     = "yolo11n-obb.pt"
if not os.path.exists(YOLO_MODEL):
    raise FileNotFoundError(f"Model file not found: {YOLO_MODEL}")
PRIMARY_METRIC = "metrics/mAP50(B)"
PATIENCE       = 15
MIN_DELTA      = 0.001
GRACE_PERIOD   = 15

def validate_obb_dataset(path):
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)
    train = cfg.get('train')
    if train:
        labels = os.path.join(os.path.dirname(train), 'labels')
        if os.path.isdir(labels):
            for fn in os.listdir(labels)[:5]:
                if fn.endswith('.txt'):
                    parts = open(os.path.join(labels, fn)).readline().split()
                    if len(parts) not in (0, 9):
                        print(f"{fn} has {len(parts)} values (expected 9).")
                        return False
    return True

def extract_map50(md):
    if PRIMARY_METRIC in md:
        return md[PRIMARY_METRIC]
    for k, v in md.items():
        if "mAP50" in k or "map50" in k.lower():
            return v
    return 0.0

class EarlyStopper:
    def __init__(self, patience=PATIENCE, min_delta=MIN_DELTA):
        self.patience  = patience
        self.min_delta = min_delta
        self.best      = None
        self.count     = 0
        self.stop      = False

    def check(self, score, epoch):
        if self.best is None or score > self.best + self.min_delta:
            self.best  = score
            self.count = 0
            return False
        self.count += 1
        if self.count >= self.patience:
            self.stop = True
            return True
        return False

def train_yolo_trial(config, checkpoint_dir=None):
    try:
        model   = YOLO(YOLO_MODEL)
        stopper = EarlyStopper()

        res = model.train(
            data        = DATA_YAML,
            epochs      = int(config["epochs"]),
            batch       = int(config["batch_size"]),
            imgsz       = int(config["imgsz"]),
            lr0         = config["lr0"],
            lrf         = config["lrf"],
            weight_decay= config["weight_decay"],
            optimizer   = "AdamW",
            task        = "obb",
            project     = os.path.join(RESULT_BASE, "results", "tune"),
            name        = "yolo_obb_exp",
            degrees     = config["degrees"],
            perspective = config["perspective"],
            fliplr      = config["fliplr"],
            flipud      = config["flipud"],
            mosaic      = config["mosaic"],
            mixup       = config["mixup"],
            hsv_h       = config["hsv_h"],
            hsv_s       = config["hsv_s"],
            hsv_v       = config["hsv_v"],
            scale       = config["scale"],
            shear       = config["shear"],
            verbose     = False,
            exist_ok    = True,
            patience    = PATIENCE
        )

        val = model.val(data=DATA_YAML, verbose=False)
        if isinstance(val, list) and val and hasattr(val[0], "metrics"):
            m = extract_map50(val[0].metrics.results_dict)
        elif hasattr(val, "metrics"):
            m = extract_map50(val.metrics.results_dict)
        else:
            m = 0.0

        if stopper.check(m, res.epoch):
            tune.report(metrics={PRIMARY_METRIC: m})
            return

        wf   = os.path.join(os.getcwd(), "runs", "detect", "train", "yolo_obb_exp", "weights")
        ckpt = tempfile.mkdtemp(dir=TMP_BASE)
        if os.path.isdir(wf):
            for name in ("best.pt", "last.pt"):
                src = os.path.join(wf, name)
                if os.path.exists(src):
                    shutil.copy(src, os.path.join(ckpt, name))

        tune.report(metrics={PRIMARY_METRIC: m}, checkpoint=tune.Checkpoint.from_directory(ckpt))

    except Exception as e:
        print("Error:", e)
        tune.report(metrics={PRIMARY_METRIC: 0.0})

def save_results(analysis):
    od = os.path.join(RESULT_BASE, "results", "tune")
    wd = os.path.join(od, "weights")
    os.makedirs(wd, exist_ok=True)

    best_cfg = analysis.get_best_config(metric=PRIMARY_METRIC, mode="max")
    with open(os.path.join(od, "best_hyperparameters.yaml"), "w") as f:
        yaml.dump(best_cfg, f)

    df = analysis.results_df
    df.to_csv(os.path.join(od, "tune_results.csv"), index=False)

    bests = df.sort_values(PRIMARY_METRIC, ascending=False)[PRIMARY_METRIC]
    plt.figure(figsize=(10,6))
    plt.plot(bests.values, marker="o")
    plt.xlabel("Trial")
    plt.ylabel(PRIMARY_METRIC)
    plt.title("Top OBB mAP50 across trials")
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(od, "best_fitness.png"))
    plt.close()

def main():
    print("Validating dataset...")
    if not validate_obb_dataset(DATA_YAML):
        print("Validation failed. Exiting.")
        return
    print("Dataset OK. Launching HPO.")

    config = {
        "lr0": tune.loguniform(1e-5, 1e-3),           # adjusted to include starting point
        "lrf": tune.loguniform(0.01, 0.2),
        "batch_size": tune.choice([16, 32]),
        "epochs": 200,
        "weight_decay": tune.loguniform(1e-6, 5e-4),
        "imgsz": tune.choice([640, 800, 960]),
        "degrees": tune.uniform(0.0, 30.0),
        "perspective": tune.uniform(0.0, 0.001),
        "fliplr": tune.choice([0.0, 0.5]),
        "flipud": tune.choice([0.0, 0.3]),
        "mosaic": tune.choice([0.0, 0.5, 1.0]),
        "mixup": tune.uniform(0.0, 0.3),
        "hsv_h": tune.uniform(0.0, 0.05),
        "hsv_s": tune.uniform(0.0, 0.7),
        "hsv_v": tune.uniform(0.0, 0.5),
        "scale": tune.uniform(0.3, 0.8),
        "shear": tune.uniform(0.0, 0.3),
    }

    starting_points = [
        {"lr0": 1e-3, "lrf": 0.1,  "batch_size": 32, "weight_decay": 5e-4, "imgsz": 640,
         "degrees": 10.0, "perspective": 0.0005, "fliplr": 0.5, "flipud": 0.0,
         "mosaic": 1.0, "mixup": 0.2, "hsv_h": 0.015, "hsv_s": 0.7, "hsv_v": 0.4,
         "scale": 0.5, "shear": 0.0},
        {"lr0": 5e-4, "lrf": 0.05, "batch_size": 16, "weight_decay": 1e-4, "imgsz": 800,
         "degrees": 0.0, "perspective": 0.0, "fliplr": 0.5, "flipud": 0.0,
         "mosaic": 0.5, "mixup": 0.1, "hsv_h": 0.0, "hsv_s": 0.0, "hsv_v": 0.0,
         "scale": 0.5, "shear": 0.0}
    ]

    scheduler = ASHAScheduler(
        metric            = PRIMARY_METRIC,
        mode              = "max",
        max_t             = 200,
        grace_period      = GRACE_PERIOD,
        reduction_factor  = 3
    )

    search_alg = OptunaSearch(
        metric             = PRIMARY_METRIC,
        mode               = "max",
        points_to_evaluate = starting_points
    )

    analysis = tune.run(
        train_yolo_trial,
        resources_per_trial    = {"cpu": 12, "gpu": 1 if torch.cuda.is_available() else 0},
        config                 = config,
        num_samples            = 200,
        scheduler              = scheduler,
        search_alg             = search_alg,
        checkpoint_score_attr  = PRIMARY_METRIC,
        storage_path           = RAY_RESULTS,
        name                   = "yolo11_obb_hpo",
        trial_dirname_creator  = lambda t: f"trial_{t.trial_id}"
    )

    print("Best config:", analysis.get_best_config(metric=PRIMARY_METRIC, mode="max"))
    print("Best mAP50:", analysis.get_best_trial(metric=PRIMARY_METRIC, mode="max").last_result[PRIMARY_METRIC])

    save_results(analysis)

if __name__ == "__main__":
    main()
