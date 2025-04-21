#This program is linux compatible version of the YOLO-RayTune hyperparameter optimization program using the yolo11 pretrained model.
#It uses Ray Tune's internal OptunaSearch algorithm along with ASHA scheduler, and yolo11 in-built AdamW loss function for hyperparamter optimzations.
#The dataset used for testing this program is: https://www.kaggle.com/datasets/redzapdos123/modified-aerial-traffic-and-visdrone-dataset-yolo

import os
#Increased Ray startup timeout and relax Tune metric checking
os.environ["RAY_NODE_STARTUP_TIMEOUT"] = "120"
os.environ["TUNE_DISABLE_STRICT_METRIC_CHECKING"] = "1"

#Use home directory.
HOME_DIR = os.path.expanduser("~")
TMP_BASE    = os.path.join(HOME_DIR, "big_disk_tmp")
RAY_TMP     = os.path.join(HOME_DIR, "ray_tmp")
TUNE_RESULTS = os.path.join(HOME_DIR, "tune_results")
RAY_RESULTS  = os.path.join(HOME_DIR, "ray_results")
#Add the missing RESULT_BASE variable.
RESULT_BASE  = os.path.join(HOME_DIR, "results")

for path in (TMP_BASE, RAY_TMP, TUNE_RESULTS, RAY_RESULTS, RESULT_BASE):
    os.makedirs(path, exist_ok=True)

import yaml
import tempfile
import shutil
from sklearn.model_selection import KFold
import torch
from ultralytics import YOLO
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch
import matplotlib.pyplot as plt

#Initialize Ray with a custom temp directory.
ray.init(_temp_dir=RAY_TMP, ignore_reinit_error=True)

#The folder and file path configurations.
DATA_YAML = "/home/test/datasets/VisDroneVehiclesYOLOPlus/data.yaml"
YOLO_MODEL = "yolo11n.pt"
#Checking if the YOLO11 model exists.
if not os.path.exists(YOLO_MODEL):
    raise FileNotFoundError(f"Model file not found: {YOLO_MODEL}")

#The mAP50 metrics extractor.
def extract_mAP50(metrics_dict):
    #Ultralytics stores mAP50 under different keys depending on version. THe YOLO version in use is YOLOv11.
    if "metrics/mAP50(B)" in metrics_dict:
        return metrics_dict["metrics/mAP50(B)"]
    for k, v in metrics_dict.items():
        if "mAP50" in k:
            return v
    return 0.0

def create_fold_yaml_with_labels(fold, total_folds, original_yaml_path):
    #Load the yaml config file.
    with open(original_yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    #Resolve train images directory
    data_dir  = os.path.dirname(original_yaml_path)
    prefix    = data.get("path", "").rstrip('/')
    train_rel = data["train"]
    cand1     = os.path.join(data_dir, prefix, train_rel)
    cand2     = os.path.join(data_dir, train_rel)
    if   os.path.isdir(cand1): train_img_dir = cand1
    elif os.path.isdir(cand2): train_img_dir = cand2
    else:
        raise FileNotFoundError(f"Train image dir not found:\n  • {cand1}\n  • {cand2}")

    train_lbl_dir = os.path.join(os.path.dirname(train_img_dir), "labels")

    #Gather the image files
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    img_files = [
        os.path.join(train_img_dir, fn)
        for fn in sorted(os.listdir(train_img_dir))
        if os.path.splitext(fn)[1].lower() in exts
    ]

    #Performing the K-Fold split.
    kf = KFold(n_splits=total_folds, shuffle=True, random_state=42)
    train_idx, val_idx = list(kf.split(img_files))[fold]
    train_imgs = [img_files[i] for i in train_idx]
    val_imgs   = [img_files[i] for i in val_idx]

    #Create fold-specific temp dirs under TMP_BASE.
    tmp_root = tempfile.mkdtemp(dir=TMP_BASE)
    dirs = {
        "t_img": os.path.join(tmp_root, "train", "images"),
        "t_lbl": os.path.join(tmp_root, "train", "labels"),
        "v_img": os.path.join(tmp_root, "val",   "images"),
        "v_lbl": os.path.join(tmp_root, "val",   "labels"),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    #Link images & labels instead of copying for reducing memory the memory usuage.
    def link_set(file_list, dst_img, dst_lbl):
        for img in file_list:
            basename = os.path.basename(img)
            dst_path = os.path.join(dst_img, basename)
            try:
                os.link(img, dst_path)
            except OSError:
                os.symlink(img, dst_path)

            label_src = os.path.join(train_lbl_dir,
                                     os.path.splitext(basename)[0] + ".txt")
            if os.path.exists(label_src):
                dst_lbl_path = os.path.join(dst_lbl, os.path.basename(label_src))
                try:
                    os.link(label_src, dst_lbl_path)
                except OSError:
                    os.symlink(label_src, dst_lbl_path)

    link_set(train_imgs, dirs["t_img"], dirs["t_lbl"])
    link_set(val_imgs,   dirs["v_img"], dirs["v_lbl"])

    #Dump a small YAML pointing to these folders.
    new_data = data.copy()
    new_data["train"] = dirs["t_img"]
    new_data["val"]   = dirs["v_img"]
    tf = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, dir=TMP_BASE
    )
    yaml.dump(new_data, tf)
    tf.close()

    return tf.name, tmp_root

def train_yolo_trial_cv(config, checkpoint_dir=None):
    lr0        = config["lr0"]
    batch_size = config["batch_size"]
    epochs     = config["epochs"]
    imgsz      = config.get("imgsz", 640)

    fold_metrics = []
    weight_folder = None

    #Use 3 folds out of 4, for training.
    for fold in range(3):
        tmp_yaml, tmp_dir = create_fold_yaml_with_labels(fold, 4, DATA_YAML)
        try:
            model = YOLO(YOLO_MODEL)
            results = model.train(
                data=tmp_yaml,
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

            #Record weights directory once.
            if fold == 0:
                weight_folder = os.path.join(
                    os.getcwd(), "runs", "detect", "train", "exp", "weights"
                )

            #Extract the mAP50 metric.
            m_val = 0.0
            if isinstance(results, list) and results and hasattr(results[0], "metrics"):
                m_val = extract_mAP50(results[0].metrics)
            elif hasattr(results, "metrics"):
                m_val = extract_mAP50(results.metrics)
            fold_metrics.append(m_val)

        finally:
            os.remove(tmp_yaml)
            shutil.rmtree(tmp_dir, ignore_errors=True)

    avg_mAP50 = sum(fold_metrics) / len(fold_metrics)

    #Save the best.pt and last.pt from weight_folder into a checkpoint.
    ckpt_dir = tempfile.mkdtemp(dir=TMP_BASE)
    if weight_folder and os.path.isdir(weight_folder):
        for fname in ("best.pt", "last.pt"):
            src = os.path.join(weight_folder, fname)
            if os.path.exists(src):
                shutil.copy(src, os.path.join(ckpt_dir, fname))

    tune.report(
        **{"metrics/mAP50(B)": avg_mAP50},
        checkpoint=tune.Checkpoint.from_directory(ckpt_dir)
    )

#The function to save the results.
def save_results_files(analysis):
    out_dir     = os.path.join(RESULT_BASE, "results", "tune")
    weights_dir = os.path.join(out_dir, "weights")
    os.makedirs(weights_dir, exist_ok=True)

    #Save the Best hyperparameters file.
    best_cfg = analysis.get_best_config(metric="metrics/mAP50(B)", mode="max")
    with open(os.path.join(out_dir, "best_hyperparameters.yaml"), "w") as f:
        yaml.dump(best_cfg, f)

    #Save the results CSV file.
    df = analysis.results_df
    df.to_csv(os.path.join(out_dir, "tune_results.csv"), index=False)

    #Plot the best fitness curve.
    bests = df.sort_values("metrics/mAP50(B)", ascending=False)["metrics/mAP50(B)"]
    plt.figure()
    plt.plot(bests.values, marker="o")
    plt.xlabel("Trial (sorted)")
    plt.ylabel("mAP50")
    plt.title("Best Fitness Across Trials")
    plt.savefig(os.path.join(out_dir, "best_fitness.png"))
    plt.close()

    #Plot the hyperparameter vs mAP50 scatter.
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

    #Copy the best.pt & last.pt to final weights folder.
    best_trial = analysis.get_best_trial(metric="metrics/mAP50(B)", mode="max")
    td = best_trial.local_path
    for wf in ("best.pt", "last.pt"):
        for root, _, files in os.walk(td):
            if wf in files:
                shutil.copy(os.path.join(root, wf), os.path.join(weights_dir, wf))
                break

def main():
    #The starting points to seed Optuna.
    starting_points = [
        {"lr0": 1e-3, "batch_size": 16, "weight_decay": 5e-4},
        {"lr0": 5e-4, "batch_size": 64, "weight_decay": 1e-4},
    ]

    #The OptunaSearch configurations.
    config = {
        "lr0": tune.loguniform(5e-5, 5e-3), #The learning rate range to be tested.
        "batch_size": tune.choice([16, 32, 64]),
        "epochs": 20, #The number of epoches.
        "weight_decay": tune.loguniform(5e-5, 5e-4),
        "imgsz": 640, #The images' size.
    }

    #The configurations of the ASHA scheduler of Ray Tune. Used this scheduler for agressiveness.
    scheduler = ASHAScheduler(
        metric="metrics/mAP50(B)", #the metric to be evaluated for optimizations.
        mode="max",
        max_t=20, #The max number of epoches.
        #The agressiveness of the ASHA scheduler. Set to agressive.
        grace_period=3, #The minimum number of epoches to be tried before the improvemnet check occurs.
        reduction_factor=5
    )

    #The Optuna search algorithm's configurations.
    search_alg = OptunaSearch(
        metric="metrics/mAP50(B)",
        mode="max",
        points_to_evaluate=starting_points
    )

    #Updated to use only storage_path and remove local_dir.
    analysis = tune.run(
        train_yolo_trial_cv,
        resources_per_trial={"cpu": 12, "gpu": 1 if torch.cuda.is_available() else 0},
        config=config,
        num_samples=20,
        scheduler=scheduler,
        search_alg=search_alg,
        storage_path=RAY_RESULTS,  #Use this for both storage and local dir.
        name="yolo11_cv_hpo",
        trial_dirname_creator=lambda t: f"trial_{t.trial_id}"
    )

    #Print the best found hyperparameters.
    print("Best hyperparameters:", analysis.get_best_config(
        metric="metrics/mAP50(B)", mode="max"
    ))
    save_results_files(analysis)


if __name__ == "__main__":
    main()
