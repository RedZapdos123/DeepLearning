import os
os.environ["RAY_NODE_STARTUP_TIMEOUT"] = "120"
os.environ["TUNE_DISABLE_STRICT_METRIC_CHECKING"] = "1"

import yaml
import tempfile
import shutil
from sklearn.model_selection import KFold
import torch
from ultralytics import YOLO
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch
import pandas as pd
import matplotlib.pyplot as plt
import ray

#The file path of the data.yaml of the dataset.
DATA_YAML = r"C:\Users\Xeron\OneDrive\Documents\LargeDatasets\VehicleDatasetYOLO\data.yaml"

#The YOLO model path.
YOLO_MODEL_PATH = "yolo11n.pt"
if not os.path.exists(YOLO_MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {YOLO_MODEL_PATH}")

#The function to extract the mAP50 values from the metrics folder.
def extract_mAP50(metrics_dict):
    if "metrics/mAP50(B)" in metrics_dict:
        return metrics_dict["metrics/mAP50(B)"]
    for key, value in metrics_dict.items():
        if "mAP50" in key:
            return value
    return 0.0

#Creating temporray folders with yaml files for 5-folds cross validation.
def create_fold_yaml_with_labels(fold, total_folds, original_yaml_path):
    with open(original_yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    train_img_dir = data["train"]

    #Assuming labels are stored in a sibling folder 'labels'.
    train_lbl_dir = os.path.join(os.path.dirname(train_img_dir), "labels")
    image_extensions = [".jpg", ".jpeg", ".png", ".bmp"]
    img_files = [os.path.join(train_img_dir, fname)
                 for fname in sorted(os.listdir(train_img_dir))
                 if os.path.splitext(fname)[1].lower() in image_extensions]
    kf = KFold(n_splits=total_folds, shuffle=True, random_state=42)
    splits = list(kf.split(img_files))
    train_idx, val_idx = splits[fold]
    train_imgs = [img_files[i] for i in train_idx]
    val_imgs = [img_files[i] for i in val_idx]
    temp_dir = tempfile.mkdtemp()
    temp_train_img = os.path.join(temp_dir, "train", "images")
    temp_train_lbl = os.path.join(temp_dir, "train", "labels")
    temp_val_img = os.path.join(temp_dir, "val", "images")
    temp_val_lbl = os.path.join(temp_dir, "val", "labels")
    os.makedirs(temp_train_img, exist_ok=True)
    os.makedirs(temp_train_lbl, exist_ok=True)
    os.makedirs(temp_val_img, exist_ok=True)
    os.makedirs(temp_val_lbl, exist_ok=True)
    def copy_files(file_list, dest_img_dir, dest_lbl_dir):
        for img_path in file_list:
            shutil.copy(img_path, dest_img_dir)
            base = os.path.splitext(os.path.basename(img_path))[0]
            lbl_path = os.path.join(train_lbl_dir, base + ".txt")
            if os.path.exists(lbl_path):
                shutil.copy(lbl_path, dest_lbl_dir)
    copy_files(train_imgs, temp_train_img, temp_train_lbl)
    copy_files(val_imgs, temp_val_img, temp_val_lbl)
    new_data = data.copy()
    new_data["train"] = temp_train_img
    new_data["val"] = temp_val_img
    tmp_yaml = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
    yaml.dump(new_data, tmp_yaml)
    tmp_yaml.close()
    return tmp_yaml.name, temp_dir

#The function for the 5-folds cross validation, and hyperparameter optimization with OptunaSearch alogorithm, and ASHA scheduler.
def train_yolo_trial_cv(config, checkpoint_dir=None):
    lr0 = config["lr0"]
    batch_size = config["batch_size"]
    epochs = config["epochs"]
    imgsz = config.get("imgsz", 640)
    
    fold_metrics = []

    #Collect the weight folder path from one of the folds (e.g., the first fold)
    weight_folder = None

    #Numbers of folders for k-folds.
    total_folds = 5

    for fold in range(total_folds):
        tmp_yaml, temp_dir = create_fold_yaml_with_labels(fold, total_folds, DATA_YAML)
        try:
            model = YOLO(YOLO_MODEL_PATH)
            #Forcing YOLO to save outputs in a given folder by setting 'project' and 'name':
            results = model.train(
                data=tmp_yaml,
                epochs=epochs,
                batch=batch_size,
                imgsz=imgsz,
                optimizer="AdamW", #Using Adapative Moment Estimation with Weight decay Optimizer
                lr0=lr0,
                weight_decay=config.get("weight_decay", 0.0005),
                verbose=False,
                project=r"C:\Users\Xeron\OneDrive\Documents\Programs\tune", 
                name="yolo_exp"
            )
            #Capture the weights folder from the first fold for checkpointing.
            if fold == 0:
                #YOLOv11 saves weights in its output folder. This path can be adjusted based on requirements.
                weight_folder = os.path.join(os.getcwd(), "runs", "detect", "train", "exp", "weights")
            mAP50 = 0.0 #Default mAP50
            if isinstance(results, list) and results:
                res0 = results[0]
                if hasattr(res0, "metrics") and isinstance(res0.metrics, dict):
                    mAP50 = extract_mAP50(res0.metrics)
            elif hasattr(results, "metrics"):
                mAP50 = extract_mAP50(results.metrics)
            fold_metrics.append(mAP50)
        except Exception as e:
            print(f"Error during training fold {fold}: {e}")
            raise
        finally:
            os.remove(tmp_yaml)
            shutil.rmtree(temp_dir)
    avg_mAP50 = sum(fold_metrics) / len(fold_metrics)
    
    #Save a checkpoint: Copy the YOLO weights to a temporary checkpoint directory.
    ckpt_dir = tempfile.mkdtemp()
    if weight_folder and os.path.exists(weight_folder):
        #Copy weight files if they exist in weight folder.
        for fname in os.listdir(weight_folder):
            if fname in ["best.pt", "last.pt"]:
                shutil.copy(os.path.join(weight_folder, fname), os.path.join(ckpt_dir, fname))
                print(f"Checkpoint saved: {fname} copied to {ckpt_dir}")
    else:
        print("Warning: Weight folder not found; checkpoint will be empty.")
    
    #Report the metric and the checkpoint directory to Ray Tune.
    tune.report(**{"metrics/mAP50(B)": avg_mAP50}, checkpoint=tune.Checkpoint.from_directory(ckpt_dir))

#The function to save the results, reports and metrics for human evaluation.
def save_results_files(analysis):
    # Define our overall output directory and weights subfolder:
    output_dir = r"C:\Users\Xeron\OneDrive\Documents\Programs\tune"
    weights_dir = os.path.join(output_dir, "weights")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(weights_dir, exist_ok=True)
    
    #Save the best hyperparameters and results' CSV files, and plots in output directory.
    best_config = analysis.get_best_config(metric="metrics/mAP50(B)", mode="max")
    with open(os.path.join(output_dir, "best_hyperparameters.yaml"), "w") as f:
        yaml.dump(best_config, f)
    df_results = analysis.results_df
    df_results.to_csv(os.path.join(output_dir, "tune_results.csv"), index=False)
    df_sorted = df_results.sort_values("metrics/mAP50(B)", ascending=False)
    plt.figure()
    plt.plot(df_sorted["metrics/mAP50(B)"].values, marker="o")
    plt.xlabel("Trial (sorted by mAP50)")
    plt.ylabel("mAP50")
    plt.title("Best Fitness across Trials")
    plt.savefig(os.path.join(output_dir, "best_fitness.png"))
    plt.close()
    hyper_keys = [col for col in df_results.columns if col.startswith("config.")]
    if hyper_keys:
        num_plots = len(hyper_keys)
        plt.figure(figsize=(8, 4 * num_plots))
        for i, key in enumerate(hyper_keys, 1):
            plt.subplot(num_plots, 1, i)
            plt.scatter(df_results[key], df_results["metrics/mAP50(B)"], alpha=0.7)
            plt.xlabel(key.replace("config.", ""))
            plt.ylabel("mAP50")
            plt.title(f"{key.replace('config.', '')} vs mAP50")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "tune_scatter_plots.png"))
        plt.close()
    
    #Debug: Print trial the directory structure.
    best_trial = analysis.get_best_trial(metric="metrics/mAP50(B)", mode="max")
    trial_dir = best_trial.local_path
    print("Debug: Best trial directory structure:")
    for root, dirs, files in os.walk(trial_dir):
        indent = " " * (root.count(os.sep) - trial_dir.count(os.sep))
        print(f"{indent}{os.path.basename(root)}/")
        if dirs:
            print(f"{indent}  Directories: {dirs}")
        if files:
            print(f"{indent}  Files: {files}")
    
    #Debug: Access and inspect checkpoint via Ray Tune API.
    if best_trial.checkpoint:
        with best_trial.checkpoint.as_directory() as cp_dir:
            print("\nDebug: Checkpoint directory from best trial:", cp_dir)
            cp_files = os.listdir(cp_dir)
            print("Debug: Files in checkpoint directory:", cp_files)
    else:
        print("\nDebug: No checkpoint found for best trial.")
    
    #Copy weight files (best.pt and last.pt) if found in trial directory to the required weights folder.
    for weight_file in ["best.pt", "last.pt"]:
        for root, dirs, files in os.walk(trial_dir):
            if weight_file in files:
                src = os.path.join(root, weight_file)
                dst = os.path.join(weights_dir, weight_file)
                shutil.copy(src, dst)
                print(f"Copied {weight_file} from {src} to {dst}")
                break

def trial_dirname_creator(trial):
    return f"trial_{trial.trial_id}"

#The driver main function.
def main():
    ray.init(ignore_reinit_error=True)

    #The configurations for the optimizer.
    config = {
        "lr0": tune.loguniform(1e-5, 1e-2), #The learning rate range for optimizations.
        "batch_size": tune.choice([2, 4, 8, 16]), #The batch sizes to be tested for optimizations.
        "epochs": 50, #The number of epoches for training, each trial.
        "momentum": tune.uniform(0.8, 0.99), #The momentum range for optimization.
        "weight_decay": tune.loguniform(1e-5, 1e-3),
        "imgsz": 640,
    }

    #The ASHA scheduler for early stopping of trials if improvements in the algorithm, don't occur.
    scheduler = ASHAScheduler(
        metric="metrics/mAP50(B)",
        mode="max",
        max_t=config["epochs"],
        grace_period=1,
        reduction_factor=2
    )

    #Using OptunaSearch for searching the best hyperparameters.
    search_alg = OptunaSearch(metric="metrics/mAP50(B)", mode="max")
    storage_path = "file:///" + os.path.abspath("./ray_results").replace("\\", "/")

    #Running the Ray Tune sprogram.
    analysis = tune.run(
         train_yolo_trial_cv,
         resources_per_trial={"cpu": 4, "gpu": 1 if torch.cuda.is_available() else 0},
         config=config,
         num_samples=10,
         scheduler=scheduler,
         search_alg=search_alg,
         storage_path=storage_path,
         name="yolo11_cv_hpo",
         trial_dirname_creator=trial_dirname_creator
    )

    #Printing the best hyperparameters found.
    print("Best hyperparameters found were:", analysis.get_best_config(metric="metrics/mAP50(B)", mode="max"))
    save_results_files(analysis)

if __name__ == "__main__":
    main()
