import os
import yaml
import json
import shutil
import glob
import sys
import torch
from ultralytics import YOLO
import matplotlib.pyplot as plt

class EarlyStopper:
    def __init__(self, patience=20, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = None
        self.counter = 0
        self.early_stop = False
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            return False
        if score > self.best_score:
            self.best_score = score
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
            return False

class YOLOTrainer:
    def __init__(self, data_yaml, model_path, output_dir):
        self.data_yaml = data_yaml
        self.model_path = model_path
        self.output_dir = output_dir
        self.weights_dir = os.path.join(output_dir, "weights")
        self.checkpoints_dir = os.path.join(output_dir, "checkpoints")
        self.logs_dir = os.path.join(output_dir, "logs")
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.weights_dir, exist_ok=True)
        os.makedirs(self.checkpoints_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        self.hyperparameters = {
            "optimizer": "AdamW",
            "lr0": 0.001100396444041285,
            "lrf": 0.022432538532785873,
            "batch_size": 32,
            "imgsz": 640,
            "epochs": 100,
            "weight_decay": 1.324691158201738e-05,
            "momentum": 0.9,
            "warmup_epochs": 0.06321711079049863,
            "mosaic": 0.0,
            "mixup": 0.07215981744702592,
            "hsv_h": 0.010711244641757033,
            "hsv_s": 0.029598115263020825,
            "hsv_v": 0.36835823959566194,
            "fliplr": 0.0,
            "flipud": 0.0,
            "copy_paste": 0.5,
            "box": 0.05502276540112764,
            "cls": 0.5697157260734642,
            "dfl": 0.7999202740031451
        }
        self.training_history = []
        self.best_map50 = 0.0

    def save_hyperparameters(self):
        with open(os.path.join(self.output_dir, "hyperparameters.yaml"), "w") as f:
            yaml.dump(self.hyperparameters, f, default_flow_style=False)

    def save_checkpoint(self, epoch, best_map50, current_map50, run_dir=None):
        checkpoint = {
            "epoch": epoch,
            "hyperparameters": self.hyperparameters,
            "best_map50": best_map50,
            "current_map50": current_map50,
            "training_history": self.training_history,
            "run_dir": run_dir
        }
        checkpoint_path = os.path.join(self.checkpoints_dir, f"checkpoint_epoch_{epoch}.json")
        with open(checkpoint_path, "w") as f:
            json.dump(checkpoint, f, indent=2, default=str)
        latest_checkpoint_path = os.path.join(self.checkpoints_dir, "latest_checkpoint.json")
        with open(latest_checkpoint_path, "w") as f:
            json.dump(checkpoint, f, indent=2, default=str)
        print(f"Checkpoint saved for epoch {epoch} (mAP50: {current_map50:.4f})")

    def load_checkpoint(self):
        latest_checkpoint_path = os.path.join(self.checkpoints_dir, "latest_checkpoint.json")
        if os.path.exists(latest_checkpoint_path):
            try:
                with open(latest_checkpoint_path, "r") as f:
                    checkpoint = json.load(f)
                self.training_history = checkpoint.get("training_history", [])
                self.best_map50 = checkpoint.get("best_map50", 0.0)
                return checkpoint
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Error loading checkpoint: {e}")
                return None
        return None

    def find_latest_exp_dir(self, project_dir, name="train"):
        base = os.path.join(project_dir, name)
        if not os.path.exists(base):
            return None
        subdirs = [os.path.join(base, d) for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))]
        if not subdirs:
            return base
        subdirs.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        return subdirs[0]

    def parse_yolo_results(self, run_dir):
        if not run_dir:
            return
        candidates = []
        for ext in ("csv","json"):
            candidates += glob.glob(os.path.join(run_dir, "**", f"results*.{ext}"), recursive=True)
            candidates += glob.glob(os.path.join(run_dir, "**", f"*result*.{ext}"), recursive=True)
        if not candidates:
            return
        path = next((p for p in candidates if p.lower().endswith(".csv")), candidates[0])
        try:
            if path.lower().endswith(".csv"):
                import pandas as pd, re
                df = pd.read_csv(path)
                df.columns = df.columns.str.strip()
                if "epoch" in df.columns:
                    epochs = df["epoch"].astype(int).tolist()
                else:
                    epochs = list(range(1, len(df) + 1))
                map_cols = [c for c in df.columns if re.search(r"(?i)m\s*ap.*50", c) or ("map" in c.lower() and "50" in c)]
                if not map_cols:
                    map_cols = [c for c in df.columns if "map" in c.lower()]
                if not map_cols:
                    return
                map_col = map_cols[0]
                for ep, val in zip(epochs, df[map_col].fillna(0.0)):
                    current_map50 = float(val)
                    is_best = current_map50 > self.best_map50
                    if is_best:
                        self.best_map50 = current_map50
                    self.training_history.append({"epoch": int(ep), "map50": current_map50, "is_best": is_best})
                    if int(ep) % 10 == 0 or is_best:
                        self.save_checkpoint(int(ep), self.best_map50, current_map50, run_dir=run_dir)
            else:
                import json as _json
                with open(path, "r") as f:
                    data = _json.load(f)
                found = None
                if isinstance(data, dict):
                    for k, v in data.items():
                        if isinstance(v, (int, float)) and "map" in k.lower() and "50" in k:
                            found = float(v)
                            break
                if found is None:
                    return
                current_map50 = float(found)
                is_best = current_map50 > self.best_map50
                if is_best:
                    self.best_map50 = current_map50
                self.training_history.append({"epoch": 0, "map50": current_map50, "is_best": is_best})
                self.save_checkpoint(0, self.best_map50, current_map50, run_dir=run_dir)
        except ImportError:
            print("Warning: pandas not available for parsing training results")
        except Exception as e:
            print(f"Error parsing YOLO results: {e}")

    def save_training_plot(self):
        if not self.training_history:
            return
        epochs = [entry["epoch"] for entry in self.training_history]
        map50_values = [entry["map50"] for entry in self.training_history]
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, map50_values, marker="o", linewidth=2, markersize=4)
        plt.xlabel("Epoch")
        plt.ylabel("mAP50")
        plt.title("Training Progress - mAP50 over Epochs")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.logs_dir, "training_progress.png"), dpi=300)
        plt.close()

    def copy_best_weights(self, run_dir):
        if not run_dir:
            return
        best_candidates = glob.glob(os.path.join(run_dir, "**", "best*.pt"), recursive=True)
        last_candidates = glob.glob(os.path.join(run_dir, "**", "last*.pt"), recursive=True)
        if best_candidates:
            shutil.copy2(best_candidates[0], os.path.join(self.weights_dir, "best.pt"))
        if last_candidates:
            shutil.copy2(last_candidates[0], os.path.join(self.weights_dir, "last.pt"))

    def train(self, resume=False):
        self.save_hyperparameters()
        resume_path = None
        if resume:
            checkpoint = self.load_checkpoint()
            last_run = self.find_latest_exp_dir(self.output_dir, "train")
            if checkpoint and last_run:
                last_weights = glob.glob(os.path.join(last_run, "**", "last*.pt"), recursive=True)
                if last_weights:
                    resume_path = last_weights[0]
                    print(f"Resuming training from {resume_path}")
                    print(f"Best mAP50 so far: {self.best_map50:.4f}")
        model = YOLO(resume_path if resume_path else self.model_path)
        try:
            results = model.train(
                data=self.data_yaml,
                epochs=self.hyperparameters["epochs"],
                batch=self.hyperparameters["batch_size"],
                imgsz=self.hyperparameters["imgsz"],
                lr0=self.hyperparameters["lr0"],
                lrf=self.hyperparameters["lrf"],
                weight_decay=self.hyperparameters["weight_decay"],
                optimizer=self.hyperparameters["optimizer"],
                momentum=self.hyperparameters["momentum"],
                warmup_epochs=self.hyperparameters["warmup_epochs"],
                task="detect",
                project=self.output_dir,
                name="train",
                fliplr=self.hyperparameters["fliplr"],
                flipud=self.hyperparameters["flipud"],
                mosaic=self.hyperparameters["mosaic"],
                mixup=self.hyperparameters["mixup"],
                copy_paste=self.hyperparameters["copy_paste"],
                hsv_h=self.hyperparameters["hsv_h"],
                hsv_s=self.hyperparameters["hsv_s"],
                hsv_v=self.hyperparameters["hsv_v"],
                box=self.hyperparameters["box"],
                cls=self.hyperparameters["cls"],
                dfl=self.hyperparameters["dfl"],
                verbose=True,
                exist_ok=True,
                patience=20,
                save_period=10,
                resume=bool(resume_path)
            )
            run_dir = self.find_latest_exp_dir(self.output_dir, "train")
            self.copy_best_weights(run_dir)
            if not resume_path:
                self.parse_yolo_results(run_dir)
                self.save_training_plot()
            final_results = {
                "best_map50": self.best_map50,
                "final_map50": self.training_history[-1]["map50"] if self.training_history else 0.0,
                "total_epochs": len(self.training_history),
                "hyperparameters": self.hyperparameters
            }
            with open(os.path.join(self.output_dir, "training_summary.json"), "w") as f:
                json.dump(final_results, f, indent=2, default=str)
            print(f"Training completed.")
            print(f"Best mAP50: {self.best_map50:.4f}")
            print(f"Final mAP50: {final_results['final_map50']:.4f}")
            print(f"Results saved to: {self.output_dir}")
            return final_results
        except Exception as e:
            print(f"Training failed with error: {str(e)}")
            return None

def main():
    DATA_YAML = r"/home/sailesh/IDDDetectionsYOLODataset/data.yaml"
    YOLO_MODEL = r"/home/sailesh/yolo11m.pt"
    OUTPUT_DIR = r"/home/sailesh/IDDDetectionsYOLODataset_TrainingOutput"
    if not os.path.exists(DATA_YAML):
        print(f"Dataset YAML not found: {DATA_YAML}")
        return
    if not os.path.exists(YOLO_MODEL):
        print(f"Model file not found: {YOLO_MODEL}")
        return
    trainer = YOLOTrainer(DATA_YAML, YOLO_MODEL, OUTPUT_DIR)
    resume_training = False
    latest_checkpoint = os.path.join(trainer.checkpoints_dir, "latest_checkpoint.json")
    if os.path.exists(latest_checkpoint) and sys.stdin.isatty():
        try:
            response = input("Previous training found. Resume? (y/n): ").lower().strip()
            resume_training = response == 'y'
        except (EOFError, KeyboardInterrupt):
            print("\nDefaulting to new training session")
            resume_training = False
    elif os.path.exists(latest_checkpoint):
        try:
            with open(latest_checkpoint, 'r') as f:
                ck = json.load(f)
            if ck.get('run_dir'):
                resume_training = True
        except Exception:
            resume_training = False
    results = trainer.train(resume=resume_training)
    if results:
        print("Training completed successfully.")
    else:
        print("Training failed.")

if __name__ == "__main__":
    main()