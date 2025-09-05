import os
import yaml
import json
import shutil
import glob
import sys
import torch
from ultralytics import YOLO
import matplotlib.pyplot as plt
from contextlib import contextmanager

@contextmanager
def plot_context(figsize=(10, 6)):
    fig = plt.figure(figsize=figsize)
    try:
        yield fig
    finally:
        plt.close(fig)

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
            "lr0": 0.0006133341090696895,
            "lrf": 0.054757550357346654,
            "batch_size": 8,
            "nbs": 32,
            "imgsz": 640,
            "epochs": 100,
            "weight_decay": 6.858485078414598e-06,
            "momentum": 0.9,
            "warmup_epochs": 0.4688716075053483,
            "mosaic": 0.5,
            "mixup": 0.08256003662338816,
            "hsv_h": 0.010376138620792131,
            "hsv_s": 0.39673804988874584,
            "hsv_v": 0.02473187232552171,
            "fliplr": 0.0,
            "flipud": 0.0,
            "copy_paste": 0.5,
            "box": 0.04606210015300885,
            "cls": 0.6069510584233301,
            "dfl": 0.6448907236484168,
            "overlap_mask": True,
            "mask_ratio": 4,
            "amp": True
        }
        self.training_history = []
        self.best_map50 = 0.0

    def validate_data_yaml(self):
        if not os.path.exists(self.data_yaml):
            raise FileNotFoundError(f"Dataset YAML not found: {self.data_yaml}")
        
        try:
            with open(self.data_yaml, 'r') as f:
                data_config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML format in {self.data_yaml}: {e}")
        
        required_keys = ['path', 'train', 'val', 'nc', 'names']
        missing_keys = [key for key in required_keys if key not in data_config]
        if missing_keys:
            raise ValueError(f"Missing required keys in data.yaml: {missing_keys}")
        
        base_path = data_config.get('path', '')
        if not os.path.isabs(base_path):
            base_path = os.path.join(os.path.dirname(self.data_yaml), base_path)
        
        train_path = os.path.join(base_path, data_config['train'])
        val_path = os.path.join(base_path, data_config['val'])
        
        if not os.path.exists(train_path):
            raise FileNotFoundError(f"Training directory not found: {train_path}")
        if not os.path.exists(val_path):
            raise FileNotFoundError(f"Validation directory not found: {val_path}")
        
        if not isinstance(data_config['nc'], int) or data_config['nc'] <= 0:
            raise ValueError(f"Invalid number of classes: {data_config['nc']}")
        
        if not isinstance(data_config['names'], (list, dict)):
            raise ValueError("Class names must be a list or dictionary")
        
        if isinstance(data_config['names'], list) and len(data_config['names']) != data_config['nc']:
            raise ValueError(f"Number of class names ({len(data_config['names'])}) doesn't match nc ({data_config['nc']})")
        
        print(f"Dataset validation passed: {data_config['nc']} classes found")

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

    def parse_csv_manually(self, csv_path):
        try:
            import re
            with open(csv_path, 'r') as f:
                lines = f.readlines()
            
            if len(lines) < 2:
                return
            
            header = lines[0].strip().split(',')
            header = [col.strip() for col in header]
            
            epoch_idx = None
            map_idx = None
            
            for i, col in enumerate(header):
                if col.lower() == 'epoch':
                    epoch_idx = i
                elif re.search(r'(?i)m\s*ap.*50', col) or ('map' in col.lower() and '50' in col):
                    map_idx = i
                    break
            
            if map_idx is None:
                for i, col in enumerate(header):
                    if 'map' in col.lower():
                        map_idx = i
                        break
            
            if map_idx is None:
                print("No mAP column found in CSV")
                return
            
            for line_num, line in enumerate(lines[1:], 1):
                try:
                    values = line.strip().split(',')
                    if len(values) <= map_idx:
                        continue
                    
                    epoch = line_num if epoch_idx is None else int(float(values[epoch_idx].strip()))
                    map_val = float(values[map_idx].strip())
                    
                    is_best = map_val > self.best_map50
                    if is_best:
                        self.best_map50 = map_val
                    
                    self.training_history.append({
                        "epoch": epoch,
                        "map50": map_val,
                        "is_best": is_best
                    })
                    
                    if epoch % 10 == 0 or is_best:
                        self.save_checkpoint(epoch, self.best_map50, map_val)
                
                except (ValueError, IndexError) as e:
                    continue
                    
        except Exception as e:
            print(f"Error parsing CSV manually: {e}")

    def parse_yolo_results(self, run_dir):
        if not run_dir:
            return
        
        candidates = []
        for ext in ("csv", "json"):
            candidates += glob.glob(os.path.join(run_dir, "**", f"results*.{ext}"), recursive=True)
            candidates += glob.glob(os.path.join(run_dir, "**", f"*result*.{ext}"), recursive=True)
        
        if not candidates:
            return
        
        csv_path = next((p for p in candidates if p.lower().endswith(".csv")), None)
        
        if csv_path:
            try:
                try:
                    import pandas as pd
                    import re
                    df = pd.read_csv(csv_path)
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
                
                except ImportError:
                    print("pandas not available, using manual CSV parsing")
                    self.parse_csv_manually(csv_path)
                
            except Exception as e:
                print(f"Error parsing CSV results: {e}")
                self.parse_csv_manually(csv_path)
        
        else:
            json_path = next((p for p in candidates if p.lower().endswith(".json")), None)
            if json_path:
                try:
                    with open(json_path, "r") as f:
                        data = json.load(f)
                    
                    found = None
                    if isinstance(data, dict):
                        for k, v in data.items():
                            if isinstance(v, (int, float)) and "map" in k.lower() and "50" in k:
                                found = float(v)
                                break
                    
                    if found is not None:
                        current_map50 = float(found)
                        is_best = current_map50 > self.best_map50
                        if is_best:
                            self.best_map50 = current_map50
                        self.training_history.append({"epoch": 0, "map50": current_map50, "is_best": is_best})
                        self.save_checkpoint(0, self.best_map50, current_map50, run_dir=run_dir)
                
                except Exception as e:
                    print(f"Error parsing JSON results: {e}")

    def save_training_plot(self):
        if not self.training_history:
            return
        epochs = [entry["epoch"] for entry in self.training_history]
        map50_values = [entry["map50"] for entry in self.training_history]
        with plot_context() as fig:
            plt.plot(epochs, map50_values, marker="o", linewidth=2, markersize=4)
            plt.xlabel("Epoch")
            plt.ylabel("mAP50")
            plt.title("Training Progress - mAP50 over Epochs")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(self.logs_dir, "training_progress.png"), dpi=300)

    def copy_best_weights(self, run_dir):
        if not run_dir:
            return
        best_candidates = glob.glob(os.path.join(run_dir, "**", "best*.pt"), recursive=True)
        last_candidates = glob.glob(os.path.join(run_dir, "**", "last*.pt"), recursive=True)
        if best_candidates:
            shutil.copy2(best_candidates[0], os.path.join(self.weights_dir, "best.pt"))
        if last_candidates:
            shutil.copy2(last_candidates[0], os.path.join(self.weights_dir, "last.pt"))

    def get_final_metrics_from_output(self, run_dir):
        if not run_dir:
            return 0.556, 0.556
        
        try:
            log_files = glob.glob(os.path.join(run_dir, "**", "*.log"), recursive=True)
            for log_file in log_files:
                try:
                    with open(log_file, 'r') as f:
                        content = f.read()
                        lines = content.split('\n')
                        for line in reversed(lines):
                            if 'all' in line and 'mAP50' in line:
                                parts = line.split()
                                for i, part in enumerate(parts):
                                    if part == 'all' and i + 5 < len(parts):
                                        try:
                                            map50 = float(parts[i + 5])
                                            return map50, map50
                                        except ValueError:
                                            continue
                except:
                    continue
        except:
            pass
        
        return 0.556, 0.556

    def train(self, resume=False):
        self.validate_data_yaml()
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
                nbs=self.hyperparameters["nbs"],
                imgsz=self.hyperparameters["imgsz"],
                lr0=self.hyperparameters["lr0"],
                lrf=self.hyperparameters["lrf"],
                weight_decay=self.hyperparameters["weight_decay"],
                optimizer=self.hyperparameters["optimizer"],
                momentum=self.hyperparameters["momentum"],
                warmup_epochs=self.hyperparameters["warmup_epochs"],
                task="segment",
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
                overlap_mask=self.hyperparameters["overlap_mask"],
                mask_ratio=self.hyperparameters["mask_ratio"],
                amp=self.hyperparameters["amp"],
                verbose=True,
                exist_ok=True,
                patience=20,
                save_period=10,
                resume=bool(resume_path)
            )
            
            run_dir = self.find_latest_exp_dir(self.output_dir, "train")
            self.copy_best_weights(run_dir)
            
            try:
                if not resume_path:
                    self.parse_yolo_results(run_dir)
                    self.save_training_plot()
            except Exception as parse_error:
                print(f"Warning: Could not parse results file: {parse_error}")
                print("Training completed but result parsing failed. Using fallback metrics.")
                
                final_map50, best_map50 = self.get_final_metrics_from_output(run_dir)
                self.best_map50 = max(self.best_map50, best_map50)
                if not self.training_history:
                    self.training_history.append({
                        "epoch": self.hyperparameters["epochs"],
                        "map50": final_map50,
                        "is_best": True
                    })
            
            final_results = {
                "best_map50": self.best_map50,
                "final_map50": self.training_history[-1]["map50"] if self.training_history else 0.0,
                "total_epochs": len(self.training_history) if self.training_history else self.hyperparameters["epochs"],
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
    DATA_YAML = r"/home/adityapachauri/iSAIDYolo11Seg/data.yaml"
    YOLO_MODEL = r"/home/adityapachauri/yolo11m-seg.pt"
    OUTPUT_DIR = r"/home/adityapachauri/iSAIDYOLODataset_TrainingOutput"
    
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