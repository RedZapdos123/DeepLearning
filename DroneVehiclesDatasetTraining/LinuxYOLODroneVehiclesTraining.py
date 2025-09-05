#This program is used to train a YOLOv11-OBB model on the dataset: https://www.kaggle.com/datasets/redzapdos123/dronevehicle-dataset-yolov11-obb 

import os
import yaml
import json
import shutil
import glob
import sys
import argparse
from pathlib import Path
import torch
from ultralytics import YOLO
import matplotlib.pyplot as plt

try:
    import pandas as pd
except ImportError:
    pd = None

#This class validates the inputs, saves the hyperparameters, loads the checkpoint, 
#finds the latest experiment directory, parses the YOLO results, saves the training plot, 
#copies the best weights, and trains the model.
class YOLOTrainer:
    def __init__(self, data_yaml, model_path, output_dir):
        self.data_yaml = Path(data_yaml)
        self.model_path = Path(model_path)
        self.output_dir = Path(output_dir)
        self.weights_dir = self.output_dir / "weights"
        self.checkpoints_dir = self.output_dir / "checkpoints"
        self.logs_dir = self.output_dir / "logs"
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.weights_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        #The tuned hyperparameters from the Optuna hyperparameter tuning.
        self.hyperparameters = {
            "batch_size": 16,
            "degrees": 24.78546483753358,
            "epochs": 100,
            "fliplr": 0.0,
            "flipud": 0.0,
            "hsv_h": 0.03942101767102896,
            "hsv_s": 0.3760147672081699,
            "hsv_v": 0.10636265610164122,
            "imgsz": 960,
            "lr0": 0.0003328109781299159,
            "lrf": 0.08880603276186354,
            "mixup": 0.29754132980622755,
            "mosaic": 0.0,
            "perspective": 0.0007102058049893095,
            "scale": 0.32769924061466615,
            "shear": 0.29971929181318724,
            "weight_decay": 6.723021440127589e-05,
            "patience": 20
        }
        self.training_history = []
        self.best_map50 = 0.0

    #This function validates the inputs.
    def validate_inputs(self):
        if not self.data_yaml.exists():
            raise FileNotFoundError(f"Dataset YAML not found: {self.data_yaml}")
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        try:
            with open(self.data_yaml, 'r') as f:
                data_config = yaml.safe_load(f)
            
            required_fields = ['train', 'val', 'nc', 'names']
            missing_fields = [field for field in required_fields if field not in data_config]
            if missing_fields:
                raise ValueError(f"Missing required fields in YAML: {missing_fields}")
            
            for path_field in ['train', 'val']:
                if path_field in data_config:
                    dataset_path = Path(data_config[path_field])
                    if not dataset_path.is_absolute():
                        dataset_path = self.data_yaml.parent / dataset_path
                    if not dataset_path.exists():
                        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")
                        
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML file: {e}")

    #This function saves the hyperparameters.
    def save_hyperparameters(self):
        hyperparams_path = self.output_dir / "hyperparameters.yaml"
        with open(hyperparams_path, "w") as f:
            yaml.dump(self.hyperparameters, f, default_flow_style=False)

    #This function saves the checkpoint.
    def save_checkpoint(self, epoch, best_map50, current_map50, run_dir=None):
        checkpoint = {
            "epoch": epoch,
            "hyperparameters": self.hyperparameters,
            "best_map50": best_map50,
            "current_map50": current_map50,
            "training_history": self.training_history,
            "run_dir": str(run_dir) if run_dir else None
        }
        
        checkpoint_path = self.checkpoints_dir / f"checkpoint_epoch_{epoch}.json"
        with open(checkpoint_path, "w") as f:
            json.dump(checkpoint, f, indent=2, default=str)
        
        latest_checkpoint_path = self.checkpoints_dir / "latest_checkpoint.json"
        with open(latest_checkpoint_path, "w") as f:
            json.dump(checkpoint, f, indent=2, default=str)
        
        print(f"Checkpoint saved for epoch {epoch} (mAP50: {current_map50:.4f})")

    #This function loads the checkpoint.
    def load_checkpoint(self):
        latest_checkpoint_path = self.checkpoints_dir / "latest_checkpoint.json"
        if latest_checkpoint_path.exists():
            try:
                with open(latest_checkpoint_path, "r") as f:
                    checkpoint = json.load(f)
                self.training_history = checkpoint.get("training_history", [])
                self.best_map50 = checkpoint.get("best_map50", 0.0)
                return checkpoint
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Warning: Could not load checkpoint - {e}")
                return None
        return None

    #This function finds the latest experiment directory.
    def find_latest_exp_dir(self, project_dir, name="train"):
        base_path = Path(project_dir) / name
        if not base_path.exists():
            return None
        
        subdirs = [d for d in base_path.iterdir() if d.is_dir()]
        if not subdirs:
            return base_path if base_path.exists() else None
        
        subdirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return subdirs[0]

    #This function parses the YOLO results.
    def parse_yolo_results(self, run_dir):
        if not run_dir or not Path(run_dir).exists():
            print("Warning: Run directory not found for result parsing")
            return

        run_path = Path(run_dir)
        results_files = []
        
        results_files.extend(run_path.rglob("results.csv"))
        results_files.extend(run_path.rglob("*results*.csv"))
        
        if not results_files:
            print("Warning: No results files found")
            return

        results_file = results_files[0]
        
        try:
            if pd is not None and results_file.suffix.lower() == '.csv':
                df = pd.read_csv(results_file)
                df.columns = df.columns.str.strip()
                
                if "epoch" in df.columns:
                    epochs = df["epoch"].astype(int).tolist()
                else:
                    epochs = list(range(1, len(df) + 1))
                
                map_col = None
                possible_map_cols = [col for col in df.columns if 'map50' in col.lower() or 'map@50' in col.lower()]
                if not possible_map_cols:
                    possible_map_cols = [col for col in df.columns if 'map' in col.lower() and '50' in col]
                if not possible_map_cols:
                    possible_map_cols = [col for col in df.columns if 'map' in col.lower()]
                
                if possible_map_cols:
                    map_col = possible_map_cols[0]
                else:
                    print("Warning: No mAP column found in results")
                    return
                
                for ep, val in zip(epochs, df[map_col].fillna(0.0)):
                    current_map50 = float(val)
                    is_best = current_map50 > self.best_map50
                    if is_best:
                        self.best_map50 = current_map50
                    
                    self.training_history.append({
                        "epoch": int(ep), 
                        "map50": current_map50, 
                        "is_best": is_best
                    })
                    
                    if int(ep) % 10 == 0 or is_best:
                        self.save_checkpoint(int(ep), self.best_map50, current_map50, run_dir=run_dir)
            else:
                print("Warning: pandas not available or file not CSV, skipping result parsing")
                
        except Exception as e:
            print(f"Warning: Error parsing YOLO results - {e}")

    #This function saves the training plot.
    def save_training_plot(self):
        if not self.training_history:
            print("Warning: No training history to plot")
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
        
        plot_path = self.logs_dir / "training_progress.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Training plot saved to: {plot_path}")

    #This function copies the best weights.
    def copy_best_weights(self, run_dir):
        if not run_dir:
            print("Warning: No run directory provided for weight copying")
            return
        
        run_path = Path(run_dir)
        
        best_weights = list(run_path.rglob("best.pt"))
        last_weights = list(run_path.rglob("last.pt"))
        
        if best_weights:
            dest_path = self.weights_dir / "best.pt"
            shutil.copy2(best_weights[0], dest_path)
            print(f"Best weights copied to: {dest_path}")
        else:
            print("Warning: No best.pt weights found")
            
        if last_weights:
            dest_path = self.weights_dir / "last.pt"
            shutil.copy2(last_weights[0], dest_path)
            print(f"Last weights copied to: {dest_path}")
        else:
            print("Warning: No last.pt weights found")

    #This function trains the model.
    def train(self, resume=False):
        try:
            self.validate_inputs()
        except (FileNotFoundError, ValueError) as e:
            print(f"Validation error: {e}")
            return None
        
        self.save_hyperparameters()
        resume_path = None
        
        if resume:
            checkpoint = self.load_checkpoint()
            last_run = self.find_latest_exp_dir(self.output_dir, "train")
            
            if checkpoint and last_run:
                last_weights_files = list(Path(last_run).rglob("last.pt"))
                if last_weights_files:
                    resume_path = str(last_weights_files[0])
                    print(f"Resuming training from {resume_path}")
                    print(f"Best mAP50 so far: {self.best_map50:.4f}")
                else:
                    print("Warning: No last.pt found for resume, starting fresh")

        model = YOLO(str(resume_path) if resume_path else str(self.model_path))
        
        #This is the training loop.
        try:
            results = model.train(
                data=str(self.data_yaml),
                epochs=self.hyperparameters["epochs"],
                batch=self.hyperparameters["batch_size"],
                imgsz=self.hyperparameters["imgsz"],
                lr0=self.hyperparameters["lr0"],
                lrf=self.hyperparameters["lrf"],
                weight_decay=self.hyperparameters["weight_decay"],
                optimizer="AdamW",
                task="obb",
                project=str(self.output_dir),
                name="train",
                degrees=self.hyperparameters["degrees"],
                perspective=self.hyperparameters["perspective"],
                fliplr=self.hyperparameters["fliplr"],
                flipud=self.hyperparameters["flipud"],
                mosaic=self.hyperparameters["mosaic"],
                mixup=self.hyperparameters["mixup"],
                hsv_h=self.hyperparameters["hsv_h"],
                hsv_s=self.hyperparameters["hsv_s"],
                hsv_v=self.hyperparameters["hsv_v"],
                scale=self.hyperparameters["scale"],
                shear=self.hyperparameters["shear"],
                verbose=True,
                exist_ok=True,
                patience=self.hyperparameters["patience"],
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
                "hyperparameters": self.hyperparameters,
                "run_directory": str(run_dir) if run_dir else None
            }
            
            summary_path = self.output_dir / "training_summary.json"
            with open(summary_path, "w") as f:
                json.dump(final_results, f, indent=2, default=str)
            
            print(f"Training completed successfully.")
            print(f"Best mAP50: {self.best_map50:.4f}")
            print(f"Final mAP50: {final_results['final_map50']:.4f}")
            print(f"Results saved to: {self.output_dir}")
            
            return final_results
            
        except Exception as e:
            print(f"Training failed: {str(e)}")
            return None

#This function parses the arguments.
def parse_arguments():
    parser = argparse.ArgumentParser(description='YOLO OBB Training Script')
    parser.add_argument('--data', type=str, 
                       default=r"/home/sailesh/DroneVehiclesDatasetYOLO/data.yaml",
                       help='Path to dataset YAML file')
    parser.add_argument('--model', type=str, 
                       default=r"/home/sailesh/yolo11m-obb.pt",
                       help='Path to YOLO model file')
    parser.add_argument('--output', type=str, 
                       default=r"/home/sailesh/DroneVehiclesYOLO_OBB_TrainingOutput",
                       help='Output directory for training results')
    parser.add_argument('--resume', action='store_true', help='Resume training from last checkpoint')
    parser.add_argument('--auto-resume', action='store_true', help='Automatically resume if checkpoint exists')
    return parser.parse_args()

#This is the main function.
def main():
    args = parse_arguments()
    
    trainer = YOLOTrainer(args.data, args.model, args.output)
    
    resume_training = args.resume
    
    #This automatically resumes the training if a checkpoint exists.
    if args.auto_resume and not resume_training:
        latest_checkpoint = trainer.checkpoints_dir / "latest_checkpoint.json"
        if latest_checkpoint.exists():
            try:
                with open(latest_checkpoint, 'r') as f:
                    checkpoint_data = json.load(f)
                if checkpoint_data.get('run_dir'):
                    resume_training = True
                    print("Auto-resuming from existing checkpoint")
            except Exception:
                pass
    
    #This asks the user if they want to resume the training.
    if not resume_training and not args.auto_resume:
        latest_checkpoint = trainer.checkpoints_dir / "latest_checkpoint.json"
        if latest_checkpoint.exists() and sys.stdin.isatty():
            try:
                response = input("Previous training found. Resume? (y/n): ").lower().strip()
                resume_training = response == 'y'
            except (EOFError, KeyboardInterrupt):
                print("\nDefaulting to new training session")
                resume_training = False
    
    results = trainer.train(resume=resume_training)
    
    if results:
        print("Training completed successfully.")
        return 0
    else:
        print("Training failed.")
        return 1

if __name__ == "__main__":
    sys.exit(main())