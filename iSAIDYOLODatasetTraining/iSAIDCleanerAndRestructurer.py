import os
import shutil
import random
from pathlib import Path
from tqdm import tqdm

def remove_empty_label_pairs(images_dir, labels_dir):
    removed_count = 0
    
    if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
        print(f"Directory not found: {images_dir} or {labels_dir}")
        return removed_count
        
    label_files = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]
    
    for label_file in tqdm(label_files, desc=f"Cleaning {os.path.basename(images_dir)}"):
        label_path = os.path.join(labels_dir, label_file)
        
        try:
            if os.path.getsize(label_path) == 0:
                image_name = os.path.splitext(label_file)[0]
                
                for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']:
                    image_path = os.path.join(images_dir, image_name + ext)
                    if os.path.exists(image_path):
                        os.remove(image_path)
                        os.remove(label_path)
                        removed_count += 1
                        break
        except OSError:
            continue
    
    return removed_count

def count_valid_pairs(images_dir, labels_dir):
    if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
        return 0
    
    valid_count = 0
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']:
        image_files.extend(list(Path(images_dir).glob(ext)))
        image_files.extend(list(Path(images_dir).glob(ext.upper())))
    
    for image_path in image_files:
        label_path = Path(labels_dir) / f"{image_path.stem}.txt"
        if label_path.exists():
            valid_count += 1
    
    return valid_count

def collect_valid_pairs(images_dir, labels_dir):
    valid_pairs = []
    
    if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
        return valid_pairs
    
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']:
        image_files.extend(list(Path(images_dir).glob(ext)))
        image_files.extend(list(Path(images_dir).glob(ext.upper())))
    
    for image_path in image_files:
        label_path = Path(labels_dir) / f"{image_path.stem}.txt"
        if label_path.exists():
            valid_pairs.append((image_path, label_path))
    
    return valid_pairs

def restructure_dataset(dataset_dir, output_dir, train_ratio=0.8):
    random.seed(42)
    
    dataset_path = Path(dataset_dir)
    output_path = Path(output_dir)
    
    train_images_dir = dataset_path / "images" / "train"
    train_labels_dir = dataset_path / "labels" / "train"
    val_images_dir = dataset_path / "images" / "val"
    val_labels_dir = dataset_path / "labels" / "val"
    test_images_dir = dataset_path / "images" / "test"
    
    if not (train_images_dir.exists() and train_labels_dir.exists()):
        print("Train directories not found. Expected structure:")
        print(f"  {train_images_dir}")
        print(f"  {train_labels_dir}")
        return
    
    print("Step 1: Counting original pairs...")
    original_train_count = count_valid_pairs(str(train_images_dir), str(train_labels_dir))
    original_val_count = count_valid_pairs(str(val_images_dir), str(val_labels_dir))
    print(f"Original train pairs: {original_train_count}")
    print(f"Original val pairs: {original_val_count}")
    print(f"Total original pairs: {original_train_count + original_val_count}")
    
    print("\nStep 2: Cleaning empty label pairs...")
    
    removed_train = remove_empty_label_pairs(str(train_images_dir), str(train_labels_dir))
    print(f"Removed {removed_train} empty pairs from train")
    
    removed_val = 0
    if val_images_dir.exists() and val_labels_dir.exists():
        removed_val = remove_empty_label_pairs(str(val_images_dir), str(val_labels_dir))
        print(f"Removed {removed_val} empty pairs from val")
    
    total_removed = removed_train + removed_val
    print(f"Total empty pairs removed: {total_removed}")
    
    print("\nStep 3: Collecting remaining valid pairs...")
    
    train_pairs = collect_valid_pairs(str(train_images_dir), str(train_labels_dir))
    print(f"Collected {len(train_pairs)} pairs from train")
    
    val_pairs = collect_valid_pairs(str(val_images_dir), str(val_labels_dir))
    print(f"Collected {len(val_pairs)} pairs from val")
    
    all_valid_pairs = train_pairs + val_pairs
    total_valid = len(all_valid_pairs)
    
    if total_valid == 0:
        print("No valid pairs found after cleaning!")
        return
    
    print(f"Total valid pairs to restructure: {total_valid}")
    
    print("\nStep 4: Creating output structure...")
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    new_train_images_dir = output_path / "train" / "images"
    new_train_labels_dir = output_path / "train" / "labels"
    new_val_images_dir = output_path / "val" / "images"
    new_val_labels_dir = output_path / "val" / "labels"
    new_test_images_dir = output_path / "test" / "images"
    
    for dir_path in [new_train_images_dir, new_train_labels_dir, new_val_images_dir, new_val_labels_dir, new_test_images_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    random.shuffle(all_valid_pairs)
    
    new_train_count = int(total_valid * train_ratio)
    new_val_count = total_valid - new_train_count
    
    new_train_pairs = all_valid_pairs[:new_train_count]
    new_val_pairs = all_valid_pairs[new_train_count:]
    
    print(f"New split: {len(new_train_pairs)} train, {len(new_val_pairs)} val")
    
    actual_train_ratio = len(new_train_pairs) / total_valid
    actual_val_ratio = len(new_val_pairs) / total_valid
    print(f"Actual ratios - Train: {actual_train_ratio:.1%}, Val: {actual_val_ratio:.1%}")
    
    for image_path, label_path in tqdm(new_train_pairs, desc="Copying train data"):
        shutil.copy2(image_path, new_train_images_dir / image_path.name)
        shutil.copy2(label_path, new_train_labels_dir / label_path.name)
    
    for image_path, label_path in tqdm(new_val_pairs, desc="Copying val data"):
        shutil.copy2(image_path, new_val_images_dir / image_path.name)
        shutil.copy2(label_path, new_val_labels_dir / label_path.name)
    
    if test_images_dir.exists():
        print("Copying existing test images...")
        test_images = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']:
            test_images.extend(list(test_images_dir.glob(ext)))
            test_images.extend(list(test_images_dir.glob(ext.upper())))
        
        for image_path in tqdm(test_images, desc="Copying test data"):
            shutil.copy2(image_path, new_test_images_dir / image_path.name)
        
        print(f"Copied {len(test_images)} test images")
    else:
        print("No test directory found")
    
    if output_path != dataset_path:
        print("Source dataset preserved, output created in separate directory")
    
    data_yaml_content = f"""path: {output_path.name}/
train: train/images
val: val/images
test: test/images

nc: 16

names: ["unlabeled", "ship", "storage_tank", "baseball_diamond", "tennis_court", "basketball_court", "Ground_Track_Field", "Bridge", "Large_Vehicle", "Small_Vehicle", "Helicopter", "Swimming_pool", "Roundabout", "Soccer_ball_field", "plane", "Harbor"]
"""
    
    with open(output_path / "data.yaml", "w") as f:
        f.write(data_yaml_content)
    
    print(f"\nDataset restructuring completed!")
    print(f"Empty pairs removed: {total_removed}")
    print(f"Final train pairs: {len(new_train_pairs)}")
    print(f"Final val pairs: {len(new_val_pairs)}")
    if test_images_dir.exists():
        test_count = len([f for f in new_test_images_dir.iterdir() if f.is_file()])
        print(f"Final test images: {test_count}")
    print(f"Total train/val processed: {total_valid}")
    print(f"Updated data.yaml created in: {output_path / 'data.yaml'}")

if __name__ == "__main__":
    dataset_dir = r"C:\Users\Xeron\OneDrive\Documents\LargeDatasets\iSAIDYolo11Seg"
    output_dir = r"C:\Users\Xeron\OneDrive\Documents\LargeDatasets\iSAIDYolo11Seg_Restructured"
    restructure_dataset(dataset_dir, output_dir)