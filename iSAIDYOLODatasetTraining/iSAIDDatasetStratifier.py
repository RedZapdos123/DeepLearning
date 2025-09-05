import os
import shutil
from sklearn.model_selection import train_test_split
from tqdm import tqdm

INPUT_DIR = r"C:\Users\Xeron\OneDrive\Documents\LargeDatasets\iSAIDYolo11Seg"
OUTPUT_DIR = r"C:\Users\Xeron\OneDrive\Desktop\StratifiediSAIDYolo11SegDataset"
SAMPLE_FRACTION = 0.1
RANDOM_STATE = 17

def gather_split_data(split_name):
    records = []
    img_dir = os.path.join(INPUT_DIR, split_name, 'images')
    lbl_dir = os.path.join(INPUT_DIR, split_name, 'labels')
    
    if not os.path.isdir(lbl_dir) or not os.path.isdir(img_dir):
        return records
    
    label_files = [f for f in os.listdir(lbl_dir) if f.lower().endswith('.txt')]
    
    for lbl_file in tqdm(label_files, desc=f"Gathering {split_name} data"):
        img_id = lbl_file[:-4]
        
        for ext in ('.jpg', '.jpeg', '.png'):
            img_path = os.path.join(img_dir, img_id + ext)
            if os.path.exists(img_path):
                break
        else:
            continue
        
        lbl_path = os.path.join(lbl_dir, lbl_file)
        records.append((img_path, lbl_path))
    
    return records

def write_split(records, split_name):
    img_out = os.path.join(OUTPUT_DIR, split_name, 'images')
    lbl_out = os.path.join(OUTPUT_DIR, split_name, 'labels')
    os.makedirs(img_out, exist_ok=True)
    os.makedirs(lbl_out, exist_ok=True)
    
    for img_path, lbl_path in tqdm(records, desc=f"Writing {split_name} files"):
        shutil.copy2(img_path, img_out)
        shutil.copy2(lbl_path, lbl_out)

def create_data_yaml():
    output_yaml = os.path.join(OUTPUT_DIR, 'data.yaml')
    
    original_yaml = os.path.join(INPUT_DIR, 'data.yaml')
    class_names = []
    
    if os.path.exists(original_yaml):
        with open(original_yaml, 'r') as f:
            for line in f:
                if line.strip().startswith('names:'):
                    names_content = line.strip()
                    break
            else:
                names_content = "names: []"
    else:
        names_content = "names: []"
    
    yaml_content = f"""path: StratifiediSAIDYolo11SegDataset/
train: train/images
val: val/images
test: test/images

{names_content}
"""
    
    with open(output_yaml, 'w') as f:
        f.write(yaml_content)
    
    return True

def main():
    print("Starting random 10% dataset sampling...")
    
    train_data = gather_split_data('train')
    val_data = gather_split_data('val')
    test_data = gather_split_data('test')
    
    if not train_data:
        raise RuntimeError("No training data found")
    
    print("Sampling data...")
    train_sample, _ = train_test_split(train_data, train_size=SAMPLE_FRACTION, random_state=RANDOM_STATE)
    
    if val_data:
        val_sample, _ = train_test_split(val_data, train_size=SAMPLE_FRACTION, random_state=RANDOM_STATE)
    else:
        val_sample = []
    
    if test_data:
        test_sample, _ = train_test_split(test_data, train_size=SAMPLE_FRACTION, random_state=RANDOM_STATE)
    else:
        test_sample = []
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    write_split(train_sample, 'train')
    if val_sample:
        write_split(val_sample, 'val')
    if test_sample:
        write_split(test_sample, 'test')
    
    print("Creating data.yaml...")
    create_data_yaml()
    
    print(f"Finished! Random 10% sample written to: {OUTPUT_DIR}")
    print(f"Train samples: {len(train_sample)}")
    print(f"Val samples: {len(val_sample)}")
    print(f"Test samples: {len(test_sample)}")

if __name__ == '__main__':
    main()