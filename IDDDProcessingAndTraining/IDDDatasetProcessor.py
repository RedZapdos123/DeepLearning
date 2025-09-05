import os
import xml.etree.ElementTree as ET
import shutil
import random
from pathlib import Path
import yaml
from tqdm import tqdm

def parse_xml_annotation(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    filename = root.find('filename').text
    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)
    
    objects = []
    for obj in root.findall('object'):
        name = obj.find('name').text
        bndbox = obj.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)
        
        objects.append({
            'name': name,
            'bbox': [xmin, ymin, xmax, ymax]
        })
    
    return filename, width, height, objects

def convert_bbox_to_yolo(bbox, img_width, img_height):
    xmin, ymin, xmax, ymax = bbox
    
    x_center = (xmin + xmax) / 2.0 / img_width
    y_center = (ymin + ymax) / 2.0 / img_height
    width = (xmax - xmin) / img_width
    height = (ymax - ymin) / img_height
    
    return [x_center, y_center, width, height]

def collect_all_files(base_path):
    image_files = []
    annotation_files = []
    
    jpeg_path = os.path.join(base_path, 'JPEGImages')
    annotations_path = os.path.join(base_path, 'Annotations')
    
    print("Collecting image files...")
    for root, dirs, files in tqdm(os.walk(jpeg_path), desc="Scanning directories"):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_files.append(os.path.join(root, file))
    
    print("Collecting annotation files...")
    for root, dirs, files in tqdm(os.walk(annotations_path), desc="Scanning directories"):
        for file in files:
            if file.lower().endswith('.xml'):
                annotation_files.append(os.path.join(root, file))
    
    return image_files, annotation_files

def generate_unique_filename(file_path, base_path):
    rel_path = os.path.relpath(file_path, base_path)
    path_parts = rel_path.split(os.sep)
    
    if len(path_parts) >= 2:
        parent_folder = path_parts[-2]
        filename = path_parts[-1]
        name, ext = os.path.splitext(filename)
        unique_name = f"{parent_folder}_{name}{ext}"
        return unique_name
    else:
        return os.path.basename(file_path)

def find_matching_annotation(image_path, annotation_files, jpeg_base_path, ann_base_path):
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    
    image_rel_path = os.path.relpath(image_path, jpeg_base_path)
    image_path_parts = image_rel_path.split(os.sep)
    
    for ann_path in annotation_files:
        ann_name = os.path.splitext(os.path.basename(ann_path))[0]
        if ann_name == image_name:
            ann_rel_path = os.path.relpath(ann_path, ann_base_path)
            ann_path_parts = ann_rel_path.split(os.sep)
            
            if len(image_path_parts) >= 2 and len(ann_path_parts) >= 2:
                image_parent = image_path_parts[-2]
                ann_parent = ann_path_parts[-2]
                if image_parent in ann_parent or ann_parent in image_parent:
                    return ann_path
            elif ann_name == image_name:
                return ann_path
    return None

def create_yolo_dataset(input_path, output_path, class_names):
    os.makedirs(output_path, exist_ok=True)
    
    train_images_dir = os.path.join(output_path, 'train', 'images')
    train_labels_dir = os.path.join(output_path, 'train', 'labels')
    val_images_dir = os.path.join(output_path, 'val', 'images')
    val_labels_dir = os.path.join(output_path, 'val', 'labels')
    test_images_dir = os.path.join(output_path, 'test', 'images')
    test_labels_dir = os.path.join(output_path, 'test', 'labels')
    
    for dir_path in [train_images_dir, train_labels_dir, val_images_dir, val_labels_dir, test_images_dir, test_labels_dir]:
        os.makedirs(dir_path, exist_ok=True)
    
    image_files, annotation_files = collect_all_files(input_path)
    
    jpeg_base_path = os.path.join(input_path, 'JPEGImages')
    ann_base_path = os.path.join(input_path, 'Annotations')
    
    print("Matching image-annotation pairs...")
    valid_pairs = []
    for image_path in tqdm(image_files, desc="Matching pairs"):
        ann_path = find_matching_annotation(image_path, annotation_files, jpeg_base_path, ann_base_path)
        if ann_path:
            valid_pairs.append((image_path, ann_path))
    
    random.shuffle(valid_pairs)
    
    total_files = len(valid_pairs)
    train_split = int(0.8 * total_files)
    val_split = int(0.9 * total_files)
    
    train_pairs = valid_pairs[:train_split]
    val_pairs = valid_pairs[train_split:val_split]
    test_pairs = valid_pairs[val_split:]
    
    def process_split(pairs, images_dir, labels_dir, split_name):
        for i, (image_path, ann_path) in enumerate(tqdm(pairs, desc=f"Processing {split_name}")):
            try:
                filename, img_width, img_height, objects = parse_xml_annotation(ann_path)
                
                unique_image_name = generate_unique_filename(image_path, jpeg_base_path)
                unique_label_name = os.path.splitext(unique_image_name)[0] + '.txt'
                
                shutil.copy2(image_path, os.path.join(images_dir, unique_image_name))
                
                label_path = os.path.join(labels_dir, unique_label_name)
                with open(label_path, 'w') as f:
                    for obj in objects:
                        class_name = obj['name']
                        if class_name in class_names:
                            class_id = class_names.index(class_name)
                            bbox = obj['bbox']
                            yolo_bbox = convert_bbox_to_yolo(bbox, img_width, img_height)
                            
                            line = f"{class_id} {yolo_bbox[0]:.6f} {yolo_bbox[1]:.6f} {yolo_bbox[2]:.6f} {yolo_bbox[3]:.6f}\n"
                            f.write(line)
                
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
    
    process_split(train_pairs, train_images_dir, train_labels_dir, "train")
    process_split(val_pairs, val_images_dir, val_labels_dir, "val")
    process_split(test_pairs, test_images_dir, test_labels_dir, "test")
    
    print(f"Dataset conversion complete.")
    print(f"Train: {len(train_pairs)} images")
    print(f"Val: {len(val_pairs)} images")
    print(f"Test: {len(test_pairs)} images")

def extract_class_names(input_path):
    class_names = set()
    
    annotations_path = os.path.join(input_path, 'Annotations')
    
    xml_files = []
    for root, dirs, files in os.walk(annotations_path):
        for file in files:
            if file.lower().endswith('.xml'):
                xml_files.append(os.path.join(root, file))
    
    for xml_path in tqdm(xml_files, desc="Extracting class names"):
        try:
            tree = ET.parse(xml_path)
            root_elem = tree.getroot()
            
            for obj in root_elem.findall('object'):
                name = obj.find('name').text
                class_names.add(name)
        except Exception as e:
            print(f"Error parsing {xml_path}: {e}")
    
    return sorted(list(class_names))

def create_data_yaml(output_path, class_names):
    data_yaml = {
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'nc': len(class_names),
        'names': class_names
    }
    
    yaml_path = os.path.join(output_path, 'data.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False)
    
    print(f"Created data.yaml with {len(class_names)} classes: {class_names}")

def main():
    input_path = r"C:\Users\Xeron\Downloads\idd-detection\IDD_Detection"
    output_path = r"C:\Users\Xeron\OneDrive\Documents\LargeDatasets\IDDDetectionsYOLODataset"
    
    if not os.path.exists(input_path):
        print(f"Input path does not exist: {input_path}")
        return
    
    if not os.path.exists(os.path.join(input_path, 'JPEGImages')):
        print(f"JPEGImages folder not found in {input_path}")
        return
    
    if not os.path.exists(os.path.join(input_path, 'Annotations')):
        print(f"Annotations folder not found in {input_path}")
        return
    
    print("Extracting class names...")
    class_names = extract_class_names(input_path)
    
    if not class_names:
        print("No classes found in annotations.")
        return
    
    print(f"Found classes: {class_names}")
    
    print("Converting dataset...")
    create_yolo_dataset(input_path, output_path, class_names)
    
    print("Creating data.yaml...")
    create_data_yaml(output_path, class_names)
    
    print("Conversion complete.")

if __name__ == "__main__":
    main()