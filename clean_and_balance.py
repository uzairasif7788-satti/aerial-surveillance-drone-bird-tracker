import os
import cv2
import random
import shutil

# Blurry images remove karne ka function
def remove_blur_images(split_name, blur_threshold=100.0):
    images_dir = os.path.join(split_name, 'images')
    labels_dir = os.path.join(split_name, 'labels')
    
    if not os.path.exists(images_dir):
        print(f"Directory not found: {images_dir}")
        return

    removed_count = 0
    for filename in os.listdir(images_dir):
        if filename.endswith(('.jpg', '.png', '.jpeg')):
            img_path = os.path.join(images_dir, filename)
            gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
            if gray is not None:
                score = cv2.Laplacian(gray, cv2.CV_64F).var()
                if score < blur_threshold:
                    os.remove(img_path)
                    label_path = os.path.join(labels_dir, filename.rsplit('.', 1)[0] + '.txt')
                    if os.path.exists(label_path):
                        os.remove(label_path)
                    removed_count += 1
                    
    print(f"[{split_name}] Removed {removed_count} blurry images.")

# Step 1: Teeno splits se blurry images hata dein
for split in ['train', 'valid', 'test']:
    remove_blur_images(split, blur_threshold=100.0)


# Dataset balancing function (Train set ke liye)
def balance_train_dataset():
    images_dir = 'train/images'
    labels_dir = 'train/labels'
    
    if not os.path.exists(labels_dir):
        print(f"Directory not found: {labels_dir}")
        return

    class_images = {0: [], 1: []} # 0: Bird, 1: Drone

    for label_file in os.listdir(labels_dir):
        if label_file.endswith('.txt'):
            label_path = os.path.join(labels_dir, label_file)
            img_name = label_file.rsplit('.', 1)[0] + '.jpg'
            img_path = os.path.join(images_dir, img_name)
            
            if not os.path.exists(img_path):
                img_name = label_file.rsplit('.', 1)[0] + '.png'
                img_path = os.path.join(images_dir, img_name)

            if os.path.exists(img_path):
                with open(label_path, 'r') as f:
                    file_classes = {int(line.strip().split()[0]) for line in f.readlines()}
                
                for cls_id in file_classes:
                    if cls_id in class_images:
                        class_images[cls_id].append((img_path, label_path))

    if not class_images[0] or not class_images[1]:
        print("Kuch classes mein data nahi mila, balancing skip ho rahi hai.")
        return

    min_count = min(len(class_images[0]), len(class_images[1]))
    print(f"Balancing classes to {min_count} images each...")

    selected_files = set()
    for cls_id, files in class_images.items():
        random.shuffle(files)
        for img_p, lbl_p in files[:min_count]:
            selected_files.add((img_p, lbl_p))

    all_current_images = set()
    for cls_id, files in class_images.items():
        for img_p, lbl_p in files:
            all_current_images.add((img_p, lbl_p))

    to_remove = all_current_images - selected_files
    for img_p, lbl_p in to_remove:
        if os.path.exists(img_p): os.remove(img_p)
        if os.path.exists(lbl_p): os.remove(lbl_p)

    print("Dataset cleaning and balancing completed successfully!")

# Step 2: Train set ko balance karein
balance_train_dataset()