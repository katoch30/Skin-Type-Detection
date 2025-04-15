import os
import shutil
import random

input_dir = r"C:\Users\Ritika\OneDrive\Desktop\4th sem\projectdav\processdata"  # folder with dry/, oily/, normal/
output_base = 'split_data'
train_ratio = 0.7  # 70% train, 20% test

random.seed(42)

classes = os.listdir(input_dir)

for cls in classes:
    cls_path = os.path.join(input_dir, cls)
    images = [img for img in os.listdir(cls_path) if img.endswith(('jpg', 'jpeg', 'png'))]
    random.shuffle(images)

    split_index = int(len(images) * train_ratio)
    train_images = images[:split_index]
    test_images = images[split_index:]

    for split, split_images in [('train', train_images), ('test', test_images)]:
        split_folder = os.path.join(output_base, split, cls)
        os.makedirs(split_folder, exist_ok=True)
        for img in split_images:
            src = os.path.join(cls_path, img)
            dst = os.path.join(split_folder, img)
            shutil.copy(src, dst)

print("Dataset split complete. Check the 'split_data/' folder.")
