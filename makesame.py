import os
import random
from PIL import Image
import shutil
from torchvision import transforms
from tqdm import tqdm

# Parameters
input_dir = r"C:\Users\Ritika\OneDrive\Desktop\4th sem\projectdav\data"
output_dir = r"C:\Users\Ritika\OneDrive\Desktop\4th sem\projectdav\processed_data"
img_size = 128
test_ratio = 0.3
augment_count = 4
classes = ['dry', 'normal', 'oily']
image_format = 'jpeg'

# Augmentation transforms
augmentation_transforms = transforms.Compose([
    transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
    transforms.RandomAffine(degrees=20, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2)
])

# Resize transform
resize_transform = transforms.Compose([
    transforms.Resize((img_size, img_size))
])

# Clear and recreate processed_data directories
for split in ['train', 'test']:
    for cls in classes:
        path = os.path.join(output_dir, split, cls)
        os.makedirs(path, exist_ok=True)

# Process each class
for cls in classes:
    class_path = os.path.join(input_dir, cls)
    images = [f for f in os.listdir(class_path) if f.lower().endswith(('jpg', 'jpeg', 'png'))]
    random.shuffle(images)

    split_idx = int(len(images) * (1 - test_ratio))
    train_images = images[:split_idx]
    test_images = images[split_idx:]

    # ---- Test Images ----
    for img_name in test_images:
        img_path = os.path.join(class_path, img_name)
        try:
            img = Image.open(img_path).convert('RGB')
            img = resize_transform(img)
            save_path = os.path.join(output_dir, 'test', cls, os.path.splitext(img_name)[0] + '.jpeg')
            img.save(save_path, format='JPEG')
        except Exception as e:
            print(f"Error processing test image {img_name}: {e}")

    # ---- Train Images (and augmentation) ----
    for img_name in tqdm(train_images, desc=f'Processing {cls}'):
        img_path = os.path.join(class_path, img_name)
        try:
            original = Image.open(img_path).convert('RGB')
            original = resize_transform(original)

            # Save original resized image
            base_name = os.path.splitext(img_name)[0]
            original.save(os.path.join(output_dir, 'train', cls, f"{base_name}_orig.jpeg"), format='JPEG')

            # Save augmented images
            for i in range(augment_count):
                aug_img = augmentation_transforms(original)
                aug_img = resize_transform(aug_img)
                aug_img.save(os.path.join(output_dir, 'train', cls, f"{base_name}_aug{i+1}.jpeg"), format='JPEG')

        except Exception as e:
            print(f"Error processing train image {img_name}: {e}")
