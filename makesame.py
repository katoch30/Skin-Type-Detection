from PIL import Image
import os

# Your input dataset folder
input_folder = r"C:\Users\Ritika\OneDrive\Desktop\4th sem\projectdav\data"  
# Output folder where cleaned images will be saved
output_folder = r"C:\Users\Ritika\OneDrive\Desktop\4th sem\projectdav\processdata"  
target_size = (128, 128)
output_format = 'JPEG'

os.makedirs(output_folder, exist_ok=True)
for root, _, files in os.walk(input_folder):
    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
            input_path = os.path.join(root, file)
            rel_path = os.path.relpath(input_path, input_folder)
            output_path = os.path.join(output_folder, os.path.splitext(rel_path)[0] + '.jpg')
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            try:
                with Image.open(input_path) as img:
                    img = img.convert('RGB')  # to handle grayscale or RGBA
                    img = img.resize(target_size, Image.LANCZOS)
                    img.save(output_path, output_format)
            except Exception as e:
                print(f'Error processing {input_path}: {e}')
