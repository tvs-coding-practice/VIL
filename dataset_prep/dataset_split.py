import os
import random
import shutil

# --- CONFIGURATION ---
source_folder = r"E:\vil-data\train\Domain 1\Emphysema"   # Use r"" for Windows paths
destination_folder = r"E:\vil-data\test\Domain 1\Emphysema"
percentage = 0.20  # 20%

# 1. Get list of valid image files
supported_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
all_files = [f for f in os.listdir(source_folder) if f.lower().endswith(supported_extensions)]

# 2. Calculate how many to move
num_files_to_move = int(len(all_files) * percentage)

# 3. Randomly select the files
files_to_move = random.sample(all_files, num_files_to_move)

print(f"Found {len(all_files)} images. Moving {num_files_to_move} ({percentage*100}%) files...")

# 4. Move the files
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

for file_name in files_to_move:
    src_path = os.path.join(source_folder, file_name)
    dst_path = os.path.join(destination_folder, file_name)
    shutil.move(src_path, dst_path)
    print(f"Moved: {file_name}")

print("Done!")