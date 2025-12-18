import os
import shutil
import csv
from brisque import BRISQUE
from PIL import Image
import numpy as np


def process_images_for_brisque(input_folder, output_csv, top_k_folder, k=5):
    obj = BRISQUE(url=False)
    image_data = []  # List to store (filename, score)

    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')

    print(f"Starting BRISQUE evaluation in: {input_folder}")

    files = [f for f in os.listdir(input_folder) if f.lower().endswith(valid_extensions)]

    for idx, filename in enumerate(files):
        image_path = os.path.join(input_folder, filename)

        try:
            # Convert to RGB to fix the channel error
            img = Image.open(image_path).convert('RGB')

            # OPTIONAL: Skip completely flat images (solid colors) to prevent negative outliers
            # A standard deviation of 0 means the image is one solid color
            if np.std(np.array(img)) < 5:
                print(f"Skipping {filename}: Image is nearly solid/flat (outlier).")
                continue

            score = obj.score(img)

            # --- OUTLIER FILTERING ---
            # We only keep scores between 0 and 100
            if 0 <= score <= 100:
                image_data.append((filename, score))
            else:
                print(f"Ignored {filename}: Score {score:.2f} is out of bounds (0-100).")

        except Exception as e:
            print(f"Error processing {filename}: {e}")

        if (idx + 1) % 10 == 0:
            print(f"Processed {idx + 1}/{len(files)}...")

    # Write CSV
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Filename", "BRISQUE_Score"])
        writer.writerows(image_data)

    print(f"Valid scores saved to: {output_csv}")

    # Sort: Lower score = Better quality
    image_data.sort(key=lambda x: x[1])

    # Get Top K
    top_k_images = image_data[:k]

    if not os.path.exists(top_k_folder):
        os.makedirs(top_k_folder)

    print(f"Copying top {k} valid images...")
    for filename, score in top_k_images:
        src_path = os.path.join(input_folder, filename)
        dst_path = os.path.join(top_k_folder, filename)
        shutil.copy2(src_path, dst_path)
        print(f"Copied: {filename} (Score: {score:.4f})")

    print("Processing Complete.")

# --- Configuration ---
if __name__ == "__main__":
    # Update these paths
    INPUT_DIR = "E:\chexpert\Processed_Data\\No_Finding"
    OUTPUT_CSV_FILE = "brisque_scores.csv"
    TOP_K_DESTINATION = "E:\\vil-data\\train\Domain 3\\No_Finding"
    K_VALUE = 1000  # Number of top images to extract

    process_images_for_brisque(INPUT_DIR, OUTPUT_CSV_FILE, TOP_K_DESTINATION, K_VALUE)
