import os
import csv
from datasets import load_dataset
from tqdm import tqdm



# Loading BrachioLab/chestx in streaming mode...
# Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
# Resolving data files: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 35/35 [00:00<?, ?it/s]
# Resolving data files: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 35/35 [00:00<?, ?it/s]
# Starting processing...
# Metadata will be saved to: processed_chestx_images\metadata.csv
# 23094it [1:37:25,  3.95it/s]
#
# --- Processing Complete ---
# Metadata saved at: processed_chestx_images\metadata.csv
# Saved Effusion: 328 images
# Saved Infiltration: 1871 images
# Saved Nodule: 699 images

def process_chestx_dataset():
    # --- CONFIGURATION ---
    dataset_name = "BrachioLab/chestx"
    base_output_dir = "processed_chestx_images"
    metadata_file = os.path.join(base_output_dir, "metadata.csv")

    # Target Diseases (Index mapping)
    target_diseases = {
        4: "Effusion",
        8: "Infiltration",
        10: "Nodule"
    }

    # --- SETUP ---
    # 1. Create base directory and sub-folders
    if not os.path.exists(base_output_dir):
        os.makedirs(base_output_dir)

    for disease in target_diseases.values():
        dir_path = os.path.join(base_output_dir, disease)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    # 2. Initialize Metadata CSV with headers
    # We open in 'w' mode to overwrite any previous run's file
    with open(metadata_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Dataset_Index", "Image_Filename", "Disease_Label", "Pathols_Vector"])

    print(f"Loading {dataset_name} in streaming mode...")
    dataset = load_dataset(dataset_name, split="train", streaming=True)

    print("Starting processing...")
    print(f"Metadata will be saved to: {metadata_file}")

    saved_counts = {k: 0 for k in target_diseases.values()}

    # --- PROCESSING ---
    # Open CSV in 'append' mode so we can write row by row
    with open(metadata_file, mode='a', newline='') as f:
        csv_writer = csv.writer(f)

        for i, sample in tqdm(enumerate(dataset)):
            try:
                pathols = sample.get('pathols')

                # Skip invalid rows
                if not pathols: continue

                # STRICT CONDITION: Sum of bits must be exactly 1
                if sum(pathols) == 1:
                    active_index = pathols.index(1)

                    # Check if the active disease is one we want
                    if active_index in target_diseases:
                        disease_name = target_diseases[active_index]

                        # --- FILENAME LOGIC ---
                        image = sample['image']

                        # Try to extract original name from PIL object, else use Index
                        if hasattr(image, 'filename') and image.filename:
                            fname = os.path.basename(image.filename)
                        else:
                            fname = f"image_{i}.png"

                        # 1. Save the Image
                        save_path = os.path.join(base_output_dir, disease_name, fname)
                        image.save(save_path)

                        # 2. Write to Metadata CSV
                        # We convert pathols list to a string so it fits in one CSV cell
                        csv_writer.writerow([i, fname, disease_name, str(pathols)])

                        saved_counts[disease_name] += 1

            except Exception as e:
                print(f"Error processing row {i}: {e}")
                continue

    # --- SUMMARY ---
    print("\n--- Processing Complete ---")
    print(f"Metadata saved at: {metadata_file}")
    for disease, count in saved_counts.items():
        print(f"Saved {disease}: {count} images")


if __name__ == "__main__":
    process_chestx_dataset()