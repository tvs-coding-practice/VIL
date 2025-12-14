import os
import shutil
import pandas as pd
import concurrent.futures
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

# --- CONFIGURATION ---
INPUT_BASE_DIR = "E:\chexpert"
OUTPUT_DIR = "E:\chexpert\Processed_Data"

# The 12 pathologies (Excluding 'No Finding' and 'Support Devices')
DISEASE_LABELS = [
    'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
    'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia',
    'Atelectasis', 'Pneumothorax', 'Pleural Effusion',
    'Pleural Other', 'Fracture'
]

# --- PREPARATION ---
if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)

for label in ['Cardiomegaly', 'Pneumothorax']:
    os.makedirs(os.path.join(OUTPUT_DIR, label), exist_ok=True)

# Load Data
print("Loading dataset...")
df = pd.read_csv(os.path.join(INPUT_BASE_DIR, "train.csv"))
initial_count = len(df)

# --- NEW STRICT FILTERING LOGIC ---

# 1. Drop records where AP/PA is Empty/NaN
df_clean = df.dropna(subset=['AP/PA']).copy()
print(f"Dropped {initial_count - len(df_clean)} rows due to missing AP/PA.")

# 2. Drop records where ANY disease cell has -1.0 (Uncertain)
# Check if any of the 12 disease columns have a -1.0 value
has_uncertainty = df_clean[DISEASE_LABELS].eq(-1.0).any(axis=1)
df_clean = df_clean[~has_uncertainty].copy()
print(f"Dropped {has_uncertainty.sum()} rows containing uncertain (-1.0) disease labels.")

# 3. Calculate Disease Count (Positives only)
df_clean['Disease_Count'] = df_clean[DISEASE_LABELS].eq(1.0).sum(axis=1)

# 4. Filter: Exact Single Label
single_finding_df = df_clean[df_clean['Disease_Count'] == 1].copy()

# 5. Filter: Target Classes (Cardiomegaly OR Pneumothorax)
target_df = single_finding_df[
    (single_finding_df['Cardiomegaly'] == 1.0) |
    (single_finding_df['Pneumothorax'] == 1.0)
    ].copy()


# 6. Assign Label
def get_label(row):
    if row['Cardiomegaly'] == 1.0: return 'Cardiomegaly'
    if row['Pneumothorax'] == 1.0: return 'Pneumothorax'
    return None


target_df['Label'] = target_df.apply(get_label, axis=1)
target_df.dropna(subset=['Label'], inplace=True)

print(f"\nFinal count for processing: {len(target_df)} images.")


# --- PROCESSING FUNCTION ---
def process_row(row):
    # Path setup
    try:
        relative_path_start = row['Path'].split('CheXpert-v1.0-small/')[-1]
    except AttributeError:
        return None  # Handle malformed paths

    src_path = os.path.join(INPUT_BASE_DIR, relative_path_start)

    if not os.path.exists(src_path):
        return None

    label = row['Label']
    new_filename = relative_path_start.replace('/', '_')
    dst_path = os.path.join(OUTPUT_DIR, label, new_filename)

    try:
        shutil.copy(src_path, dst_path)

        # Preserve original record and add new metadata
        full_record = row.to_dict()
        full_record.update({
            'Processed_Filename': new_filename,
            'Processed_Label': label,
            'Processed_Path': dst_path
        })
        return full_record

    except Exception:
        return None


# --- EXECUTION ---
metadata_list = []
skipped_count = 0

print("Starting file copy...")
with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as executor:
    futures = [executor.submit(process_row, row) for _, row in target_df.iterrows()]

    for i, future in enumerate(concurrent.futures.as_completed(futures)):
        result = future.result()
        if result:
            metadata_list.append(result)
        else:
            skipped_count += 1

        if (i + 1) % 100 == 0:
            print(f"Progress: {i + 1}/{len(target_df)}", end='\r')

# --- SAVE RESULTS ---
if metadata_list:
    meta_df = pd.DataFrame(metadata_list)
    save_path = os.path.join(OUTPUT_DIR, "processed_metadata_full.csv")
    meta_df.to_csv(save_path, index=False)

    print(f"\n\nDone. Successfully saved {len(meta_df)} images.")
    print(f"Full metadata with original columns saved to: {save_path}")
    print("\nClass Distribution:")
    print(meta_df['Label'].value_counts())
else:
    print("\nNo images matched all criteria.")