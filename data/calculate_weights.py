import os
import numpy as np
import nibabel as nib
from tqdm import tqdm


def calculate_brats_class_weights(data_dir):
    """
    Calculates class weights for the BraTS dataset using Median Frequency Balancing.

    Args:
        data_dir (str): The path to the directory containing the ground truth
                        label files (e.g., 'BraTS2020_TrainingData/').
                        The script will search for files ending in '_seg.nii.gz'.

    Returns:
        A numpy array containing the calculated class weights.
    """
    print("Searching for segmentation files...")
    # Find all segmentation files in the training data directory
    patient_ids = [p for p in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, p))]
    for patient_id in tqdm(patient_ids, desc="Preprocessing Patients"):
        patient_dir = os.path.join(data_dir, patient_id)
        print('patient_dir : ', patient_dir)
        #print('os.path.join(root, f) for root, _, files in os.walk(patient_dir) ',[ os.path.join(root, f) for root, _, files in os.walk(patient_dir) ])
        seg_files = [os.path.join(root, f) for root, _, files in os.walk(patient_dir) 
                    for f in files if f.endswith('_seg.nii')]
        
        if not seg_files:
            raise ValueError(f"No segmentation files ('_seg.nii') found in {data_dir}")

    print(f"Found {len(seg_files)} segmentation files. Starting analysis...")

    # Initialize a counter for all 4 classes (0, 1, 2, 3)
    # Class 0: Background
    # Class 1: NCR/NET (Necrotic and Non-Enhancing Tumor)
    # Class 2: ED (Peritumoral Edema)
    # Class 3: ET (Enhancing Tumor)
    total_counts = np.zeros(4, dtype=np.int64)

    # Iterate over each segmentation file with a progress bar
    for seg_file in tqdm(seg_files, desc="Processing Files"):
        # Load the segmentation mask
        seg_mask = nib.load(seg_file).get_fdata().astype(np.uint8)
        seg_mask[seg_mask == 4] = 3
        
        # Get unique values and their counts
        unique, counts = np.unique(seg_mask, return_counts=True)
        
        # Add the counts to the total
        for i, class_label in enumerate(unique):
            if class_label < 4: # Ensure we only count the 4 valid classes
                total_counts[class_label] += counts[i]

    print("\nTotal Voxel Counts per Class:")
    print(f"  Background (0): {total_counts[0]:,}")
    print(f"  NCR/NET (1):    {total_counts[1]:,}")
    print(f"  Edema (2):        {total_counts[2]:,}")
    print(f"  Enhancing (3):  {total_counts[3]:,}")
    
    # --- Median Frequency Balancing Calculation ---
    print("\nCalculating weights with Median Frequency Balancing...")
    
    # Calculate class frequencies
    total_voxels = np.sum(total_counts)
    class_frequencies = total_counts / total_voxels
    
    # Get the median frequency
    median_freq = np.median(class_frequencies)
    
    # Calculate weights as median_frequency / frequency
    class_weights = median_freq / class_frequencies
    
    # Normalize the weights to sum to the number of classes for stability
    class_weights /= np.sum(class_weights)
    class_weights *= len(total_counts)


    print("\nCalculated Class Weights:")
    print(f"  Weight BG: {class_weights[0]:.4f}")
    print(f"  Weight NCR/NET: {class_weights[1]:.4f}")
    print(f"  Weight ED: {class_weights[2]:.4f}")
    print(f"  Weight ET: {class_weights[3]:.4f}")
    
    return class_weights.tolist()


if __name__ == '__main__':
    # --------------------------------------------------------------------------
    # IMPORTANT: Update this path to your BraTS training data directory.
    # The script expects the BraTS folder structure where label files are named
    # like 'BraTS20_Training_001_seg.nii.gz' inside their respective folders.
    # --------------------------------------------------------------------------

    BRATS_TRAINING_DATA_PATH = '/Users/kalyan/Documents/Software/segmentation-project/data/processed/BraTS2020_Train_Test/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/' 
    
    if not os.path.isdir(BRATS_TRAINING_DATA_PATH):
        print(f"Error: Directory not found at '{BRATS_TRAINING_DATA_PATH}'")
        print("Please update the 'BRATS_TRAINING_DATA_PATH' variable in the script.")
    else:
        final_weights = calculate_brats_class_weights(BRATS_TRAINING_DATA_PATH)
        print("\nFinal weights list to use in your loss function:")
        print(final_weights)