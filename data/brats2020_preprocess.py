import os
import nibabel as nib
import numpy as np
from scipy.ndimage import zoom
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import SimpleITK as sitk

from utils.helpers import ensure_dir, save_npy
from utils.logger import get_logger

logger = get_logger(__name__)

class Preprocessor:
    """
    Handles preprocessing of BraTS 2020 MRI volumes.
    """
    def __init__(self, config):
        """
        Initializes the preprocessor with settings from the config.
        """
        self.config = config
        self.input_path = config['data_path']
        self.output_path = config['processed_data_path']
        self.target_shape = tuple(config['volume_shape'][:3])
        
        # Set floating point precision from config, defaulting to float32
        dtype_str = config.get('dtype', 'float32')
        self.dtype = np.float32 if dtype_str == 'float32' else np.float64
        logger.info(f"Using floating point precision: {self.dtype}")
        
        ensure_dir(self.output_path)

    def _skull_strip(self, image_itk, mask_itk):
        """
        Applies skull stripping using the provided brain mask.
        """
        mask_array = sitk.GetArrayFromImage(mask_itk)
        image_array = sitk.GetArrayFromImage(image_itk)
        image_array[mask_array == 0] = 0
        return sitk.GetImageFromArray(image_array)

    def _normalize(self, volume_data):
        """
        Applies z-score normalization to the volume.
        """
        scaler = StandardScaler()
        # Flatten all but the channel dimension
        flat_data = volume_data.reshape(-1, volume_data.shape[-1])
        # Fit and transform
        normalized_flat = scaler.fit_transform(flat_data)
        # Reshape back to original
        return normalized_flat.reshape(volume_data.shape)


    def _resample(self, volume, original_shape, is_mask=False):
        """
        Resamples a volume to the target shape.
        """
        # We only want to resample the spatial dimensions (D, H, W)
        spatial_original_shape = original_shape[:3]
        zoom_factors = [t / o for t, o in zip(self.target_shape, spatial_original_shape)]
        
        # If the volume has a channel dimension, add a zoom factor of 1 for it.
        if volume.ndim == 4:
            zoom_factors.append(1)
            
        # Set interpolation order: 0 for masks (nearest neighbor), 1 for images (linear)
        order = 0 if is_mask else 1
        
        resampled_volume = zoom(volume, zoom_factors, order=order, prefilter=False)
        return resampled_volume

    def process_patient(self, patient_id):
        """
        Processes all modalities for a single patient.
        """
        try:
            patient_dir = os.path.join(self.input_path, patient_id)
            modalities = ['t1', 't1ce', 't2', 'flair']
            volume_channels = []
            first_modality_path = None

            for mod in modalities:
                # Construct possible file paths for both .nii and .nii.gz
                path_nii = os.path.join(patient_dir, f'{patient_id}_{mod}.nii')
                path_niigz = os.path.join(patient_dir, f'{patient_id}_{mod}.nii.gz')

                if os.path.exists(path_nii):
                    img_path = path_nii
                elif os.path.exists(path_niigz):
                    img_path = path_niigz
                else:
                    logger.error(f"Could not find file for patient {patient_id}, modality {mod}. Skipping patient.")
                    return False # Skip this patient

                if first_modality_path is None:
                    first_modality_path = img_path

                img = nib.load(img_path)
                # Cast to configured dtype to manage memory usage
                volume_channels.append(img.get_fdata(dtype=self.dtype))

            # Stack modalities to create a 4-channel volume
            volume = np.stack(volume_channels, axis=-1)
            original_shape = volume.shape

            # Resampling
            resampled_volume = self._resample(volume, original_shape)

            # Normalization
            normalized_volume = self._normalize(resampled_volume)

            # --- Segmentation mask ---
            seg_path_nii = os.path.join(patient_dir, f'{patient_id}_seg.nii')
            seg_path_niigz = os.path.join(patient_dir, f'{patient_id}_seg.nii.gz')
            seg_data = None

            if os.path.exists(seg_path_nii):
                seg_img = nib.load(seg_path_nii)
                seg_data = seg_img.get_fdata()
            elif os.path.exists(seg_path_niigz):
                seg_img = nib.load(seg_path_niigz)
                seg_data = seg_img.get_fdata()
            else:
                logger.warning(f"No segmentation file found for patient {patient_id}. Creating empty mask.")
                # Get shape from the first loaded modality to ensure it matches
                base_shape = nib.load(first_modality_path).shape
                seg_data = np.zeros(base_shape)

            resampled_seg = self._resample(seg_data, seg_data.shape, is_mask=True).astype(np.uint8)

            # --- Remap BraTS labels (0, 1, 2, 4) to (0, 1, 2, 3) ---
            remapped_seg = np.copy(resampled_seg)
            remapped_seg[remapped_seg == 4] = 3
            # This is the critical fix for the IndexError

            # Generate binary label (example logic: presence of tumor = malignant)
            is_malignant = 1 if np.sum(remapped_seg) > 0 else 0
            label = np.array([is_malignant], dtype=self.dtype)

            # Save processed data
            patient_out_dir = os.path.join(self.output_path, patient_id)
            ensure_dir(patient_out_dir)
            save_npy(normalized_volume, os.path.join(patient_out_dir, 'volume.npy'))
            save_npy(remapped_seg, os.path.join(patient_out_dir, 'segmentation.npy'))
            save_npy(label, os.path.join(patient_out_dir, 'label.npy'))

            logger.info(f"Successfully processed patient {patient_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to process patient {patient_id}: {e}")
            return False

    def run(self):
        """
        Runs the preprocessing pipeline for all patients.
        """
        patient_ids = [p for p in os.listdir(self.input_path) if os.path.isdir(os.path.join(self.input_path, p))]
        for patient_id in tqdm(patient_ids, desc="Preprocessing Patients"):
            self.process_patient(patient_id)
