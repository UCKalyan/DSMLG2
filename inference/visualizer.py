import matplotlib.pyplot as plt
from nilearn import plotting
import nibabel as nib
import numpy as np
import os
from matplotlib.colors import ListedColormap

from utils.helpers import ensure_dir

class Visualizer:
    def __init__(self, config):
        self.config = config
        self.output_path = os.path.join(config['prediction_output_path'], 'visualizations')
        ensure_dir(self.output_path)
        # Custom colormap: BG (transparent), NCR (red), ED (green), ET (blue)
        self.cmap = ListedColormap(['none', 'red', 'green', 'blue'])

    def plot_slice_comparison(self, patient_id, mri_slice, gt_slice, pred_slice, slice_idx):
        """
        Generates and saves a detailed plot for a single 2D slice,
        showing the MRI, GT, Pred, and an overlay.
        """
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        fig.suptitle(f'Patient {patient_id} - Slice {slice_idx}', fontsize=16)

        # 1. MRI Background
        axes[0].imshow(mri_slice, cmap='gray')
        axes[0].set_title('MRI (FLAIR)')
        axes[0].axis('off')

        # 2. Ground Truth Segmentation
        axes[1].imshow(mri_slice, cmap='gray')
        axes[1].imshow(gt_slice, cmap=self.cmap, alpha=0.6)
        axes[1].set_title('Ground Truth')
        axes[1].axis('off')

        # 3. Predicted Segmentation
        axes[2].imshow(mri_slice, cmap='gray')
        axes[2].imshow(pred_slice, cmap=self.cmap, alpha=0.6)
        axes[2].set_title('Prediction')
        axes[2].axis('off')

        # 4. Overlay of GT and Pred
        axes[3].imshow(mri_slice, cmap='gray')
        # Make GT slightly transparent to see prediction underneath
        axes[3].imshow(gt_slice, cmap=self.cmap, alpha=0.5, label='GT')
        axes[3].imshow(pred_slice, cmap=self.cmap, alpha=0.5, label='Pred')
        axes[3].set_title('GT & Pred Overlay')
        axes[3].axis('off')
        
        # Create output directory for the patient if it doesn't exist
        patient_viz_dir = os.path.join(self.output_path, patient_id)
        ensure_dir(patient_viz_dir)
        fig_name = os.path.join(patient_viz_dir, f"slice_{slice_idx:03d}_comparison.png")

        plt.tight_layout()
        plt.savefig(fig_name)
        plt.close(fig)

    def plot_3d_reconstruction(self, patient_id, original_mri, ground_truth_seg, predicted_seg):
        """
        Generates and saves nilearn plots for the full 3D volume.
        """
        affine = np.eye(4)
        original_nii = nib.Nifti1Image(original_mri, affine)
        gt_nii = nib.Nifti1Image(ground_truth_seg.astype(np.int16), affine)
        pred_nii = nib.Nifti1Image(predicted_seg.astype(np.int16), affine)

        patient_viz_dir = os.path.join(self.output_path, patient_id)
        ensure_dir(patient_viz_dir)
        fig_name = os.path.join(patient_viz_dir, f"{patient_id}_3d_reconstruction.png")

        # Find a good slice to display by looking for the largest tumor area
        slice_idx = np.argmax(np.sum(ground_truth_seg, axis=(0, 1)))

        plotting.plot_roi(
            pred_nii,
            bg_img=original_nii,
            title=f'3D Prediction Overlay - Patient {patient_id}',
            display_mode='z',
            cut_coords=[slice_idx],
            output_file=fig_name,
        )
