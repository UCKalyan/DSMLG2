import matplotlib.pyplot as plt
from nilearn import plotting
import nibabel as nib
import numpy as np
import os
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from utils.helpers import ensure_dir
from utils.logger import get_logger

logger = get_logger(__name__)

class Visualizer:
    def __init__(self, config):
        self.config = config
        self.output_path = os.path.join(config['prediction_output_path'], 'visualizations')
        ensure_dir(self.output_path)
        # Custom colormap: BG (transparent), NCR (red), ED (green), ET (blue)
        self.cmap = ListedColormap(['none', '#FF0000', '#00FF00', '#0000FF'])
        self.legend_elements = [
            Patch(facecolor='#FF0000', edgecolor='k', label='Necrotic Core'),
            Patch(facecolor='#00FF00', edgecolor='k', label='Edema'),
            Patch(facecolor='#0000FF', edgecolor='k', label='Enhancing Tumor')
        ]


    def plot_slice_comparison(self, patient_id, mri_slice, gt_slice, pred_slice, slice_idx):
        """
        Generates and saves a detailed plot for a single 2D slice,
        showing the MRI, GT, Pred, and an overlay.
        """
        fig, axes = plt.subplots(1, 4, figsize=(22, 5))
        fig.suptitle(f'Patient {patient_id} - Slice {slice_idx}', fontsize=16)

        # Safely normalize the MRI slice for visualization
        if mri_slice is not None:
            min_val = mri_slice.min()
            max_val = mri_slice.max()
            # Avoid division by zero for black slices
            if max_val > min_val:
                mri_slice_normalized = (mri_slice - min_val) / (max_val - min_val)
            else:
                mri_slice_normalized = mri_slice
        else:
            # Create a black image if mri_slice is None for some reason
            mri_slice_normalized = np.zeros_like(gt_slice, dtype=float)

        # Transpose for correct orientation in matplotlib
        mri_slice_normalized = mri_slice_normalized.T
        gt_slice = gt_slice.T
        pred_slice = pred_slice.T

        # 1. MRI Background
        axes[0].imshow(mri_slice_normalized, cmap='gray', origin='lower')
        axes[0].set_title('MRI (FLAIR)')
        axes[0].axis('off')

        # 2. Ground Truth Segmentation
        axes[1].imshow(mri_slice_normalized, cmap='gray', origin='lower')
        axes[1].imshow(gt_slice, cmap=self.cmap, alpha=0.6, vmin=0, vmax=self.config['num_classes']-1, origin='lower')
        axes[1].set_title('Ground Truth')
        axes[1].axis('off')

        # 3. Predicted Segmentation
        axes[2].imshow(mri_slice_normalized, cmap='gray', origin='lower')
        axes[2].imshow(pred_slice, cmap=self.cmap, alpha=0.6, vmin=0, vmax=self.config['num_classes']-1, origin='lower')
        axes[2].set_title('Prediction')
        axes[2].axis('off')

        # 4. Overlay of GT and Pred
        axes[3].imshow(mri_slice_normalized, cmap='gray', origin='lower')
        axes[3].imshow(gt_slice, cmap=self.cmap, alpha=0.5, origin='lower')
        axes[3].imshow(pred_slice, cmap=self.cmap, alpha=0.5, origin='lower')
        axes[3].set_title('GT & Pred Overlay')
        axes[3].axis('off')
        
        fig.legend(handles=self.legend_elements, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.05))
        
        patient_viz_dir = os.path.join(self.output_path, patient_id)
        ensure_dir(patient_viz_dir)
        fig_name = os.path.join(patient_viz_dir, f"slice_{slice_idx:03d}_comparison.png")

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(fig_name, bbox_inches='tight')
        plt.close(fig)

    def plot_3d_reconstruction(self, patient_id, original_mri, ground_truth_seg, predicted_seg):
        """
        Generates and saves a comprehensive 3-view nilearn plot for the full 3D volume.
        """
        affine = np.eye(4)
        original_nii = nib.Nifti1Image(original_mri, affine)
        gt_nii = nib.Nifti1Image(ground_truth_seg.astype(np.int16), affine)
        pred_nii = nib.Nifti1Image(predicted_seg.astype(np.int16), affine)

        patient_viz_dir = os.path.join(self.output_path, patient_id)
        ensure_dir(patient_viz_dir)
        
        # Find a good slice to display by looking for the largest tumor area
        if np.sum(ground_truth_seg) > 0:
            coords = np.argwhere(ground_truth_seg > 0)
            center_coords = coords.mean(axis=0)
        else:
            center_coords = np.array(predicted_seg.shape) / 2

        # --- Create the main plot ---
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(f'3D Segmentation Comparison - Patient {patient_id}', fontsize=20)
        
        display_modes = ['z', 'x', 'y']
        view_titles = ['Axial', 'Coronal', 'Sagittal']
        
        # Plot Ground Truth views
        for i, (mode, title) in enumerate(zip(display_modes, view_titles)):
            plotting.plot_roi(
                gt_nii,
                bg_img=original_nii,
                axes=axes[0, i],
                display_mode=mode,
                cut_coords=center_coords,
                title=f'Ground Truth ({title})',
                cmap='viridis'
            )

        # Plot Prediction views
        for i, (mode, title) in enumerate(zip(display_modes, view_titles)):
            plotting.plot_roi(
                pred_nii,
                bg_img=original_nii,
                axes=axes[1, i],
                display_mode=mode,
                cut_coords=center_coords,
                title=f'Prediction ({title})',
                cmap='plasma'
            )
        
        fig_name = os.path.join(patient_viz_dir, f"{patient_id}_3d_views_comparison.png")
        plt.savefig(fig_name, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"Saved 3D comparison plot to {fig_name}")
