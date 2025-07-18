import matplotlib.pyplot as plt
from nilearn import plotting
import nibabel as nib
import numpy as np
import os
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import matplotlib as mpl # Import matplotlib for colormap utilities

from utils.helpers import ensure_dir
from utils.logger import get_logger

logger = get_logger(__name__)

class Visualizer:
    def __init__(self, config):
        self.config = config
        self.output_path = os.path.join(config['prediction_output_path'], 'visualizations')
        ensure_dir(self.output_path)
        
        # Custom colormap for 2D plots: BG (transparent), NCR (red), ED (green), ET (blue)
        self.cmap_2d = ListedColormap(['none', '#FF0000', '#00FF00', '#0000FF']) # Red for 1, Green for 2, Blue for 3
        self.legend_elements = [
            Patch(facecolor='#FF0000', edgecolor='k', label='Necrotic Core (Class 1)'),
            Patch(facecolor='#00FF00', edgecolor='k', label='Edema (Class 2)'),
            Patch(facecolor='#0000FF', edgecolor='k', label='Enhancing Tumor (Class 3)')
        ]

        # Define specific colors for 3D plotting to match 2D
        # Mapping class labels (1, 2, 3) to colors
        self.class_colors = {
            1: '#FF0000', # Necrotic Core (Red)
            2: '#00FF00', # Edema (Green)
            3: '#0000FF'  # Enhancing Tumor (Blue)
        }

        # New composite region colors and legend elements
        self.composite_colors = {
            'ET': '#0000FF', # Enhancing Tumor - Blue (same as individual ET for consistency)
            'TC': '#FF8C00', # Tumor Core (NCR + ET) - Orange
            'WT': '#8A2BE2'  # Whole Tumor (NCR + ED + ET) - Purple
        }
        self.composite_legend_elements = [
            Patch(facecolor=self.composite_colors['ET'], edgecolor='k', label='Enhancing Tumor (ET)'),
            Patch(facecolor=self.composite_colors['TC'], edgecolor='k', label='Tumor Core (TC)'),
            Patch(facecolor=self.composite_colors['WT'], edgecolor='k', label='Whole Tumor (WT)')
        ]


    def _create_composite_masks(self, segmentation_mask):
        """
        Creates composite masks for ET, TC, and WT from a raw segmentation mask.
        Assumes remapped labels: 1=Necrotic, 2=Edema, 3=Enhancing Tumor.
        """
        et_mask = (segmentation_mask == 3).astype(np.int16) # Enhancing Tumor (Original Label 4 -> remapped to 3)
        tc_mask = ((segmentation_mask == 1) | (segmentation_mask == 3)).astype(np.int16) # Tumor Core (Necrotic + Enhancing)
        wt_mask = ((segmentation_mask == 1) | (segmentation_mask == 2) | (segmentation_mask == 3)).astype(np.int16) # Whole Tumor (Necrotic + Edema + Enhancing)
        return et_mask, tc_mask, wt_mask


    def plot_slice_comparison(self, patient_id, mri_slice, gt_slice, pred_slice, slice_idx):
        """
        Generates and saves a detailed plot for a single 2D slice,
        showing the MRI, GT, Pred, and an overlay for individual tumor components.
        """
        fig, axes = plt.subplots(1, 4, figsize=(22, 5))
        fig.suptitle(f'Patient {patient_id} - Slice {slice_idx} - Individual Tumor Components', fontsize=16)

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

        # Transpose for correct orientation in matplotlib (assuming image data is (H, W))
        mri_slice_normalized = mri_slice_normalized.T
        gt_slice = gt_slice.T
        pred_slice = pred_slice.T

        # 1. MRI Background
        axes[0].imshow(mri_slice_normalized, cmap='gray', origin='lower')
        axes[0].set_title('MRI (FLAIR)')
        axes[0].axis('off')

        # 2. Ground Truth Segmentation
        axes[1].imshow(mri_slice_normalized, cmap='gray', origin='lower')
        axes[1].imshow(gt_slice, cmap=self.cmap_2d, alpha=0.6, vmin=0, vmax=self.config['num_classes']-1, origin='lower')
        axes[1].set_title('Ground Truth')
        axes[1].axis('off')

        # 3. Predicted Segmentation
        axes[2].imshow(mri_slice_normalized, cmap='gray', origin='lower')
        axes[2].imshow(pred_slice, cmap=self.cmap_2d, alpha=0.6, vmin=0, vmax=self.config['num_classes']-1, origin='lower')
        axes[2].set_title('Prediction')
        axes[2].axis('off')

        # 4. Overlay of GT and Pred
        axes[3].imshow(mri_slice_normalized, cmap='gray', origin='lower')
        # Plot GT first
        axes[3].imshow(gt_slice, cmap=self.cmap_2d, alpha=0.5, origin='lower')
        # Plot Pred second, potentially overlapping
        axes[3].imshow(pred_slice, cmap=self.cmap_2d, alpha=0.5, origin='lower')
        axes[3].set_title('GT & Pred Overlay')
        axes[3].axis('off')
        
        # Add a single legend for the entire figure at the bottom
        fig.legend(handles=self.legend_elements, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.05))
        
        patient_viz_dir = os.path.join(self.output_path, patient_id)
        ensure_dir(patient_viz_dir)
        fig_name = os.path.join(patient_viz_dir, f"slice_{slice_idx:03d}_individual_comparison.png")

        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to make space for suptitle and legend
        plt.savefig(fig_name, bbox_inches='tight')
        plt.close(fig)

    def plot_slice_composite_comparison(self, patient_id, mri_slice, gt_slice, pred_slice, slice_idx):
        """
        Generates and saves a detailed plot for a single 2D slice,
        showing the MRI, and overlays for ET, TC, WT composite regions.
        """
        # Create composite masks for GT and Prediction
        gt_et, gt_tc, gt_wt = self._create_composite_masks(gt_slice)
        pred_et, pred_tc, pred_wt = self._create_composite_masks(pred_slice)

        fig, axes = plt.subplots(2, 3, figsize=(18, 12)) # 2 rows (GT, Pred), 3 columns (ET, TC, WT)
        fig.suptitle(f'Patient {patient_id} - Slice {slice_idx} - Composite Tumor Regions', fontsize=16)

        # Safely normalize the MRI slice for visualization
        if mri_slice is not None:
            min_val = mri_slice.min()
            max_val = mri_slice.max()
            if max_val > min_val:
                mri_slice_normalized = (mri_slice - min_val) / (max_val - min_val)
            else:
                mri_slice_normalized = mri_slice
        else:
            mri_slice_normalized = np.zeros_like(gt_slice, dtype=float)

        # Transpose for correct orientation in matplotlib (assuming image data is (H, W))
        mri_slice_normalized = mri_slice_normalized.T

        composite_masks_gt = {'ET': gt_et.T, 'TC': gt_tc.T, 'WT': gt_wt.T}
        composite_masks_pred = {'ET': pred_et.T, 'TC': pred_tc.T, 'WT': pred_wt.T}

        titles = ['Enhancing Tumor (ET)', 'Tumor Core (TC)', 'Whole Tumor (WT)']
        composite_keys = ['ET', 'TC', 'WT']

        # Plot Ground Truth composite views
        for i, key in enumerate(composite_keys):
            axes[0, i].imshow(mri_slice_normalized, cmap='gray', origin='lower')
            axes[0, i].imshow(composite_masks_gt[key], cmap=ListedColormap(['none', self.composite_colors[key]]), alpha=0.6, origin='lower')
            axes[0, i].set_title(f'Ground Truth: {titles[i]}')
            axes[0, i].axis('off')

        # Plot Predicted composite views
        for i, key in enumerate(composite_keys):
            axes[1, i].imshow(mri_slice_normalized, cmap='gray', origin='lower')
            axes[1, i].imshow(composite_masks_pred[key], cmap=ListedColormap(['none', self.composite_colors[key]]), alpha=0.6, origin='lower')
            axes[1, i].set_title(f'Prediction: {titles[i]}')
            axes[1, i].axis('off')

        # Add a single legend for the entire figure at the bottom
        fig.legend(handles=self.composite_legend_elements, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.05))

        patient_viz_dir = os.path.join(self.output_path, patient_id)
        ensure_dir(patient_viz_dir)
        fig_name = os.path.join(patient_viz_dir, f"slice_{slice_idx:03d}_composite_comparison.png")

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(fig_name, bbox_inches='tight')
        plt.close(fig)


    def plot_3d_individual_reconstruction(self, patient_id, original_mri, ground_truth_seg, predicted_seg):
        """
        Generates and saves a comprehensive 3-view nilearn plot for the full 3D volume,
        with consistent color mapping for INDIVIDUAL tumor sub-regions (NCR, ED, ET).
        """
        affine = np.eye(4) # Identity affine matrix
        original_nii = nib.Nifti1Image(original_mri, affine)
        
        patient_viz_dir = os.path.join(self.output_path, patient_id)
        ensure_dir(patient_viz_dir)
        
        # Find a good slice to display by looking for the largest tumor area
        # If no tumor in GT, use the center of the volume
        if np.sum(ground_truth_seg) > 0:
            coords = np.argwhere(ground_truth_seg > 0)
            center_coords = coords.mean(axis=0)
        else:
            center_coords = np.array(predicted_seg.shape) / 2

        # --- Create the main plot ---
        # Use plot_anat for background and then add overlays for segmentation classes
        display_modes = ['z', 'x', 'y']
        view_titles = ['Axial', 'Coronal', 'Sagittal']
        
        # Create a figure for Ground Truth and Prediction side-by-side
        fig, axes = plt.subplots(2, 3, figsize=(18, 12)) # Increased height for better spacing
        fig.suptitle(f'3D Segmentation Comparison - Patient {patient_id} - Individual Components', fontsize=20)

        # Plot Ground Truth views
        for i, (mode, title) in enumerate(zip(display_modes, view_titles)):
            display_gt = plotting.plot_anat(
                original_nii,
                axes=axes[0, i],
                display_mode=mode,
                cut_coords=center_coords,
                title=f'Ground Truth ({title})',
                cmap='gray' # Background MRI in grayscale
            )
            # Add each class as an overlay with specific color
            for class_idx, color in self.class_colors.items():
                mask = (ground_truth_seg == class_idx).astype(np.int16)
                if np.any(mask): # Only add overlay if the class is present in the mask
                    display_gt.add_overlay(
                        nib.Nifti1Image(mask, affine),
                        cmap=ListedColormap([color]), # Use a single-color colormap
                        transparency=0.6, # Changed from alpha to transparency
                        vmin=0.5, # To ensure only 1s are colored
                        vmax=1.5
                    )
            display_gt.annotate(size=10) # Add coordinate annotations

        # Plot Prediction views
        for i, (mode, title) in enumerate(zip(display_modes, view_titles)):
            display_pred = plotting.plot_anat(
                original_nii,
                axes=axes[1, i],
                display_mode=mode,
                cut_coords=center_coords,
                title=f'Prediction ({title})',
                cmap='gray' # Background MRI in grayscale
            )
            # Add each class as an overlay with specific color
            for class_idx, color in self.class_colors.items():
                mask = (predicted_seg == class_idx).astype(np.int16)
                if np.any(mask): # Only add overlay if the class is present in the mask
                    display_pred.add_overlay(
                        nib.Nifti1Image(mask, affine),
                        cmap=ListedColormap([color]), # Use a single-color colormap
                        transparency=0.6, # Changed from alpha to transparency
                        vmin=0.5, # To ensure only 1s are colored
                        vmax=1.5
                    )
            display_pred.annotate(size=10) # Add coordinate annotations

        # Add a single legend for the entire figure at the bottom
        fig.legend(handles=self.legend_elements, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.05))
        
        fig_name = os.path.join(patient_viz_dir, f"{patient_id}_3d_views_individual_comparison.png")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to make space for suptitle and legend
        plt.savefig(fig_name, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"Saved 3D individual component comparison plot to {fig_name}")


    def plot_3d_composite_reconstruction(self, patient_id, original_mri, ground_truth_seg, predicted_seg):
        """
        Generates and saves a comprehensive 3-view nilearn plot for the full 3D volume,
        now focusing on composite tumor regions (ET, TC, WT).
        """
        affine = np.eye(4) # Identity affine matrix
        original_nii = nib.Nifti1Image(original_mri, affine)
        
        patient_viz_dir = os.path.join(self.output_path, patient_id)
        ensure_dir(patient_viz_dir)
        
        # Find a good slice to display by looking for the largest tumor area
        # If no tumor in GT, use the center of the volume
        if np.sum(ground_truth_seg) > 0:
            coords = np.argwhere(ground_truth_seg > 0)
            center_coords = coords.mean(axis=0)
        else:
            center_coords = np.array(predicted_seg.shape) / 2

        display_modes = ['z', 'x', 'y']
        view_titles = ['Axial', 'Coronal', 'Sagittal']
        composite_keys = ['ET', 'TC', 'WT']

        # Create composite masks for GT and Prediction
        gt_et_mask, gt_tc_mask, gt_wt_mask = self._create_composite_masks(ground_truth_seg)
        pred_et_mask, pred_tc_mask, pred_wt_mask = self._create_composite_masks(predicted_seg)

        gt_composite_volumes = {
            'ET': nib.Nifti1Image(gt_et_mask, affine),
            'TC': nib.Nifti1Image(gt_tc_mask, affine),
            'WT': nib.Nifti1Image(gt_wt_mask, affine)
        }
        pred_composite_volumes = {
            'ET': nib.Nifti1Image(pred_et_mask, affine),
            'TC': nib.Nifti1Image(pred_tc_mask, affine),
            'WT': nib.Nifti1Image(pred_wt_mask, affine)
        }

        # Plotting for each composite region (ET, TC, WT)
        for comp_key in composite_keys:
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle(f'3D Segmentation Comparison - Patient {patient_id} - {comp_key} Region', fontsize=20)

            # Plot Ground Truth views for the current composite region
            for i, (mode, title) in enumerate(zip(display_modes, view_titles)):
                display_gt = plotting.plot_anat(
                    original_nii,
                    axes=axes[0, i],
                    display_mode=mode,
                    cut_coords=center_coords,
                    title=f'Ground Truth ({title})',
                    cmap='gray'
                )
                display_gt.add_overlay(
                    gt_composite_volumes[comp_key],
                    cmap=ListedColormap([self.composite_colors[comp_key]]),
                    transparency=0.6,
                    vmin=0.5, vmax=1.5 # Ensures only 1s are colored
                )
                display_gt.annotate(size=10)

            # Plot Prediction views for the current composite region
            for i, (mode, title) in enumerate(zip(display_modes, view_titles)):
                display_pred = plotting.plot_anat(
                    original_nii,
                    axes=axes[1, i],
                    display_mode=mode,
                    cut_coords=center_coords,
                    title=f'Prediction ({title})',
                    cmap='gray'
                )
                display_pred.add_overlay(
                    pred_composite_volumes[comp_key],
                    cmap=ListedColormap([self.composite_colors[comp_key]]),
                    transparency=0.6,
                    vmin=0.5, vmax=1.5
                )
                display_pred.annotate(size=10)

            # Add a single legend for the current composite region
            comp_patch = Patch(facecolor=self.composite_colors[comp_key], edgecolor='k', label=comp_key)
            fig.legend(handles=[comp_patch], loc='lower center', ncol=1, bbox_to_anchor=(0.5, -0.05))
            
            fig_name = os.path.join(patient_viz_dir, f"{patient_id}_3d_views_composite_{comp_key}.png")
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.savefig(fig_name, bbox_inches='tight')
            plt.close(fig)
            
            logger.info(f"Saved 3D composite plot for {comp_key} to {fig_name}")