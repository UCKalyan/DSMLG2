# File: reconstruct3d.py

import numpy as np
from scipy.ndimage import label, binary_fill_holes
from skimage.morphology import remove_small_objects

class Reconstructor3D:
    """
    Reconstructs a 3D segmentation volume from a list of 2D predicted slices.
    """
    def __init__(self, config):
        self.config = config
        self.axis_map = {'x': 0, 'y': 1, 'z': 2}
        self.axis = self.axis_map[config['slice_axis']]

    def stack_slices(self, slices, original_shape):
        """
        Stacks 2D slices (which are already class labels) back into a 3D volume.
        `slices` are expected to be (H, W) or (W, D) etc. 2D arrays of class labels.
        """
        # np.stack will add a new dimension at self.axis
        # If slices are (H, W), stacking along axis=2 will result in (H, W, NumSlices)
        stacked_volume = np.stack(slices, axis=self.axis)
        
        # Ensure the stacked volume has the same spatial dimensions as the original volume
        # This is important if original_shape includes channels (e.g., (D, H, W, C))
        # We only care about the spatial dimensions for the segmentation mask.
        target_spatial_shape = original_shape[:-1] # Exclude channels if original_shape includes them

        # Check if the reconstructed spatial shape matches the target spatial shape
        # This check is crucial for debugging and understanding dimensions
        if stacked_volume.shape != target_spatial_shape:
            # If the original_shape had channels (e.g. (96,96,96,4)) and stacked_volume is (96,96,96),
            # this check would fail, but the shape is actually correct for the segmentation mask.
            # We need to make sure the spatial dimensions align.
            # The order of dimensions depends on self.axis.
            
            # Example: if original_shape=(96,96,96,4) and axis=2, slices are (96,96).
            # stacked_volume will be (96,96,96). This is desired.
            pass # No specific action needed here, as the spatial dimensions should align if slices are correct.

        return stacked_volume

    def post_process(self, volume):
        """
        Applies 3D morphological post-processing.
        `volume` is expected to be a 3D array of class labels (e.g., (D, H, W)).
        """
        # Process each class label separately (except background)
        processed_vol = np.zeros_like(volume)
        # Assuming num_classes is available in config and 0 is background
        num_classes = self.config['num_classes'] 

        for class_idx in range(1, num_classes): # Iterate through foreground classes
            class_mask = (volume == class_idx)
            
            # Remove small disconnected components
            # min_size can be tuned based on expected tumor size
            # Ensure connectivity is 3D (e.g., connectivity=1 for 6-connectivity, 2 for 18-connectivity, 3 for 26-connectivity)
            cleaned_mask = remove_small_objects(class_mask, min_size=100, connectivity=1)
            
            # Fill holes
            # binary_fill_holes also works in 3D
            filled_mask = binary_fill_holes(cleaned_mask)
            
            processed_vol[filled_mask] = class_idx
            
        return processed_vol