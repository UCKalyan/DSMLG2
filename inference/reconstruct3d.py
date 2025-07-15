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
        Stacks 2D slices back into a 3D volume.
        """
        num_classes = self.config['num_classes']
        # We need to handle the fact that we might have filtered some slices.
        # This is a simplified reconstruction assuming all slices are present.
        # A more robust implementation would map slice indices back to their original positions.
        
        # For simplicity, we assume slices are in order and stack them.
        stacked = np.stack(slices, axis=self.axis)
        
        # Convert from one-hot back to class labels
        reconstructed_vol = np.argmax(stacked, axis=-1)
        
        return reconstructed_vol

    def post_process(self, volume):
        """
        Applies 3D morphological post-processing.
        """
        # Process each class label separately (except background)
        processed_vol = np.zeros_like(volume)
        for class_idx in range(1, self.config['num_classes']):
            class_mask = (volume == class_idx)
            
            # Remove small disconnected components
            # min_size can be tuned based on expected tumor size
            cleaned_mask = remove_small_objects(class_mask, min_size=100)
            
            # Fill holes
            filled_mask = binary_fill_holes(cleaned_mask)
            
            processed_vol[filled_mask] = class_idx
            
        return processed_vol
