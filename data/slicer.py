import numpy as np
from utils.logger import get_logger

logger = get_logger("Slicer")

class Slicer:
    """
    Utility to slice 3D volumes into 2D slices along a specified axis.
    """
    def __init__(self, axis='z'):
        if axis not in ['x', 'y', 'z']:
            raise ValueError("Axis must be one of 'x', 'y', or 'z'")
        self.axis_map = {'x': 0, 'y': 1, 'z': 2}
        self.axis = self.axis_map[axis]

    def slice_volume(self, volume, segmentation=None):
        """
        Slices a 3D volume (and optionally its segmentation mask) into 2D slices.
        This method is primarily for generating training data.
        """
        slices = []
        # Keep track of both tumor and non-tumor slices
        tumor_slices = []
        healthy_slices = []
        num_total_slices = volume.shape[self.axis]
        logger.debug(f"[SLICER] Slicing along axis={self.axis} with {num_total_slices} total slices")


        for i in range(num_total_slices):
            if self.axis == 0:
                vol_slice = volume[i, :, :, :]
                seg_slice = segmentation[i, :, :] if segmentation is not None else None
            elif self.axis == 1:
                vol_slice = volume[:, i, :, :]
                seg_slice = segmentation[:, i, :] if segmentation is not None else None
            else: # axis == 2
                vol_slice = volume[:, :, i, :]
                seg_slice = segmentation[:, :, i] if segmentation is not None else None

            # Filter out empty slices (optional, but good for training)
            # Ensure the slice has content AND the segmentation mask is not empty
            if np.sum(vol_slice) > 0 and segmentation is not None and np.sum(seg_slice) > 0:
                slices.append((vol_slice, seg_slice))
            else:
                # Optionally, you could still append slices without tumors but with a
                # much lower probability to help the model learn what "healthy" tissue looks like.
                # For now, skipping them is the most direct way to focus the training.
                pass
        # Check if the slice has brain tissue but no tumor
            has_brain = np.sum(vol_slice) > 0
            has_tumor = segmentation is not None and np.sum(seg_slice) > 0

            if has_brain:
                if has_tumor:
                    tumor_slices.append((vol_slice, seg_slice))
                else:
                    # Only append healthy slices if a segmentation mask was provided
                    if segmentation is not None:
                        healthy_slices.append((vol_slice, seg_slice))
        
        #logger.debug(f"[SLICER] Returning {len(slices)} non-empty slices")
        #return slices
        # --- Strategy: Combine all tumor slices with a fraction of healthy slices ---
        num_healthy_to_keep = int(len(tumor_slices) * 0.25) # e.g., keep 25% as many healthy slices as tumor slices
        np.random.shuffle(healthy_slices) # Shuffle to get a random sample
        final_slices = tumor_slices + healthy_slices[:num_healthy_to_keep]

        logger.debug(f"[SLICER] Returning {len(tumor_slices)} tumor slices and {len(healthy_slices[:num_healthy_to_keep])} healthy slices. Total: {len(final_slices)}")
        return final_slices

    def slice_for_prediction(self, volume):
        """
        Slices a 3D volume for prediction, returning both the model input slices
        and the corresponding visualization slices.
        """
        model_input_slices = []
        viz_slices = []
        num_total_slices = volume.shape[self.axis]

        for i in range(num_total_slices):
            if self.axis == 0:
                vol_slice = volume[i, :, :, :]
            elif self.axis == 1:
                vol_slice = volume[:, i, :, :]
            else: # axis == 2
                vol_slice = volume[:, :, i, :]
            
            model_input_slices.append(vol_slice)
            # Assuming FLAIR is the last channel (index 3) and is best for visualization
            viz_slices.append(vol_slice[:, :, 3])
            
        return model_input_slices, viz_slices
