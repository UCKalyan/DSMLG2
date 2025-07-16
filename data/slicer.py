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
            if np.sum(vol_slice) > 0:
                if segmentation is not None:
                    slices.append((vol_slice, seg_slice))
                else:
                    slices.append(vol_slice)
        
        logger.debug(f"[SLICER] Returning {len(slices)} non-empty slices")
        return slices

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
