import numpy as np

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

        Returns:
            A list of 2D slices. If segmentation is provided, returns a list of
            (slice, seg_slice) tuples.
        """
        slices = []
        num_slices = volume.shape[self.axis]

        for i in range(num_slices):
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
        return slices
