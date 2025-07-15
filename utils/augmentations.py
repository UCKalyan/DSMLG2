import tensorflow as tf
import numpy as np

class Augmenter:
    """
    Applies real-time data augmentation.
    Handles both 2D slices (rank 3) and 3D volumes (rank 4).
    """
    def __init__(self, flip=True, rotate=True, intensity_shift=True):
        self.flip = flip
        self.rotate = rotate
        self.intensity_shift = intensity_shift

    @tf.function
    def augment(self, volume, label):
        """
        Applies a series of augmentations to a volume and its corresponding label.
        """
        volume = tf.cast(volume, tf.float32)
        is_segmentation = label.shape.ndims == volume.shape.ndims

        # --- Intensity Shift ---
        if self.intensity_shift and tf.random.uniform(()) > 0.5:
            shift = tf.random.uniform((), -0.1, 0.1)
            scale = tf.random.uniform((), 0.9, 1.1)
            volume = (volume + shift) * scale

        # --- Flipping ---
        if self.flip:
            # Flip on axis 0 (Height for 2D, Depth for 3D)
            if tf.random.uniform(()) > 0.5:
                volume = tf.reverse(volume, axis=[0])
                if is_segmentation: label = tf.reverse(label, axis=[0])
            # Flip on axis 1 (Width for 2D, Height for 3D)
            if tf.random.uniform(()) > 0.5:
                volume = tf.reverse(volume, axis=[1])
                if is_segmentation: label = tf.reverse(label, axis=[1])
            # Flip on axis 2 (Width for 3D only)
            if volume.shape.ndims == 4 and tf.random.uniform(()) > 0.5:
                volume = tf.reverse(volume, axis=[2])
                if is_segmentation: label = tf.reverse(label, axis=[2])

        # --- Rotation ---
        if self.rotate and tf.random.uniform(()) > 0.5:
            k = tf.random.uniform(shape=[], minval=1, maxval=4, dtype=tf.int32)
            # For 2D (H, W, C), rotate in the H,W plane
            if volume.shape.ndims == 3:
                volume = tf.image.rot90(volume, k=k)
                if is_segmentation: label = tf.image.rot90(label, k=k)
            # For 3D (D, H, W, C), rotate in the H,W plane (axial rotation)
            elif volume.shape.ndims == 4:
                # Transpose D,H,W,C -> H,W,D,C to rotate the axial plane
                volume = tf.transpose(volume, perm=[1, 2, 0, 3])
                volume = tf.image.rot90(volume, k=k)
                # Transpose back H,W,D,C -> D,H,W,C
                volume = tf.transpose(volume, perm=[2, 0, 1, 3])
                if is_segmentation:
                    label = tf.transpose(label, perm=[1, 2, 0, 3])
                    label = tf.image.rot90(label, k=k)
                    label = tf.transpose(label, perm=[2, 0, 1, 3])

        return volume, label

def get_augmenter(config):
    """Factory function to create an Augmenter instance from config."""
    aug_config = config.get('augmentations', {})
    return Augmenter(
        flip=aug_config.get('flip', True),
        rotate=aug_config.get('rotate', True),
        intensity_shift=aug_config.get('intensity_shift', True)
    )
