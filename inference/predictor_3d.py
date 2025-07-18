import numpy as np
import tensorflow as tf
from tqdm import tqdm
from utils.logger import get_logger
from training.metrics import get_custom_objects
from scipy.ndimage import zoom # Import zoom for 3D resizing

logger = get_logger(__name__)

class Predictor3D:
    def __init__(self, config, model_path):
        self.config = config
        self.model = tf.keras.models.load_model(
            model_path,
            custom_objects=get_custom_objects()
        )
        # Get the expected input shape from the model's first input layer
        # This assumes the input shape is defined in the model's configuration
        # e.g., (64, 64, 64, 4)
        self.target_input_shape = self.model.input_shape[1:] # Exclude batch dimension
        logger.info(f"Expected input shape for 3D model: {self.target_input_shape}")


    def predict_volume(self, volume):
        """
        Predicts segmentation for a full 3D volume, resizing it to match the model's
        expected input shape if necessary.
        """
        logger.info("Predicting on full 3D volume...")

        original_shape = volume.shape
        logger.info(f"Original volume shape: {original_shape}")

        # Check if resizing is necessary
        if original_shape != self.target_input_shape:
            logger.info(f"Resizing volume from {original_shape} to {self.target_input_shape} for 3D model prediction.")
            
            # Calculate zoom factors for each dimension (excluding channels)
            zoom_factors = [
                target_dim / original_dim
                for target_dim, original_dim in zip(self.target_input_shape[:-1], original_shape[:-1])
            ]
            # Add zoom factor for the channel dimension (which should be 1, as we don't resize channels)
            zoom_factors.append(1.0) 

            # Resize the volume using scipy.ndimage.zoom
            # order=1 for linear interpolation (suitable for medical images)
            resized_volume = zoom(volume, zoom_factors, order=1, mode='nearest') 
            volume_to_predict = resized_volume
        else:
            volume_to_predict = volume

        # Add batch dimension
        volume_expanded = np.expand_dims(volume_to_predict, axis=0)
        
        # Get prediction (it will have a batch dimension, so we remove it)
        prediction = self.model.predict(volume_expanded, verbose=0)[0]

        # If the input was resized, the prediction will also be in the resized shape.
        # We need to resize the prediction back to the original volume's shape
        # before converting to class labels, if resizing occurred.
        if original_shape != self.target_input_shape:
            logger.info(f"Resizing prediction back from {prediction.shape} to {original_shape} for original context.")
            
            # Prediction shape will be (target_dim_x, target_dim_y, target_dim_z, num_classes)
            # Original shape of prediction should be (original_dim_x, original_dim_y, original_dim_z, num_classes)
            prediction_zoom_factors = [
                original_dim / target_dim
                for original_dim, target_dim in zip(original_shape[:-1], self.target_input_shape[:-1])
            ]
            prediction_zoom_factors.append(1.0) # For num_classes

            # For prediction (segmentation masks), it's often better to use order=0 (nearest neighbor)
            # after argmax for categorical data, or keep it one-hot and then resize.
            # Here, we resize the probabilities before argmax, which is generally more accurate.
            resized_prediction = zoom(prediction, prediction_zoom_factors, order=1, mode='nearest')
            prediction = resized_prediction
            
        # Convert from one-hot encoding to class labels after potential resizing
        predicted_mask = np.argmax(prediction, axis=-1)
        
        logger.info("3D prediction complete.")
        return predicted_mask