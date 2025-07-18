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
        self.target_input_shape = self.model.input_shape[1:] # Exclude batch dimension
        self.prediction_threshold = config['prediction_threshold'] # Define the best threshold here


    def predict_volume(self, volume):
        """
        Predicts segmentation for a full 3D volume, resizing it to match the model's
        expected input shape if necessary, and applying a post-prediction threshold.
        """
        logger.info("Predicting on full 3D volume...")

        original_shape = volume.shape
        volume_to_predict = volume

        # Check if resizing is necessary for input
        if original_shape[:-1] != self.target_input_shape[:-1]: # Compare spatial dimensions only
            logger.info(f"Resizing volume from {original_shape} to {self.target_input_shape} for 3D model prediction.")
            zoom_factors = [
                target_dim / original_dim
                for target_dim, original_dim in zip(self.target_input_shape[:-1], original_shape[:-1])
            ]
            zoom_factors.append(1.0) # For channel dimension
            resized_volume = zoom(volume, zoom_factors, order=1, mode='nearest') 
            volume_to_predict = resized_volume

        # Add batch dimension
        volume_expanded = np.expand_dims(volume_to_predict, axis=0)
        
        # Get raw prediction probabilities
        prediction_raw = self.model.predict(volume_expanded, verbose=0)[0]

        # If the input was resized, the prediction will also be in the resized shape.
        # We need to resize the prediction probabilities back to the original volume's shape
        if original_shape[:-1] != self.target_input_shape[:-1]:
            logger.info(f"Resizing prediction back from {prediction_raw.shape} to {original_shape} for original context.")
            prediction_zoom_factors = [
                original_dim / target_dim
                for original_dim, target_dim in zip(original_shape[:-1], self.target_input_shape[:-1])
            ]
            prediction_zoom_factors.append(1.0) # For num_classes
            resized_prediction_raw = zoom(prediction_raw, prediction_zoom_factors, order=1, mode='nearest')
            prediction_raw = resized_prediction_raw
            
        logger.info("Applying best threshold and converting to class labels...")
        # Convert from one-hot encoding to class labels
        predicted_class_labels = np.argmax(prediction_raw, axis=-1) # Get highest probability class
        
        # Get the maximum probability across all classes for each voxel
        max_probabilities = np.max(prediction_raw, axis=-1)
        
        # Apply threshold: if max_probability is below threshold AND it's a foreground class (not background)
        # then set it to background (class 0). We assume class 0 is background.
        predicted_mask = predicted_class_labels.copy()
        predicted_mask[(predicted_mask > 0) & (max_probabilities < self.prediction_threshold)] = 0 
        
        logger.info("3D prediction complete.")
        return predicted_mask