import numpy as np
import tensorflow as tf
from tqdm import tqdm
from utils.logger import get_logger
from training.metrics import get_custom_objects # Assuming custom objects are needed for classifier too

logger = get_logger(__name__)

class Predictor3DClassifier:
    """
    Predictor class for a 3D classification model.
    """
    def __init__(self, config, model_path='classifier3d_best.keras'):
        self.config = config
        self.model = tf.keras.models.load_model(
            model_path,
            custom_objects=get_custom_objects() # Ensure custom objects are relevant for classifier
        )
        logger.info(f"3D Classifier model loaded from: {model_path}")

    def predict_volume(self, volume):
        """
        Predicts the class for a single 3D volume.
        
        Args:
            volume (np.ndarray): The input 3D MRI volume (e.g., (D, H, W, C)).
                                 Expected to be preprocessed and ready for model input.

        Returns:
            np.ndarray: Predicted probabilities for each class (e.g., (num_classes,)).
        """
        # Ensure volume has the correct data type as specified in config
        volume = volume.astype(self.config['dtype'])

        if volume.ndim == 4: # Assuming input is (D, H, W, C)
            # Add batch dimension
            volume_expanded = np.expand_dims(volume, axis=0)
        elif volume.ndim == 5: # Already has batch dimension (B, D, H, W, C)
            volume_expanded = volume
        else:
            logger.error(f"Unexpected volume dimensions: {volume.ndim}. Expected 4 or 5.")
            return None

        logger.info(f"Predicting class for volume with shape: {volume_expanded.shape}")
        
        # Perform prediction
        predictions = self.model.predict(volume_expanded, verbose=0)[0] # Get rid of batch dim
        
        logger.info("Prediction complete.")
        return predictions

