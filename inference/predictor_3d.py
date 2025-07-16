import numpy as np
import tensorflow as tf
from tqdm import tqdm
from utils.logger import get_logger
from training.metrics import get_custom_objects

logger = get_logger(__name__)

class Predictor3D:
    def __init__(self, config, model_path):
        self.config = config
        self.model = tf.keras.models.load_model(
            model_path,
            custom_objects=get_custom_objects()
        )

    def predict_volume(self, volume):
        """
        Predicts segmentation for a full 3D volume.
        """
        logger.info("Predicting on full 3D volume...")
        # Add batch dimension
        volume_expanded = np.expand_dims(volume, axis=0)
        # Get prediction (it will have a batch dimension, so we remove it)
        prediction = self.model.predict(volume_expanded, verbose=0)[0]
        # Convert from one-hot encoding to class labels
        predicted_mask = np.argmax(prediction, axis=-1)
        logger.info("3D prediction complete.")
        return predicted_mask
