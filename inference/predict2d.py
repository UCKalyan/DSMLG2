import numpy as np
import tensorflow as tf
from tqdm import tqdm
from data.slicer import Slicer
from utils.logger import get_logger
from utils.helpers import load_npy
from training.metrics import get_custom_objects

logger = get_logger(__name__)

class Predictor2D:
    def __init__(self, config, model_path='unet2d_best.keras'):
        self.config = config
        self.model = tf.keras.models.load_model(
            model_path,
            custom_objects=get_custom_objects()
        )
        self.slicer = Slicer(axis=config['slice_axis'])

    def predict_volume(self, volume):
        """
        Predicts segmentation for a full 3D volume by predicting each 2D slice.
        Returns both the predicted masks and the corresponding MRI slices for visualization.
        """
        logger.info("Slicing volume for 2D prediction...")
        # Use the new slicer method to get both model inputs and viz slices
        model_input_slices, viz_slices = self.slicer.slice_for_prediction(volume)
        
        predicted_slices = []
        logger.info(f"Predicting on {len(model_input_slices)} slices...")
        for s in tqdm(model_input_slices, desc="Predicting Slices"):
            # Add batch dimension
            s_expanded = np.expand_dims(s, axis=0)
            pred = self.model.predict(s_expanded, verbose=0)[0]
            predicted_slices.append(pred)

        logger.info("Prediction on all slices complete.")
        # Return both lists
        return predicted_slices, viz_slices
