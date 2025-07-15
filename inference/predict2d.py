import numpy as np
import tensorflow as tf
from data.slicer import Slicer
from utils.logger import get_logger
from utils.helpers import load_npy
from training.metrics import combined_loss, dice_coef

logger = get_logger(__name__)

class Predictor2D:
    def __init__(self, config, model_path='unet2d_best.keras'):
        self.config = config
        self.model = tf.keras.models.load_model(model_path, custom_objects={
            'combined_loss': combined_loss,
            'dice_coef': dice_coef
        })
        self.slicer = Slicer(axis=config['slice_axis'])

    def predict_volume(self, volume):
            """
            Predicts segmentation for a full 3D volume by predicting each 2D slice.
            Returns both the predicted slices and the original MRI slices.
            """
            logger.info("Slicing volume for 2D prediction...")
            # Store the original MRI slices that are created
            original_slices = self.slicer.slice_volume(volume)
            predicted_slices = []

            # Iterate through the original slices to make predictions
            for s in original_slices:
                # Add batch dimension
                s_expanded = np.expand_dims(s, axis=0)
                pred = self.model.predict(s_expanded)[0]
                predicted_slices.append(pred)

            logger.info("Prediction on all slices complete.")
            # Return both the list of predictions and the list of original slices
            return predicted_slices, original_slices
