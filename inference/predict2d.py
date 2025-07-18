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
        self.prediction_threshold = config['prediction_threshold'] # Define the best threshold here

    def predict_volume(self, volume):
        """
        Predicts segmentation for a full 3D volume by predicting each 2D slice,
        applying a post-prediction threshold.
        Returns both the predicted masks (after thresholding) and the corresponding MRI slices for visualization.
        """
        logger.info("Slicing volume for 2D prediction...")
        model_input_slices, viz_slices = self.slicer.slice_for_prediction(volume)
        
        predicted_slices_raw = [] # Store raw probability outputs
        logger.info(f"Predicting on {len(model_input_slices)} slices...")
        for s in tqdm(model_input_slices, desc="Predicting Slices"):
            # Add batch dimension
            s_expanded = np.expand_dims(s, axis=0)
            pred = self.model.predict(s_expanded, verbose=0)[0] # Get raw probabilities
            predicted_slices_raw.append(pred)

        logger.info("Applying best threshold and converting to class labels...")
        predicted_masks_thresholded = []
        for pred_raw_slice in predicted_slices_raw:
            # pred_raw_slice shape: (H, W, num_classes)
            
            # Get the predicted class for each pixel based on highest probability
            predicted_class = np.argmax(pred_raw_slice, axis=-1)
            
            # Get the maximum probability for the predicted class at each pixel
            max_probabilities = np.max(pred_raw_slice, axis=-1)
            
            # Apply threshold: if max_probability is below threshold AND it's a foreground class (not background)
            # then set it to background (class 0)
            # We assume class 0 is background.
            thresholded_mask = predicted_class.copy()
            # Only apply thresholding to foreground classes (classes > 0)
            # If the predicted class is foreground AND its max probability is below threshold, set to background (0)
            thresholded_mask[(thresholded_mask > 0) & (max_probabilities < self.prediction_threshold)] = 0 
            
            predicted_masks_thresholded.append(thresholded_mask)

        logger.info("Prediction on all slices complete.")
        # Return thresholded masks and MRI slices
        return predicted_masks_thresholded, viz_slices