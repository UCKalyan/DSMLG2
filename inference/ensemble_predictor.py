# ensemble_predictor.py

import os
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from training.metrics import get_custom_objects # To load models correctly
from data.slicer import Slicer
from inference.reconstruct3d import Reconstructor3D
from utils.logger import get_logger
from utils.helpers import load_npy

logger = get_logger(__name__)

class EnsemblePredictor:
    def __init__(self, config, model_paths):
        self.config = config
        self.model_paths = model_paths
        self.models = []
        self._load_models()

        if config['model_type'] == '2d':
            self.slicer = Slicer(axis=config['slice_axis'])
            self.reconstructor_3d = Reconstructor3D(config)
        elif config['model_type'] == '3d':
            # No slicer/reconstructor needed if models are already 3D
            pass
        else:
            raise ValueError("model_type must be '2d' or '3d' for EnsemblePredictor.")

    def _load_models(self):
        logger.info(f"Loading {len(self.model_paths)} models for ensemble...")
        for path in self.model_paths:
            try:
                model = tf.keras.models.load_model(
                    path,
                    custom_objects=get_custom_objects() # Use your defined custom objects
                )
                self.models.append(model)
                logger.info(f"Successfully loaded model from: {path}")
            except Exception as e:
                logger.error(f"Error loading model from {path}: {e}")
                raise

    def predict_volume(self, volume):
        """
        Generates an ensemble prediction for a single 3D volume.
        Combines predictions from all loaded models by averaging their probability maps.
        """
        ensemble_predictions_sum = None
        num_models = len(self.models)

        if num_models == 0:
            logger.error("No models loaded for ensemble prediction.")
            return None

        logger.info(f"Starting ensemble prediction for a volume using {num_models} models...")

        if self.config['model_type'] == '2d':
            # For 2D models, slice, predict, and then accumulate 2D probabilities
            model_input_slices, _ = self.slicer.slice_for_prediction(volume)
            
            # Placeholder for summed 2D probability slices from all models
            ensemble_2d_probs_per_slice = [] # List of (H, W, NumClasses) summed probs

            for slice_idx, s in enumerate(tqdm(model_input_slices, desc="Ensembling 2D Slices")):
                s_expanded = np.expand_dims(s, axis=0) # Add batch dim

                slice_probs_sum = None
                for model in self.models:
                    pred_probs = model.predict(s_expanded, verbose=0)[0] # (H, W, NumClasses)
                    if slice_probs_sum is None:
                        slice_probs_sum = pred_probs
                    else:
                        slice_probs_sum += pred_probs
                
                ensemble_2d_probs_per_slice.append(slice_probs_sum)
            
            # Average the summed 2D probabilities
            averaged_2d_probs_per_slice = [p / num_models for p in ensemble_2d_probs_per_slice]

            # Reconstruct to a 3D volume (still probabilities)
            # The reconstructor needs the original 3D segmentation shape to stack correctly.
            # Assuming volume.shape is (D, H, W, Channels)
            original_seg_shape = volume.shape[:-1] # (D, H, W)
            
            # Pass probabilities to stack_slices, it will argmax at the end
            ensemble_pred_3d_labels = self.reconstructor_3d.stack_slices(
                averaged_2d_probs_per_slice, original_seg_shape
            )
            # Apply post-processing to the final hard labels
            final_ensemble_mask = self.reconstructor_3d.post_process(ensemble_pred_3d_labels)

        elif self.config['model_type'] == '3d':
            # For 3D models, predict on the full volume and average probabilities
            volume_expanded = np.expand_dims(volume, axis=0) # Add batch dim
            
            for model in self.models:
                pred_probs = model.predict(volume_expanded, verbose=0)[0] # (D, H, W, NumClasses)
                if ensemble_predictions_sum is None:
                    ensemble_predictions_sum = pred_probs
                else:
                    ensemble_predictions_sum += pred_probs
            
            averaged_probs_3d = ensemble_predictions_sum / num_models
            
            # Convert averaged probabilities to hard labels
            final_ensemble_mask = np.argmax(averaged_probs_3d, axis=-1)
            # Apply post-processing if desired for 3D models too
            final_ensemble_mask = self.reconstructor_3d.post_process(final_ensemble_mask) # Re-use reconstructor for post-processing
            
        logger.info("Ensemble prediction complete.")
        return final_ensemble_mask # This is a 3D volume with class labels (D, H, W)