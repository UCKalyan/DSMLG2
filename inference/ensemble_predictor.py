# inference/ensemble_predictor.py

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
            pass
        else:
            raise ValueError("model_type must be '2d' or '3d' for EnsemblePredictor.")

    def _load_models(self):
        logger.info(f"Loading {len(self.model_paths)} models for ensemble...")
        for path in self.model_paths:
            try:
                model = tf.keras.models.load_model(
                    path,
                    custom_objects=get_custom_objects()
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
        num_models = len(self.models)
        if num_models == 0:
            logger.error("No models loaded for ensemble prediction.")
            return None

        logger.info(f"Starting ensemble prediction for a volume using {num_models} models...")

        if self.config['model_type'] == '2d':
            model_input_slices, _ = self.slicer.slice_for_prediction(volume)
            
            ensemble_2d_probs_per_slice = []

            for slice_idx, s in enumerate(tqdm(model_input_slices, desc="Ensembling 2D Slices")):
                s_expanded = np.expand_dims(s, axis=0)

                slice_probs_sum = None
                for model in self.models:
                    pred_probs = model.predict(s_expanded, verbose=0)[0]
                    if slice_probs_sum is None:
                        slice_probs_sum = pred_probs
                    else:
                        slice_probs_sum += pred_probs
                
                ensemble_2d_probs_per_slice.append(slice_probs_sum)
            
            averaged_2d_probs_per_slice = [p / num_models for p in ensemble_2d_probs_per_slice]

            original_seg_shape = volume.shape[:-1]
            
            # Reconstruct the probability volume
            reconstructed_prob_volume = self.reconstructor_3d.stack_slices(
                averaged_2d_probs_per_slice, original_seg_shape
            )

            # --- START OF FIX ---
            # Convert the 4D probability volume to a 3D label map
            ensemble_pred_3d_labels = np.argmax(reconstructed_prob_volume, axis=-1)
            # --- END OF FIX ---

            # Apply post-processing to the final 3D label map
            final_ensemble_mask = self.reconstructor_3d.post_process(ensemble_pred_3d_labels)

        elif self.config['model_type'] == '3d':
            ensemble_predictions_sum = None
            volume_expanded = np.expand_dims(volume, axis=0)
            
            for model in self.models:
                pred_probs = model.predict(volume_expanded, verbose=0)[0]
                if ensemble_predictions_sum is None:
                    ensemble_predictions_sum = pred_probs
                else:
                    ensemble_predictions_sum += pred_probs
            
            averaged_probs_3d = ensemble_predictions_sum / num_models
            
            final_ensemble_mask = np.argmax(averaged_probs_3d, axis=-1)
            final_ensemble_mask = self.reconstructor_3d.post_process(final_ensemble_mask)
            
        logger.info("Ensemble prediction complete.")
        return final_ensemble_mask
