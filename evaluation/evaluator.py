import os
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import pandas as pd

from data.dataset_loader import BratsDataset2D, BratsDataset3D
from training.metrics import get_custom_objects
from utils.helpers import ensure_dir
from utils.logger import get_logger

logger = get_logger(__name__)

class Evaluator:
    def __init__(self, config):
        self.config = config
        self.output_path = config['evaluation_output_path']
        ensure_dir(self.output_path)

    def evaluate_2d_model(self, model_path):
        """
        Evaluates a 2D UNET model on the test set.
        """
        logger.info("Starting 2D model evaluation on the test set...")
        model = tf.keras.models.load_model(
            model_path,
            custom_objects=get_custom_objects()
        )

        test_loader = BratsDataset2D(self.config, mode='test')
        test_dataset = test_loader.get_dataset(self.config['batch_size'])
        
        # Estimate steps for the test set since it's not repeated
        avg_slices_per_patient = 64
        test_steps = int(np.ceil((test_loader.dataset_size * avg_slices_per_patient) / self.config['batch_size']))
        
        logger.info(f"Evaluating on {test_loader.dataset_size} test patients, with approx. {test_steps} steps.")

        results = model.evaluate(test_dataset, steps=test_steps, verbose=1)
        
        # Manually define metric names to match the order in the trainer's compile step.
        # This is a robust way to handle cases where model.metrics_names is not reliable.
        metrics_names = [
            'loss', 'dice_coef', 'iou', 'precision', 'sensitivity', 'specificity',
            'dice_coef_necrotic', 'dice_coef_edema', 'dice_coef_enhancing'
        ]
        
        # Ensure the number of results matches the number of names
        if len(results) != len(metrics_names):
            logger.error(f"Mismatch between number of results ({len(results)}) and metric names ({len(metrics_names)}).")
            logger.error("Please check the metrics in trainer2d.py and update the list in evaluator.py.")
            return

        results_df = pd.DataFrame([results], columns=metrics_names)
        
        # Save results to a CSV file
        results_path = os.path.join(self.output_path, 'unet2d_test_results.csv')
        results_df.to_csv(results_path, index=False)
        logger.info(f"Evaluation results saved to {results_path}")
        logger.info(f"Test Results:\n{results_df.to_string()}")

    def evaluate_3d_model(self, model_path):
        """
        Evaluates a 3D model on the test set.
        """
        logger.info("Starting 3D model evaluation on the test set...")
        model = tf.keras.models.load_model(
            model_path,
            custom_objects=get_custom_objects()
        )

        test_loader = BratsDataset3D(self.config, mode='test')
        test_dataset = test_loader.get_dataset(self.config['batch_size'])

        logger.info(f"Evaluating on {test_loader.dataset_size} test patients.")

        results = model.evaluate(test_dataset, verbose=1)
        
        # Manually define metric names for the 3D segmentation model
        metrics_names = ['loss', 'dice_coef'] 
        
        if len(results) != len(metrics_names):
             logger.error(f"Mismatch between number of results ({len(results)}) and metric names ({len(metrics_names)}).")
             logger.error("Please check the metrics in trainer3d_seg.py and update the list in evaluator.py.")
             return

        results_df = pd.DataFrame([results], columns=metrics_names)
        
        results_path = os.path.join(self.output_path, 'unet3d_test_results.csv')
        results_df.to_csv(results_path, index=False)
        logger.info(f"Evaluation results saved to {results_path}")
        logger.info(f"Test Results:\n{results_df.to_string()}")
