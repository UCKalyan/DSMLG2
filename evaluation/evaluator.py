import os
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import pandas as pd

from data.dataset_loader import BratsDataset2D, BratsDataset3D
from training.metrics import (dice_coef, iou, precision, recall, sensitivity, specificity,
                              dice_coef_necrotic, dice_coef_edema, dice_coef_enhancing)
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
        model = tf.keras.models.load_model(model_path, custom_objects={
            'combined_loss': lambda y_true, y_pred: 0,
            'dice_coef': dice_coef,
            'iou': iou,
            'precision': precision,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'dice_coef_necrotic': dice_coef_necrotic,
            'dice_coef_edema': dice_coef_edema,
            'dice_coef_enhancing': dice_coef_enhancing
        })

        test_loader = BratsDataset2D(self.config, mode='test')
        test_dataset = test_loader.get_dataset(self.config['batch_size'])
        
        logger.info(f"Evaluating on {test_loader.dataset_size} test patients.")

        results = model.evaluate(test_dataset, steps=test_loader.dataset_size, verbose=1)
        
        metrics_names = model.metrics_names
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
        # Add custom objects as needed for the 3D model
        model = tf.keras.models.load_model(model_path, custom_objects={
             'combined_loss': lambda y_true, y_pred: 0,
             'dice_coef': dice_coef
        })

        test_loader = BratsDataset3D(self.config, mode='test')
        test_dataset = test_loader.get_dataset(self.config['batch_size'])

        logger.info(f"Evaluating on {test_loader.dataset_size} test patients.")

        results = model.evaluate(test_dataset, verbose=1)
        
        metrics_names = model.metrics_names
        results_df = pd.DataFrame([results], columns=metrics_names)
        
        results_path = os.path.join(self.output_path, 'unet3d_test_results.csv')
        results_df.to_csv(results_path, index=False)
        logger.info(f"Evaluation results saved to {results_path}")
        logger.info(f"Test Results:\n{results_df.to_string()}")

