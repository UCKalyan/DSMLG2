import os
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix, classification_report

from data.dataset_loader import BratsDataset2D, BratsDataset3D
from training.metrics import get_custom_objects, dice_coef # Import dice_coef
from utils.helpers import ensure_dir
from utils.logger import get_logger

logger = get_logger(__name__)

class Evaluator:
    def __init__(self, config):
        self.config = config
        self.output_path = config['evaluation_output_path']
        ensure_dir(self.output_path)
        # Load DSC thresholds from config, or use a default if not specified
        self.thresholds = self.config.get('dsc_thresholds', [0.0, 0.25, 0.50, 0.75, 1.0])


    def evaluate_2d_model(self, model_path):
        """
        Evaluates a 2D UNET model on the test set, including Dice coefficients at different thresholds.
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

        # Perform standard evaluation first
        standard_results = model.evaluate(test_dataset, steps=test_steps, verbose=1)
        
        metrics_names = [
            'loss', 'dice_coef', 'iou', 'precision', 'sensitivity', 'specificity',
            'dice_coef_necrotic', 'dice_coef_edema', 'dice_coef_enhancing'
        ]
        
        if len(standard_results) != len(metrics_names):
            logger.error(f"Mismatch between number of results ({len(standard_results)}) and metric names ({len(metrics_names)}).")
            logger.error("Please check the metrics in trainer2d.py and update the list in evaluator.py.")
            return

        # --- Calculate Dice at specified thresholds ---
        # Use thresholds loaded from config
        dice_at_thresholds = {f'dice_coef_thresh_{int(t*100)}': [] for t in self.thresholds}

        logger.info("Calculating Dice coefficients at different thresholds...")
        # Reset the dataset iterator for prediction
        # Using .unbatch().batch(1) to process samples individually for prediction collection
        test_dataset_for_pred = test_loader.get_dataset(self.config['batch_size']).unbatch().batch(1)
        
        all_y_true = []
        all_y_pred = []

        # Get all true labels and predictions
        for x_batch, y_true_batch in tqdm(test_dataset_for_pred, desc="Collecting Predictions"):
            y_pred_batch = model.predict(x_batch, verbose=0)
            all_y_true.append(y_true_batch)
            all_y_pred.append(y_pred_batch)

        # Concatenate all batches into single tensors
        y_true_all = tf.concat(all_y_true, axis=0)
        y_pred_all = tf.concat(all_y_pred, axis=0)

        for t in self.thresholds: # Use self.thresholds
            # Binarize predictions for the whole tumor (excluding background, index 0)
            # Only consider foreground classes (indices 1, 2, 3) for thresholding
            y_pred_binarized_foreground = tf.where(y_pred_all[..., 1:] >= t, 1.0, 0.0)
            
            # Reconstruct to original one-hot shape by adding a background channel of zeros
            bg_channel = tf.zeros_like(y_pred_all[..., 0:1])
            y_pred_binarized_full = tf.concat([bg_channel, y_pred_binarized_foreground], axis=-1)

            # Calculate Dice for the binarized predictions
            # The dice_coef function is designed to work with one-hot like inputs (0/1 for foreground)
            current_dice = dice_coef(y_true_all, y_pred_binarized_full).numpy()
            dice_at_thresholds[f'dice_coef_thresh_{int(t*100)}'].append(current_dice)

        # Average the Dice scores for each threshold (over all slices/samples)
        for key in dice_at_thresholds:
            dice_at_thresholds[key] = np.mean(dice_at_thresholds[key])
            
        # Combine standard results with thresholded Dice results
        combined_results = standard_results + list(dice_at_thresholds.values())
        combined_metrics_names = metrics_names + list(dice_at_thresholds.keys())

        results_df = pd.DataFrame([combined_results], columns=combined_metrics_names)
        
        # Save results to a CSV file
        results_path = os.path.join(self.output_path, 'unet2d_test_results.csv')
        results_df.to_csv(results_path, index=False)
        logger.info(f"Evaluation results saved to {results_path}")

        # Improved formatted output for logger
        logger.info("\n--- Detailed 2D Model Evaluation Results ---")
        for i, name in enumerate(combined_metrics_names):
            logger.info(f"  {name:<25}: {combined_results[i]:.4f}") # Added left-alignment and 4 decimal places
        logger.info("------------------------------------------")


    def evaluate_3d_model(self, model_path):
        """
        Evaluates a 3D model on the test set, including Dice coefficients at different thresholds.
        """
        logger.info("Starting 3D model evaluation on the test set...")
        model = tf.keras.models.load_model(
            model_path,
            custom_objects=get_custom_objects()
        )

        test_loader = BratsDataset3D(self.config, mode='test')
        test_dataset = test_loader.get_dataset(self.config['batch_size'])

        logger.info(f"Evaluating on {test_loader.dataset_size} test patients.")

        # Perform standard evaluation first
        standard_results = model.evaluate(test_dataset, verbose=1)
        
        # Updated metrics_names for 3D model to match the 9 results returned by model.evaluate
        metrics_names = [
            'loss', 'dice_coef', 'iou', 'precision', 'sensitivity', 'specificity',
            'dice_coef_necrotic', 'dice_coef_edema', 'dice_coef_enhancing'
        ]
        
        if len(standard_results) != len(metrics_names):
             logger.error(f"Mismatch between number of results ({len(standard_results)}) and metric names ({len(metrics_names)}).")
             logger.error("Please check the metrics in trainer3d_seg.py and update the list in evaluator.py.")
             return

        # --- Calculate Dice at specified thresholds for 3D model ---
        dice_at_thresholds = {f'dice_coef_thresh_{int(t*100)}': [] for t in self.thresholds}

        logger.info("Calculating Dice coefficients at different thresholds for 3D model...")
        # Reset the dataset iterator for prediction
        # Using .unbatch().batch(1) to process samples individually for prediction collection
        test_dataset_for_pred = test_loader.get_dataset(self.config['batch_size']).unbatch().batch(1)
        
        all_y_true = []
        all_y_pred = []

        # Get all true labels and predictions
        for x_batch, y_true_batch in tqdm(test_dataset_for_pred, desc="Collecting 3D Predictions"):
            y_pred_batch = model.predict(x_batch, verbose=0)
            all_y_true.append(y_true_batch)
            all_y_pred.append(y_pred_batch)

        # Concatenate all batches into single tensors
        y_true_all = tf.concat(all_y_true, axis=0)
        y_pred_all = tf.concat(all_y_pred, axis=0)

        for t in self.thresholds: # Use self.thresholds
            # Binarize predictions for the whole tumor (excluding background, index 0)
            # Only consider foreground classes (indices 1, 2, 3) for thresholding
            y_pred_binarized_foreground = tf.where(y_pred_all[..., 1:] >= t, 1.0, 0.0)
            
            # Reconstruct to original one-hot shape by adding a background channel of zeros
            bg_channel = tf.zeros_like(y_pred_all[..., 0:1])
            y_pred_binarized_full = tf.concat([bg_channel, y_pred_binarized_foreground], axis=-1)

            # Calculate Dice for the binarized predictions
            current_dice = dice_coef(y_true_all, y_pred_binarized_full).numpy()
            dice_at_thresholds[f'dice_coef_thresh_{int(t*100)}'].append(current_dice)

        # Average the Dice scores for each threshold (over all volumes/samples)
        for key in dice_at_thresholds:
            dice_at_thresholds[key] = np.mean(dice_at_thresholds[key])
            
        # Combine standard results with thresholded Dice results
        combined_results = standard_results + list(dice_at_thresholds.values())
        combined_metrics_names = metrics_names + list(dice_at_thresholds.keys())

        results_df = pd.DataFrame([combined_results], columns=combined_metrics_names)
        
        results_path = os.path.join(self.output_path, 'unet3d_test_results.csv')
        results_df.to_csv(results_path, index=False)
        logger.info(f"Evaluation results saved to {results_path}")

        # Improved formatted output for logger
        logger.info("\n--- Detailed 3D Model Evaluation Results ---")
        for i, name in enumerate(combined_metrics_names):
            logger.info(f"  {name:<25}: {combined_results[i]:.4f}") # Added left-alignment and 4 decimal places
        logger.info("------------------------------------------")

    def evaluate_3d_classifier_model(self, model_path):
        """
        Evaluates a 3D classification model on the test set.
        """
        logger.info("Starting 3D classifier model evaluation on the test set...")
        model = tf.keras.models.load_model(
            model_path,
            custom_objects=get_custom_objects()
        )

        # Create a temporary config for the loader to ensure it yields classification labels
        eval_config = self.config.copy()
        eval_config['output_type'] = 'benign_vs_malignant'
        logger.info("Overriding data loader to yield classification labels for evaluation.")
        
        test_loader = BratsDataset3D(eval_config, mode='test') # Use the modified config
        test_dataset = test_loader.get_dataset(self.config['batch_size'])

        logger.info(f"Evaluating on {test_loader.dataset_size} test patients for classification.")

        all_y_true = []
        all_y_pred_probs = []

        # Collect true labels and predicted probabilities
        for x_batch, y_true_batch in tqdm(test_dataset, desc="Collecting Classifier Predictions"):
            y_pred_batch = model.predict(x_batch, verbose=0)
            all_y_true.extend(y_true_batch.numpy())
            all_y_pred_probs.extend(y_pred_batch)

        y_true_flat = np.array(all_y_true).flatten()
        y_pred_probs_flat = np.array(all_y_pred_probs)

        # Convert probabilities to predicted classes
        # Assuming binary classification or multi-class where argmax gives the class index
        if y_pred_probs_flat.shape[-1] > 1: # Multi-class
            y_pred_classes = np.argmax(y_pred_probs_flat, axis=-1)
            # If y_true is one-hot encoded, convert it to class labels
            if y_true_flat.ndim > 1 and y_true_flat.shape[-1] > 1:
                y_true_classes = np.argmax(y_true_flat, axis=-1)
            else:
                y_true_classes = y_true_flat
        else: # Binary classification (sigmoid output)
            y_pred_classes = (y_pred_probs_flat > 0.5).astype(int).flatten()
            y_true_classes = y_true_flat

        # Calculate metrics
        accuracy = accuracy_score(y_true_classes, y_pred_classes)
        
        # Precision, Recall, F1-score
        # 'average' can be 'binary', 'micro', 'macro', 'weighted', None
        # For multi-class, 'macro' or 'weighted' are common.
        # Ensure labels are explicitly passed if not all classes are present in y_true_classes
        
        # Determine unique labels for precision/recall/f1 calculation
        unique_labels = np.unique(y_true_classes)
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true_classes, y_pred_classes, average='weighted', labels=unique_labels, zero_division=0
        )

        # AUC-ROC (for binary or multi-class one-vs-rest)
        auc_score = np.nan # Initialize as NaN
        if y_pred_probs_flat.shape[-1] == 1: # Binary classification
            try:
                auc_score = roc_auc_score(y_true_classes, y_pred_probs_flat)
            except ValueError:
                logger.warning("AUC-ROC score cannot be calculated for binary classification with only one class present in y_true.")
        elif y_pred_probs_flat.shape[-1] > 1: # Multi-class
            try:
                # If y_true_flat is one-hot, use it directly. Otherwise, convert to one-hot.
                if y_true_flat.ndim == 1:
                    num_classes = y_pred_probs_flat.shape[-1]
                    y_true_one_hot = tf.keras.utils.to_categorical(y_true_classes, num_classes=num_classes)
                else:
                    y_true_one_hot = y_true_flat # Already one-hot
                auc_score = roc_auc_score(y_true_one_hot, y_pred_probs_flat, multi_class='ovr', average='weighted')
            except ValueError:
                logger.warning("AUC-ROC score cannot be calculated for multi-class classification with insufficient classes or samples.")

        # Confusion Matrix
        cm = confusion_matrix(y_true_classes, y_pred_classes, labels=unique_labels)
        
        # Classification Report (provides precision, recall, f1-score for each class)
        class_report = classification_report(y_true_classes, y_pred_classes, labels=unique_labels, zero_division=0, output_dict=True)

        results_data = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc_roc': auc_score,
        }

        results_df = pd.DataFrame([results_data])
        
        results_path = os.path.join(self.output_path, 'classifier3d_test_results.csv')
        results_df.to_csv(results_path, index=False)
        logger.info(f"Evaluation results saved to {results_path}")

        # Improved formatted output for logger
        logger.info("\n--- Detailed 3D Classifier Model Evaluation Results ---")
        logger.info(f"  Accuracy             : {accuracy:.4f}")
        logger.info(f"  Precision (weighted) : {precision:.4f}")
        logger.info(f"  Recall (weighted)    : {recall:.4f}")
        logger.info(f"  F1-Score (weighted)  : {f1:.4f}")
        if not np.isnan(auc_score):
            logger.info(f"  AUC-ROC (weighted)   : {auc_score:.4f}")
        else:
            logger.info(f"  AUC-ROC (weighted)   : N/A (Insufficient data or classes)")
        
        logger.info("\n  Confusion Matrix:")
        logger.info(f"\n{pd.DataFrame(cm, index=unique_labels, columns=unique_labels).to_string()}")
        
        logger.info("\n  Classification Report:")
        logger.info(f"\n{pd.DataFrame(class_report).transpose().to_string(float_format='%.4f')}")
        logger.info("------------------------------------------")
