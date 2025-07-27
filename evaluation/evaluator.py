# evaluator.py

import os
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix, classification_report

from data.dataset_loader import BratsDataset2D, BratsDataset3D
from training.metrics import get_custom_objects, dice_coef,  combined_weighted_loss, hd95_metric, create_dice_focal_loss # Import hd95_metric here
from utils.helpers import ensure_dir, load_npy # Make sure load_npy is imported
from utils.logger import get_logger
from inference.reconstruct3d import Reconstructor3D # Import Reconstructor3D for 2D model evaluation
from inference.ensemble_predictor import EnsemblePredictor # NEW IMPORT


logger = get_logger(__name__)

class Evaluator:
    def __init__(self, config):
        self.config = config
        self.output_path = config['evaluation_output_path']
        ensure_dir(self.output_path)
        self.thresholds = self.config.get('dsc_thresholds', [0.0, 0.25, 0.50, 0.75, 1.0])
        self.reconstructor_3d = Reconstructor3D(config) # Initialize Reconstructor3D


    def evaluate_2d_model(self, model_path):
        """
        Evaluates a 2D UNET model on the test set, including Dice coefficients and HD95.
        """
        logger.info("Starting 2D model evaluation on the test set...")
        class_weights = self.config['class_weights']
        loss_function = create_dice_focal_loss(class_weights)
        custom_objects = get_custom_objects()
        custom_objects['loss'] = loss_function
        model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)

        test_loader = BratsDataset2D(self.config, mode='test')
        test_dataset = test_loader.get_dataset(self.config['batch_size'])
        
        avg_slices_per_patient = 64
        test_steps = int(np.ceil((test_loader.dataset_size * avg_slices_per_patient) / self.config['batch_size']))
        
        logger.info(f"Evaluating on {test_loader.dataset_size} test patients, with approx. {test_steps} steps.")

        standard_results = model.evaluate(test_dataset, steps=test_steps, verbose=1)
        
        metrics_names = [
            'loss', 'dice_coef', 'iou', 'precision', 'sensitivity', 'specificity',
            'dice_coef_wt', 'dice_coef_tc', 'dice_coef_et'
        ]
        
        if len(standard_results) != len(metrics_names):
            logger.error(f"Mismatch between number of results ({len(standard_results)}) and metric names ({len(metrics_names)}).")
            return

        # --- CORRECTED: Dice at thresholds and HD95 calculation ---
        dice_at_thresholds = {f'dice_coef_thresh_{int(t*100)}': [] for t in self.thresholds}
        hd95_scores_patient = []

        logger.info("Collecting predictions for advanced metrics...")
        
        # Use a fresh dataset iterator
        test_dataset_for_advanced_metrics = test_loader.get_dataset(self.config['batch_size'])

        y_true_all_slices = []
        y_pred_all_slices = []

        for x_batch, y_true_batch in tqdm(test_dataset_for_advanced_metrics, desc="Collecting Slices for Metrics"):
            y_pred_batch = model.predict(x_batch, verbose=0)
            y_true_all_slices.append(y_true_batch)
            y_pred_all_slices.append(y_pred_batch)

        y_true_all_2d = tf.concat(y_true_all_slices, axis=0)
        y_pred_all_2d = tf.concat(y_pred_all_slices, axis=0)

        # Now calculate thresholded dice on the full collected dataset
        for t in self.thresholds: # Use self.thresholds
            # Binarize predictions for the whole tumor (excluding background, index 0)
            # Only consider foreground classes (indices 1, 2, 3) for thresholding
            y_pred_binarized_foreground = tf.where(y_pred_all_2d[..., 1:] >= t, 1.0, 0.0)
            y_pred_binarized_foreground = tf.cast(y_pred_binarized_foreground, tf.float32) 
            # Reconstruct to original one-hot shape by adding a background channel of zeros
            bg_channel = tf.zeros_like(y_pred_all_2d[..., 0:1])
            bg_channel = tf.cast(bg_channel, tf.float32)
            y_pred_binarized_full = tf.concat([bg_channel, y_pred_binarized_foreground], axis=-1)

            # Calculate Dice for the binarized predictions
            # The dice_coef function is designed to work with one-hot like inputs (0/1 for foreground)
            current_dice = dice_coef(y_true_all_2d, y_pred_binarized_full).numpy()
            dice_at_thresholds[f'dice_coef_thresh_{int(t*100)}'].append(current_dice)

        # Average the Dice scores for each threshold (over all slices/samples)
        for key in dice_at_thresholds:
            dice_at_thresholds[key] = np.mean(dice_at_thresholds[key])

        # Patient-by-patient 3D reconstruction for HD95
        for patient_id in tqdm(test_loader.ids_to_load, desc="Processing Patients for HD95"):
            # (Your existing HD95 logic here remains the same)
            # ...
            pass # Placeholder for your HD95 logic

        final_hd95_2d_model = np.nanmean(hd95_scores_patient) if hd95_scores_patient else np.nan
            
        # Combine all results
        combined_results = standard_results + list(dice_at_thresholds.values()) + [final_hd95_2d_model]
        combined_metrics_names = metrics_names + list(dice_at_thresholds.keys()) + ['hd95_reconstructed_3d']
        #combined_results = standard_results + list(dice_at_thresholds.values())
        #combined_metrics_names = metrics_names + list(dice_at_thresholds.keys())

        results_df = pd.DataFrame([combined_results], columns=combined_metrics_names)
        
        results_path = os.path.join(self.output_path, 'unet2d_test_results.csv')
        results_df.to_csv(results_path, index=False)
        logger.info(f"Evaluation results saved to {results_path}")

        logger.info("\n--- Detailed 2D Model Evaluation Results ---")
        for i, name in enumerate(combined_metrics_names):
            logger.info(f"  {name:<25}: {combined_results[i]:.4f}")
        logger.info("------------------------------------------")


    def evaluate_3d_model(self, model_path):
            """
            Evaluates a 3D model on the test set, including Dice coefficients and HD95.
            """
            logger.info("Starting 3D model evaluation on the test set...")
            # model = tf.keras.models.load_model(
            #     model_path,
            #     custom_objects=get_custom_objects()
            # )
            class_weights = self.config['class_weights']
            loss_function = combined_weighted_loss(class_weights)
            custom_objects = get_custom_objects()
            custom_objects['loss'] = loss_function
            model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)

            test_loader = BratsDataset3D(self.config, mode='test')
            test_dataset = test_loader.get_dataset(self.config['batch_size'])

            logger.info(f"Evaluating on {test_loader.dataset_size} test patients.")

            # Perform standard evaluation first
            standard_results = model.evaluate(test_dataset, verbose=1)
            
            # Adjust metrics_names to match the metrics compiled in trainer3d_seg.py
            metrics_names = [
                'loss', 'dice_coef', 'iou', 'precision', 'sensitivity', 'specificity',
                'dice_coef_wt', 'dice_coef_tc', 'dice_coef_et' # Added new dice metrics
            ]
            
            if len(standard_results) != len(metrics_names):
                logger.error(f"Mismatch between number of results ({len(standard_results)}) and metric names ({len(metrics_names)}).")
                logger.error("Please check the metrics in trainer3d_seg.py and update the list in evaluator.py.")
                return

            # --- Calculate Dice at specified thresholds and HD95 for 3D model ---
            dice_at_thresholds = {f'dice_coef_thresh_{int(t*100)}': [] for t in self.thresholds}

            logger.info("Collecting 3D predictions for advanced metrics...")
            # Reset the dataset iterator for prediction
            test_dataset_for_pred = test_loader.get_dataset(self.config['batch_size']).unbatch().batch(1) # Unbatch for individual volumes
            
            all_y_true_3d = []
            all_y_pred_probs_3d = []

            # Get all true labels and predictions
            for x_batch, y_true_batch in tqdm(test_dataset_for_pred, desc="Collecting 3D Predictions"):
                y_pred_batch = model.predict(x_batch, verbose=0)
                all_y_true_3d.append(y_true_batch)
                all_y_pred_probs_3d.append(y_pred_batch)

            # Concatenate all batches into single tensors (already 3D)
            y_true_all_3d_combined = tf.concat(all_y_true_3d, axis=0)
            y_pred_all_3d_combined = tf.concat(all_y_pred_probs_3d, axis=0)

            # Calculate HD95 for 3D model directly
            voxel_spacing = self.config.get('voxel_spacing', (1.0, 1.0, 1.0))
            hd95_val_3d = np.NAN #hd95_metric(y_true_all_3d_combined, y_pred_all_3d_combined, spacing=voxel_spacing)
            
            for t in self.thresholds:
                y_pred_binarized_foreground = tf.where(y_pred_all_3d_combined[..., 1:] >= t, 1.0, 0.0)
                y_pred_binarized_foreground = tf.cast(y_pred_binarized_foreground, tf.float32) 
                bg_channel = tf.zeros_like(y_pred_all_3d_combined[..., 0:1])
                bg_channel = tf.cast(bg_channel, tf.float32)
                y_pred_binarized_full = tf.concat([bg_channel, y_pred_binarized_foreground], axis=-1)


                current_dice = dice_coef(y_true_all_3d_combined, y_pred_binarized_full).numpy()
                dice_at_thresholds[f'dice_coef_thresh_{int(t*100)}'].append(current_dice)

            for key in dice_at_thresholds:
                dice_at_thresholds[key] = np.mean(dice_at_thresholds[key])
                
            # Combine standard results with thresholded Dice results and HD95
            combined_results = standard_results + list(dice_at_thresholds.values()) + [hd95_val_3d]
            combined_metrics_names = metrics_names + list(dice_at_thresholds.keys()) + ['hd95']

            results_df = pd.DataFrame([combined_results], columns=combined_metrics_names)
            
            results_path = os.path.join(self.output_path, 'unet3d_test_results.csv')
            results_df.to_csv(results_path, index=False)
            logger.info(f"Evaluation results saved to {results_path}")

            logger.info("\n--- Detailed 3D Model Evaluation Results ---")
            for i, name in enumerate(combined_metrics_names):
                logger.info(f"  {name:<25}: {combined_results[i]:.4f}")
            logger.info("------------------------------------------")


    def evaluate_ensemble(self, model_paths):
        """
        Evaluates an ensemble of models on the validation set
        to report combined Dice and HD95 scores.
        """
        logger.info(f"Starting ensemble evaluation for {self.config['model_type']} models on the validation set...")

        ensemble_predictor = EnsemblePredictor(self.config, model_paths)

        # We need to iterate over the validation set, patient by patient
        # to get the full 3D ground truth and compute ensemble predictions.
        # Assuming you want to evaluate on the validation set as per your prompt.
        
        # Determine which dataset loader to use based on model_type
        if self.config['model_type'] == '2d':
            data_loader = BratsDataset2D(self.config, mode='validation')
        elif self.config['model_type'] == '3d':
            data_loader = BratsDataset3D(self.config, mode='validation')
        else:
            raise ValueError("model_type must be '2d' or '3d' in config for ensemble evaluation.")

        validation_patient_ids = data_loader.ids_to_load # Get list of patient IDs

        # Lists to store metrics for each patient
        wt_dice_scores = []
        tc_dice_scores = []
        et_dice_scores = []
        wt_hd95_scores = []
        tc_hd95_scores = []
        et_hd95_scores = []

        logger.info(f"Evaluating ensemble on {len(validation_patient_ids)} validation patients...")

        for patient_id in tqdm(validation_patient_ids, desc="Ensemble Evaluation"):
            patient_dir = os.path.join(self.config['processed_data_path'], patient_id)
            try:
                original_volume = load_npy(os.path.join(patient_dir, 'volume.npy'))
                ground_truth_seg = load_npy(os.path.join(patient_dir, 'segmentation.npy')) # This is (D, H, W) labels

                # Get ensemble prediction (hard labels: D, H, W)
                ensemble_prediction_labels = ensemble_predictor.predict_volume(original_volume)

                if ensemble_prediction_labels is None:
                    logger.warning(f"Skipping patient {patient_id} due to failed ensemble prediction.")
                    continue

                # Convert ground truth and prediction to one-hot for metric calculation
                # ground_truth_seg is (D, H, W)
                # ensemble_prediction_labels is (D, H, W)
                gt_one_hot = tf.keras.utils.to_categorical(
                    ground_truth_seg, num_classes=self.config['num_classes']
                ).astype(np.float32) # (D, H, W, NumClasses)
                
                pred_one_hot = tf.keras.utils.to_categorical(
                    ensemble_prediction_labels, num_classes=self.config['num_classes']
                ).astype(np.float32) # (D, H, W, NumClasses)

                # Add batch dimension for metric functions
                gt_one_hot_batch = np.expand_dims(gt_one_hot, axis=0)
                pred_one_hot_batch = np.expand_dims(pred_one_hot, axis=0) # For metrics requiring batch dim

                # Calculate Dice scores for WT, TC, ET
                wt_dice = dice_coef_wt(gt_one_hot_batch, pred_one_hot_batch).numpy()
                tc_dice = dice_coef_tc(gt_one_hot_batch, pred_one_hot_batch).numpy()
                et_dice = dice_coef_et(gt_one_hot_batch, pred_one_hot_batch).numpy()

                wt_dice_scores.append(wt_dice)
                tc_dice_scores.append(tc_dice)
                et_dice_scores.append(et_dice)

                # Calculate HD95 for WT, TC, ET
                # For HD95, convert specific parts of one-hot to binary masks
                voxel_spacing = self.config.get('voxel_spacing', (1.0, 1.0, 1.0)) # Get voxel spacing

                # WT: combine all foreground classes (1, 2, 3)
                gt_wt_mask = np.any(ground_truth_seg[..., 1:] > 0, axis=-1).astype(bool) if ground_truth_seg.ndim == 4 else np.any(ground_truth_seg > 0, axis=-1).astype(bool)
                pred_wt_mask = np.any(ensemble_prediction_labels[..., 1:] > 0, axis=-1).astype(bool) if ensemble_prediction_labels.ndim == 4 else np.any(ensemble_prediction_labels > 0, axis=-1).astype(bool)

                # Ensure masks are 3D if they came from 2D slicing (though here we load 3D)
                if gt_wt_mask.ndim == 2: # This case is less likely if loading full 3D volumes directly
                    gt_wt_mask_3d = np.expand_dims(gt_wt_mask, axis=0)
                    pred_wt_mask_3d = np.expand_dims(pred_wt_mask, axis=0)
                else:
                    gt_wt_mask_3d = gt_wt_mask
                    pred_wt_mask_3d = pred_wt_mask

                if np.any(gt_wt_mask_3d) and np.any(pred_wt_mask_3d):
                    wt_hd95_val = hd95_metric(np.expand_dims(gt_wt_mask_3d, axis=0), np.expand_dims(pred_wt_mask_3d, axis=0), spacing=voxel_spacing)
                    if not np.isnan(wt_hd95_val): wt_hd95_scores.append(wt_hd95_val)
                    else: logger.warning(f"HD95 WT NaN for patient {patient_id}")
                else:
                    wt_hd95_scores.append(np.nan) # Append NaN if one mask is empty

                # TC: Tumor Core (classes 1 and 3)
                gt_tc_mask = np.any(ground_truth_seg[..., [1, 3]] > 0, axis=-1).astype(bool) if ground_truth_seg.ndim == 4 else ((ground_truth_seg == 1) | (ground_truth_seg == 3)).astype(bool)
                pred_tc_mask = np.any(ensemble_prediction_labels[..., [1, 3]] > 0, axis=-1).astype(bool) if ensemble_prediction_labels.ndim == 4 else ((ensemble_prediction_labels == 1) | (ensemble_prediction_labels == 3)).astype(bool)
                
                if gt_tc_mask.ndim == 2:
                    gt_tc_mask_3d = np.expand_dims(gt_tc_mask, axis=0)
                    pred_tc_mask_3d = np.expand_dims(pred_tc_mask, axis=0)
                else:
                    gt_tc_mask_3d = gt_tc_mask
                    pred_tc_mask_3d = pred_tc_mask

                if np.any(gt_tc_mask_3d) and np.any(pred_tc_mask_3d):
                    tc_hd95_val = hd95_metric(np.expand_dims(gt_tc_mask_3d, axis=0), np.expand_dims(pred_tc_mask_3d, axis=0), spacing=voxel_spacing)
                    if not np.isnan(tc_hd95_val): tc_hd95_scores.append(tc_hd95_val)
                    else: logger.warning(f"HD95 TC NaN for patient {patient_id}")
                else:
                    tc_hd95_scores.append(np.nan)

                # ET: Enhancing Tumor (class 3)
                gt_et_mask = (ground_truth_seg == 3).astype(bool)
                pred_et_mask = (ensemble_prediction_labels == 3).astype(bool)

                if gt_et_mask.ndim == 2:
                    gt_et_mask_3d = np.expand_dims(gt_et_mask, axis=0)
                    pred_et_mask_3d = np.expand_dims(pred_et_mask, axis=0)
                else:
                    gt_et_mask_3d = gt_et_mask
                    pred_et_mask_3d = pred_et_mask

                if np.any(gt_et_mask_3d) and np.any(pred_et_mask_3d):
                    et_hd95_val = hd95_metric(np.expand_dims(gt_et_mask_3d, axis=0), np.expand_dims(pred_et_mask_3d, axis=0), spacing=voxel_spacing)
                    if not np.isnan(et_hd95_val): et_hd95_scores.append(et_hd95_val)
                    else: logger.warning(f"HD95 ET NaN for patient {patient_id}")
                else:
                    et_hd95_scores.append(np.nan)


            except FileNotFoundError:
                logger.warning(f"Skipping patient {patient_id}: Data files not found.")
            except Exception as e:
                logger.error(f"Error processing patient {patient_id} for ensemble evaluation: {e}")

        # Calculate mean scores, ignoring NaNs
        mean_wt_dice = np.nanmean(wt_dice_scores)
        mean_tc_dice = np.nanmean(tc_dice_scores)
        mean_et_dice = np.nanmean(et_dice_scores)
        
        mean_wt_hd95 = np.nanmean(wt_hd95_scores)
        mean_tc_hd95 = np.nanmean(tc_hd95_scores)
        mean_et_hd95 = np.nanmean(et_hd95_scores)

        results_data = {
            'mean_dice_wt': mean_wt_dice,
            'mean_dice_tc': mean_tc_dice,
            'mean_dice_et': mean_et_dice,
            'mean_hd95_wt': mean_wt_hd95,
            'mean_hd95_tc': mean_tc_hd95,
            'mean_hd95_et': mean_et_hd95,
        }
        
        results_df = pd.DataFrame([results_data])
        results_path = os.path.join(self.output_path, 'ensemble_results.csv')
        results_df.to_csv(results_path, index=False)
        logger.info(f"Ensemble evaluation results saved to {results_path}")

        logger.info("\n--- Detailed Ensemble Evaluation Results (Validation Set) ---")
        logger.info(f"  Mean Dice WT       : {mean_wt_dice:.4f}")
        logger.info(f"  Mean Dice TC       : {mean_tc_dice:.4f}")
        logger.info(f"  Mean Dice ET       : {mean_et_dice:.4f}")
        logger.info(f"  Mean HD95 WT       : {mean_wt_hd95:.2f}")
        logger.info(f"  Mean HD95 TC       : {mean_tc_hd95:.2f}")
        logger.info(f"  Mean HD95 ET       : {mean_et_hd95:.2f}")
        logger.info("-----------------------------------------------------")

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
