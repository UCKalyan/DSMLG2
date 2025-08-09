# evaluation/evaluator.py

import os
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix, classification_report
# --- NEW IMPORT ---
from medpy.metric.binary import hd95

from data.dataset_loader import BratsDataset2D, BratsDataset3D
from training.metrics import (get_custom_objects, dice_coef, dice_coef_wt, dice_coef_tc, dice_coef_et,
                              combined_weighted_loss, create_dice_focal_loss,
                              create_w_t_e_loss, create_weighted_categorical_crossentropy)
from utils.helpers import ensure_dir, load_npy
from utils.logger import get_logger
from inference.reconstruct3d import Reconstructor3D
from inference.ensemble_predictor import EnsemblePredictor


logger = get_logger(__name__)

class Evaluator:
    def __init__(self, config):
        self.config = config
        self.output_path = config['evaluation_output_path']
        ensure_dir(self.output_path)
        self.thresholds = self.config.get('dsc_thresholds', [0.0, 0.25, 0.50, 0.75, 1.0])
        self.reconstructor_3d = Reconstructor3D(config)


    def evaluate_2d_model(self, model_path):
        """
        Evaluates a 2D UNET model on the test set, including Dice coefficients and HD95.
        """
        logger.info("Starting 2D model evaluation on the test set...")
        class_weights = self.config['class_weights']
        loss_function = create_w_t_e_loss(class_weights, wt_w=0.35, tc_w=0.35, et_w=0.3)
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

        dice_at_thresholds = {f'dice_coef_thresh_{int(t*100)}': [] for t in self.thresholds}
        hd95_scores_patient = []

        logger.info("Collecting predictions for advanced metrics...")
        
        test_dataset_for_advanced_metrics = test_loader.get_dataset(self.config['batch_size'])

        y_true_all_slices = []
        y_pred_all_slices = []

        for x_batch, y_true_batch in tqdm(test_dataset_for_advanced_metrics, desc="Collecting Slices for Metrics"):
            y_pred_batch = model.predict(x_batch, verbose=0)
            y_true_all_slices.append(y_true_batch)
            y_pred_all_slices.append(y_pred_batch)

        y_true_all_2d = tf.concat(y_true_all_slices, axis=0)
        y_pred_all_2d = tf.concat(y_pred_all_slices, axis=0)

        for t in self.thresholds:
            y_pred_binarized_foreground = tf.where(y_pred_all_2d[..., 1:] >= t, 1.0, 0.0)
            y_pred_binarized_foreground = tf.cast(y_pred_binarized_foreground, tf.float32) 
            bg_channel = tf.zeros_like(y_pred_all_2d[..., 0:1])
            bg_channel = tf.cast(bg_channel, tf.float32)
            y_pred_binarized_full = tf.concat([bg_channel, y_pred_binarized_foreground], axis=-1)

            current_dice = dice_coef(y_true_all_2d, y_pred_binarized_full).numpy()
            dice_at_thresholds[f'dice_coef_thresh_{int(t*100)}'].append(current_dice)

        for key in dice_at_thresholds:
            dice_at_thresholds[key] = np.mean(dice_at_thresholds[key])

        for patient_id in tqdm(test_loader.ids_to_load, desc="Processing Patients for HD95"):
            pass

        final_hd95_2d_model = np.nanmean(hd95_scores_patient) if hd95_scores_patient else np.nan
            
        combined_results = standard_results + list(dice_at_thresholds.values()) + [final_hd95_2d_model]
        combined_metrics_names = metrics_names + list(dice_at_thresholds.keys()) + ['hd95_reconstructed_3d']

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
            class_weights = self.config['class_weights']
            loss_function = combined_weighted_loss(class_weights)
            custom_objects = get_custom_objects()
            custom_objects['loss'] = loss_function
            model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)

            test_loader = BratsDataset3D(self.config, mode='test')
            test_dataset = test_loader.get_dataset(self.config['batch_size'])

            logger.info(f"Evaluating on {test_loader.dataset_size} test patients.")

            standard_results = model.evaluate(test_dataset, verbose=1)
            
            metrics_names = [
                'loss', 'dice_coef', 'iou', 'precision', 'sensitivity', 'specificity',
                'dice_coef_wt', 'dice_coef_tc', 'dice_coef_et'
            ]
            
            if len(standard_results) != len(metrics_names):
                logger.error(f"Mismatch between number of results ({len(standard_results)}) and metric names ({len(metrics_names)}).")
                logger.error("Please check the metrics in trainer3d_seg.py and update the list in evaluator.py.")
                return

            dice_at_thresholds = {f'dice_coef_thresh_{int(t*100)}': [] for t in self.thresholds}

            logger.info("Collecting 3D predictions for advanced metrics...")
            test_dataset_for_pred = test_loader.get_dataset(self.config['batch_size']).unbatch().batch(1)
            
            all_y_true_3d = []
            all_y_pred_probs_3d = []

            for x_batch, y_true_batch in tqdm(test_dataset_for_pred, desc="Collecting 3D Predictions"):
                y_pred_batch = model.predict(x_batch, verbose=0)
                all_y_true_3d.append(y_true_batch)
                all_y_pred_probs_3d.append(y_pred_batch)

            y_true_all_3d_combined = tf.concat(all_y_true_3d, axis=0)
            y_pred_all_3d_combined = tf.concat(all_y_pred_probs_3d, axis=0)

            voxel_spacing = self.config.get('voxel_spacing', (1.0, 1.0, 1.0))
            hd95_val_3d = np.NAN
            
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
        
        if self.config['model_type'] == '2d':
            data_loader = BratsDataset2D(self.config, mode='validation')
        elif self.config['model_type'] == '3d':
            data_loader = BratsDataset3D(self.config, mode='validation')
        else:
            raise ValueError("model_type must be '2d' or '3d' in config for ensemble evaluation.")

        validation_patient_ids = data_loader.ids_to_load

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
                ground_truth_seg = load_npy(os.path.join(patient_dir, 'segmentation.npy'))

                ensemble_prediction_labels = ensemble_predictor.predict_volume(original_volume)

                if ensemble_prediction_labels is None:
                    logger.warning(f"Skipping patient {patient_id} due to failed ensemble prediction.")
                    continue

                # --- START OF NUMPY DICE CALCULATION FIX ---
                # Calculate Dice scores directly from the label maps using NumPy
                smooth = 1e-6

                # WT Dice (all labels > 0)
                gt_wt_mask = ground_truth_seg > 0
                pred_wt_mask = ensemble_prediction_labels > 0
                intersection_wt = np.sum(gt_wt_mask & pred_wt_mask)
                denominator_wt = np.sum(gt_wt_mask) + np.sum(pred_wt_mask)
                wt_dice = (2. * intersection_wt + smooth) / (denominator_wt + smooth)

                # TC Dice (Labels 1 and 3 after remapping)
                gt_tc_mask = (ground_truth_seg == 1) | (ground_truth_seg == 3)
                pred_tc_mask = (ensemble_prediction_labels == 1) | (ensemble_prediction_labels == 3)
                intersection_tc = np.sum(gt_tc_mask & pred_tc_mask)
                denominator_tc = np.sum(gt_tc_mask) + np.sum(pred_tc_mask)
                tc_dice = (2. * intersection_tc + smooth) / (denominator_tc + smooth)

                # ET Dice (Label 3 after remapping)
                gt_et_mask = ground_truth_seg == 3
                pred_et_mask = ensemble_prediction_labels == 3
                intersection_et = np.sum(gt_et_mask & pred_et_mask)
                denominator_et = np.sum(gt_et_mask) + np.sum(pred_et_mask)
                et_dice = (2. * intersection_et + smooth) / (denominator_et + smooth)
                # --- END OF NUMPY DICE CALCULATION FIX ---

                wt_dice_scores.append(wt_dice)
                tc_dice_scores.append(tc_dice)
                et_dice_scores.append(et_dice)

                voxel_spacing = self.config.get('voxel_spacing', (1.0, 1.0, 1.0))

                # --- HD95 Calculation (using the same masks) ---
                if np.any(gt_wt_mask) and np.any(pred_wt_mask):
                    try:
                        wt_hd95_val = hd95(pred_wt_mask, gt_wt_mask, voxelspacing=voxel_spacing)
                        wt_hd95_scores.append(wt_hd95_val)
                    except RuntimeError:
                        wt_hd95_scores.append(np.nan)
                else:
                    wt_hd95_scores.append(np.nan)

                if np.any(gt_tc_mask) and np.any(pred_tc_mask):
                    try:
                        tc_hd95_val = hd95(pred_tc_mask, gt_tc_mask, voxelspacing=voxel_spacing)
                        tc_hd95_scores.append(tc_hd95_val)
                    except RuntimeError:
                        tc_hd95_scores.append(np.nan)
                else:
                    tc_hd95_scores.append(np.nan)

                if np.any(gt_et_mask) and np.any(pred_et_mask):
                    try:
                        et_hd95_val = hd95(pred_et_mask, gt_et_mask, voxelspacing=voxel_spacing)
                        et_hd95_scores.append(et_hd95_val)
                    except RuntimeError:
                        et_hd95_scores.append(np.nan)
                else:
                    et_hd95_scores.append(np.nan)

            except FileNotFoundError:
                logger.warning(f"Skipping patient {patient_id}: Data files not found.")
            except Exception as e:
                logger.error(f"Error processing patient {patient_id} for ensemble evaluation: {e}")

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

        eval_config = self.config.copy()
        eval_config['output_type'] = 'benign_vs_malignant'
        logger.info("Overriding data loader to yield classification labels for evaluation.")
        
        test_loader = BratsDataset3D(eval_config, mode='test')
        test_dataset = test_loader.get_dataset(self.config['batch_size'])

        logger.info(f"Evaluating on {test_loader.dataset_size} test patients for classification.")

        all_y_true = []
        all_y_pred_probs = []

        for x_batch, y_true_batch in tqdm(test_dataset, desc="Collecting Classifier Predictions"):
            y_pred_batch = model.predict(x_batch, verbose=0)
            all_y_true.extend(y_true_batch.numpy())
            all_y_pred_probs.extend(y_pred_batch)

        y_true_flat = np.array(all_y_true).flatten()
        y_pred_probs_flat = np.array(all_y_pred_probs)

        if y_pred_probs_flat.shape[-1] > 1:
            y_pred_classes = np.argmax(y_pred_probs_flat, axis=-1)
            if y_true_flat.ndim > 1 and y_true_flat.shape[-1] > 1:
                y_true_classes = np.argmax(y_true_flat, axis=-1)
            else:
                y_true_classes = y_true_flat
        else:
            y_pred_classes = (y_pred_probs_flat > 0.5).astype(int).flatten()
            y_true_classes = y_true_flat

        accuracy = accuracy_score(y_true_classes, y_pred_classes)
        
        unique_labels = np.unique(y_true_classes)
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true_classes, y_pred_classes, average='weighted', labels=unique_labels, zero_division=0
        )

        auc_score = np.nan
        if y_pred_probs_flat.shape[-1] == 1:
            try:
                auc_score = roc_auc_score(y_true_classes, y_pred_probs_flat)
            except ValueError:
                logger.warning("AUC-ROC score cannot be calculated for binary classification with only one class present in y_true.")
        elif y_pred_probs_flat.shape[-1] > 1:
            try:
                if y_true_flat.ndim == 1:
                    num_classes = y_pred_probs_flat.shape[-1]
                    y_true_one_hot = tf.keras.utils.to_categorical(y_true_classes, num_classes=num_classes)
                else:
                    y_true_one_hot = y_true_flat
                auc_score = roc_auc_score(y_true_one_hot, y_pred_probs_flat, multi_class='ovr', average='weighted')
            except ValueError:
                logger.warning("AUC-ROC score cannot be calculated for multi-class classification with insufficient classes or samples.")

        cm = confusion_matrix(y_true_classes, y_pred_classes, labels=unique_labels)
        
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
