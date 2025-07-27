import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from utils.logger import get_logger
from utils.helpers import ensure_dir
from training.metrics import get_custom_objects
from data.dataset_loader import BratsDataset3D


logger = get_logger(__name__)

class Predictor3DClassifier:
    """
    Predictor class for a 3D classification model.
    """
    def __init__(self, config, model_path='classifier3d_best.keras'):
        self.config = config
        # Ensure the output_type is correctly set for the dataset loader
        self.config['output_type'] = 'benign_vs_malignant'
        self.model = tf.keras.models.load_model(
            model_path,
            custom_objects=get_custom_objects() # Ensure custom objects are relevant for classifier
        )
        logger.info(f"3D Classifier model loaded from: {model_path}")

    def predict_volume(self, volume):
        """
        Predicts the class for a single 3D volume.
        
        Args:
            volume (np.ndarray): The input 3D MRI volume (e.g., (D, H, W, C)).
                                 Expected to be preprocessed and ready for model input.

        Returns:
            np.ndarray: Predicted probabilities for each class (e.g., (num_classes,)).
        """
        # Ensure volume has the correct data type as specified in config
        volume = volume.astype(self.config['dtype'])

        if volume.ndim == 4: # Assuming input is (D, H, W, C)
            # Add batch dimension
            volume_expanded = np.expand_dims(volume, axis=0)
        elif volume.ndim == 5: # Already has batch dimension (B, D, H, W, C)
            volume_expanded = volume
        else:
            logger.error(f"Unexpected volume dimensions: {volume.ndim}. Expected 4 or 5.")
            return None

        # Perform prediction
        predictions = self.model.predict(volume_expanded, verbose=0)[0] # Get rid of batch dim
        
        return predictions

    def predict_and_visualize(self, volume, patient_id, true_class):
        """
        Performs prediction and generates visual outputs (text file and plot).
        """
        # Get the prediction probability
        predicted_prob = self.predict_volume(volume)
        
        prediction_score = predicted_prob[0]
        predicted_class = 1 if prediction_score > 0.5 else 0
        
        class_map = {0: 'Benign', 1: 'Malignant'}

        # --- Log to console ---
        logger.info("----- 3D Classification Result -----")
        logger.info(f"Patient ID: {patient_id}")
        logger.info(f"Predicted Score (Malignant Probability): {prediction_score:.4f}")
        logger.info(f"Predicted Class: {predicted_class} ({class_map.get(predicted_class, 'Unknown')})")
        logger.info(f"Ground Truth Class: {true_class} ({class_map.get(true_class, 'Unknown')})")
        logger.info("------------------------------------")
        
        # --- Create visual output files ---
        output_dir = os.path.join(self.config['prediction_output_path'], 'visualizations', patient_id)
        ensure_dir(output_dir)

        # 1. Save results to a text file
        result_text = (
            f"Patient ID: {patient_id}\n"
            f"Predicted Score (Malignant Probability): {prediction_score:.4f}\n"
            f"Predicted Class: {class_map.get(predicted_class, 'Unknown')} ({predicted_class})\n"
            f"Ground Truth Class: {class_map.get(true_class, 'Unknown')} ({true_class})\n"
        )
        txt_path = os.path.join(output_dir, f"{patient_id}_classification_result.txt")
        with open(txt_path, 'w') as f:
            f.write(result_text)
        logger.info(f"Classification result saved to: {txt_path}")

        # 2. Save a plot of the prediction
        fig, ax = plt.subplots(figsize=(6, 5))
        categories = ['Benign', 'Malignant']
        scores = [1 - prediction_score, prediction_score]
        colors = ['skyblue', 'salmon']
        bars = ax.bar(categories, scores, color=colors)
        
        ax.set_ylabel('Probability Score')
        ax.set_title(f'Classification Prediction for Patient {patient_id}')
        ax.set_ylim(0, 1)

        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.4f}', va='bottom', ha='center')

        predicted_bar = bars[predicted_class]
        predicted_bar.set_edgecolor('black')
        predicted_bar.set_linewidth(2)

        plot_path = os.path.join(output_dir, f"{patient_id}_classification_plot.png")
        plt.savefig(plot_path)
        plt.close(fig)
        logger.info(f"Classification plot saved to: {plot_path}")

    def predict_dataset(self, dataset_mode='test', batch_size=None):
        """
        Iterates through a dataset, predicts the class for each volume, and saves results to a CSV file.
        
        Args:
            dataset_mode (str): The dataset split to predict on ('validation' or 'test').
            batch_size (int, optional): The batch size for prediction. If None, uses batch_size from config.
        """
        logger.info(f"----- Running Prediction on '{dataset_mode}' Dataset -----")

        if dataset_mode == 'train':
            logger.error("Dataset prediction with CSV output is not supported for 'train' mode due to data shuffling.")
            return

        if batch_size is None:
            batch_size = self.config.get('batch_size', 4)

        # --- Get Patient IDs in the correct order ---
        # This logic replicates the split in tfrecord_writer.py to get an ordered list of IDs.
        processed_path = self.config['processed_data_path']
        split_ratios = self.config['train_val_test_split']
        # Sort directory listing to ensure a deterministic order before splitting
        all_patient_ids = sorted([
            p for p in os.listdir(processed_path)
            if os.path.isdir(os.path.join(processed_path, p)) and p != 'tfrecords'
        ])
        
        train_val_ids, test_ids = train_test_split(all_patient_ids, test_size=split_ratios[2], random_state=42)
        
        ids_for_mode = []
        if dataset_mode == 'validation':
            _, val_ids = train_test_split(train_val_ids, test_size=split_ratios[1] / (1 - split_ratios[2]), random_state=42)
            ids_for_mode = val_ids
        elif dataset_mode == 'test':
            ids_for_mode = test_ids
        
        # --- Load and Predict from Dataset ---
        dataset_loader = BratsDataset3D(self.config, mode=dataset_mode)
        if dataset_loader.dataset_size == 0:
            logger.error(f"No data found for mode '{dataset_mode}'. Please check your TFRecord files.")
            return
        dataset = dataset_loader.get_dataset(batch_size=batch_size)
        
        all_predictions, all_true_labels = [], []
        logger.info(f"Starting prediction on {dataset_loader.dataset_size} samples in the '{dataset_mode}' dataset...")
        
        for volumes, true_labels in tqdm(dataset, desc=f"Predicting on {dataset_mode} set", total=int(np.ceil(dataset_loader.dataset_size / batch_size))):
            batch_predictions = self.model.predict(volumes, verbose=0)
            all_predictions.extend(batch_predictions)
            all_true_labels.extend(true_labels.numpy())

        if len(ids_for_mode) != len(all_predictions):
            logger.error(f"Mismatch between patient IDs ({len(ids_for_mode)}) and predictions ({len(all_predictions)}). Cannot generate CSV.")
            return

        # --- Process Results ---
        prediction_scores = np.array([p[0] for p in all_predictions])
        predicted_classes = (prediction_scores > 0.5).astype(int)
        true_classes = np.array([int(label[0]) for label in all_true_labels])

        # --- Save results to CSV file ---
        class_map = {0: 'Benign', 1: 'Malignant'}
        results_df = pd.DataFrame({
            'Patient_ID': sorted(ids_for_mode), # Sort IDs to match the dataset order if necessary
            'True_Label': [class_map.get(tc) for tc in true_classes],
            'Predicted_Label': [class_map.get(pc) for pc in predicted_classes],
            'Malignant_Probability_Score': prediction_scores,
        })
        
        output_dir = self.config['evaluation_output_path']
        ensure_dir(output_dir)
        csv_path = os.path.join(output_dir, f"classification_results_{dataset_mode}.csv")
        results_df.to_csv(csv_path, index=False)
        logger.info(f"Prediction results saved to: {csv_path}")

        # --- Generate and save classification report ---
        accuracy = accuracy_score(true_classes, predicted_classes)
        #report = classification_report(true_classes, predicted_classes, target_names=class_map.values())
        # ----- THIS IS THE FIX -----
        # Explicitly define all possible labels and their names.
        all_possible_labels = list(class_map.keys()) # [0, 1]
        all_target_names = list(class_map.values()) # ['Benign', 'Malignant']

        report = classification_report(
            true_classes,
            predicted_classes,
            labels=all_possible_labels,      # Pass all possible labels
            target_names=all_target_names,  # Pass the corresponding names
            zero_division=0                 # Avoids warnings if a class is never predicted
        )
        # ---------------------------
        logger.info("----- Dataset Prediction Complete -----")
        logger.info(f"\nOverall Accuracy on '{dataset_mode}' set: {accuracy:.4f}\n")
        logger.info(f"Classification Report:\n{report}")

        report_path = os.path.join(output_dir, f"classification_report_{dataset_mode}.txt")
        with open(report_path, 'w') as f:
            f.write(f"Classification Report for '{dataset_mode}' Dataset\n")
            f.write("="*50 + "\n")
            f.write(f"Overall Accuracy: {accuracy:.4f}\n\n")
            f.write(report)
        logger.info(f"Full classification report saved to: {report_path}")