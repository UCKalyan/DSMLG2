import numpy as np
import tensorflow as tf
from tqdm import tqdm
from utils.logger import get_logger
from training.metrics import get_custom_objects # Assuming custom objects are needed for classifier too
import os
import matplotlib.pyplot as plt
from utils.helpers import ensure_dir

logger = get_logger(__name__)

class Predictor3DClassifier:
    """
    Predictor class for a 3D classification model.
    """
    def __init__(self, config, model_path='classifier3d_best.keras'):
        self.config = config
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

        logger.info(f"Predicting class for volume with shape: {volume_expanded.shape}")
        
        # Perform prediction
        predictions = self.model.predict(volume_expanded, verbose=0)[0] # Get rid of batch dim
        
        logger.info("Prediction complete.")
        return predictions

    def predict_and_visualize(self, volume, patient_id, true_class):
        """
        Performs prediction and generates visual outputs (text file and plot).
        """
        # Get the prediction probability
        predicted_prob = self.predict_volume(volume)
        
        # The output is a single value from a sigmoid function
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

        # Add score labels on top of bars
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.4f}', va='bottom', ha='center')

        # Highlight the predicted class
        predicted_bar = bars[predicted_class]
        predicted_bar.set_edgecolor('black')
        predicted_bar.set_linewidth(2)

        plot_path = os.path.join(output_dir, f"{patient_id}_classification_plot.png")
        plt.savefig(plot_path)
        plt.close(fig)
        logger.info(f"Classification plot saved to: {plot_path}")
