import tensorflow as tf
import math
import os
import matplotlib.pyplot as plt

from models.unet3d import UNET3D
from data.dataset_loader import BratsDataset3D
from training.metrics import (
    combined_weighted_loss, dice_coef, iou, precision, 
    sensitivity, specificity, dice_coef_wt, dice_coef_tc, dice_coef_et
)
from utils.logger import get_logger
from utils.helpers import ensure_dir
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, TensorBoard
from tensorflow.keras.optimizers.schedules import CosineDecayRestarts

# Enable mixed precision for performance
tf.keras.mixed_precision.set_global_policy('mixed_float16')

logger = get_logger(__name__)

class Trainer3DSegmentation:
    def __init__(self, config):
        self.config = config
        self.model_builder = UNET3D(config)
        self.model = self.model_builder.build_model()

    def train(self):
        logger.info("Starting 3D UNET Segmentation training...")

        # Datasets
        train_loader = BratsDataset3D(self.config, mode='train')
        val_loader = BratsDataset3D(self.config, mode='validation')

        train_dataset = train_loader.get_dataset(self.config['batch_size'])
        val_dataset = val_loader.get_dataset(self.config['batch_size'])
        
        # Use steps from config
        steps_per_epoch = self.config['steps_per_epoch_3d']
        validation_steps = self.config['validation_steps_3d']

                # Dynamically calculate steps to define a true epoch
        steps_per_epoch = math.ceil(train_loader.dataset_size / self.config['batch_size'])
        validation_steps = math.ceil(val_loader.dataset_size / self.config['batch_size'])

        logger.info(f"Number of training patches: {train_loader.dataset_size}")
        logger.info(f"Number of validation patches: {val_loader.dataset_size}")
        logger.info(f"Calculated training steps: {steps_per_epoch} per epoch")
        logger.info(f"Calculated validation steps: {validation_steps}")

        # Define class weights and the loss function
        # Define class weights and the loss function
        class_weights = self.config['class_weights'] # For BG, NCR/NET, ED, ET
        loss_function = combined_weighted_loss(class_weights)

        # --- OPTIMIZATION: Use a better learning rate scheduler ---
        lr_schedule = CosineDecayRestarts(
            initial_learning_rate=self.config['learning_rate'],
            first_decay_steps=steps_per_epoch * 10,  # e.g., restart every 10 epochs
            t_mul=2.0,
            m_mul=0.9,
            alpha=1e-6
        )
        
        # Compile model with the new scheduler
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
            loss=loss_function,
            metrics=[
                dice_coef, iou, precision, sensitivity, specificity,
                dice_coef_wt, dice_coef_tc, dice_coef_et
            ]
        )

        # Training History Visualization Callback
        history_plot_callback = self.TrainingHistoryPlotter(log_dir='logs/unet3d_seg')

        # --- OPTIMIZATION: Remove ReduceLROnPlateau ---
        callbacks = [
            ModelCheckpoint(
                'unet3d_seg_best.keras', 
                save_best_only=True, 
                monitor='val_dice_coef',
                mode='max'
            ),
            EarlyStopping(
                patience=self.config['early_stopping_patience'], 
                monitor='val_loss',
                mode='min',
                restore_best_weights=True
            ),
            TensorBoard(log_dir='logs/unet3d_seg'),
            history_plot_callback
        ]

        # Train
        self.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=self.config['epochs'],
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            callbacks=callbacks
        )
        logger.info("3D UNET Segmentation training finished.")

    class TrainingHistoryPlotter(tf.keras.callbacks.Callback):
        """A callback to plot training and validation metrics at the end of each epoch."""
        def __init__(self, log_dir):
            super().__init__()
            self.log_dir = log_dir
            ensure_dir(self.log_dir)
            self.metrics = {}

        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            for k, v in logs.items():
                self.metrics.setdefault(k, []).append(v)
            
            if epoch > 0:
                self.plot_metrics()

        def plot_metrics(self):
            plt.figure(figsize=(12, 5))
            
            # Plot Loss
            plt.subplot(1, 2, 1)
            plt.plot(self.metrics['loss'], label='Training Loss')
            plt.plot(self.metrics['val_loss'], label='Validation Loss')
            plt.title('Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)

            # Plot Dice Coefficient
            plt.subplot(1, 2, 2)
            plt.plot(self.metrics['dice_coef'], label='Training Dice')
            plt.plot(self.metrics['val_dice_coef'], label='Validation Dice')
            plt.title('Dice Coefficient')
            plt.xlabel('Epoch')
            plt.ylabel('Dice')
            plt.legend()
            plt.grid(True)

            plt.tight_layout()
            plt.savefig(os.path.join(self.log_dir, 'training_plots.png'))
            plt.close()