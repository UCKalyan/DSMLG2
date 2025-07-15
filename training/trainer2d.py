import tensorflow as tf
import math
from models.unet2d import UNET2D
from data.dataset_loader import BratsDataset2D
from training.metrics import (dice_coef, combined_loss, precision, iou, 
                              sensitivity, specificity, dice_coef_necrotic, 
                              dice_coef_edema, dice_coef_enhancing)
from utils.logger import get_logger
from utils.helpers import ensure_dir
import os
import matplotlib.pyplot as plt


logger = get_logger(__name__)

class Trainer2D:
    def __init__(self, config):
        self.config = config
        self.model_builder = UNET2D(config)
        self.model = self.model_builder.build_model()

    def train(self):
        logger.info("Starting 2D UNET training...")

        # Datasets
        train_loader = BratsDataset2D(self.config, mode='train')
        val_loader = BratsDataset2D(self.config, mode='validation')
        
        train_dataset = train_loader.get_dataset(self.config['batch_size'])
        val_dataset = val_loader.get_dataset(self.config['batch_size'])

        # Redefine an epoch to be a more reasonable size.
        avg_slices_per_patient = 64
        
        #steps_per_epoch = math.ceil((train_loader.dataset_size * avg_slices_per_patient) / self.config['batch_size'])
        #validation_steps = math.ceil((val_loader.dataset_size * avg_slices_per_patient) / self.config['batch_size'])
        steps_per_epoch = 100
        validation_steps = 20

        logger.info(f"Number of training patients: {train_loader.dataset_size}")
        logger.info(f"Number of validation patients: {val_loader.dataset_size}")
        logger.info(f"Recalculated training with {steps_per_epoch} steps per epoch and {validation_steps} validation steps.")

        # Compile model with all metrics
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.config['learning_rate']),
            loss=combined_loss,
            metrics=[
                dice_coef, iou, precision, sensitivity, specificity,
                dice_coef_necrotic, dice_coef_edema, dice_coef_enhancing
            ]
        )
        
        # Training History Visualization Callback
        history_plot_callback = self.TrainingHistoryPlotter(log_dir='logs/unet2d')

        # Callbacks
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint('unet2d_best.keras', save_best_only=True, monitor='val_dice_coef', mode='max'),
            tf.keras.callbacks.EarlyStopping(patience=self.config['early_stopping_patience'], monitor='val_dice_coef', mode='max'),
            tf.keras.callbacks.TensorBoard(log_dir='logs/unet2d'),
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
        logger.info("2D UNET training finished.")

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
            # Plot loss
            plt.figure(figsize=(12, 6))
            plt.plot(self.metrics['loss'], label='Training Loss')
            plt.plot(self.metrics['val_loss'], label='Validation Loss')
            plt.title('Loss over Epochs')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(self.log_dir, 'loss_plot.png'))
            plt.close()

            # Plot main Dice coefficient
            plt.figure(figsize=(12, 6))
            plt.plot(self.metrics['dice_coef'], label='Training Dice')
            plt.plot(self.metrics['val_dice_coef'], label='Validation Dice')
            plt.title('Dice Coefficient over Epochs')
            plt.xlabel('Epoch')
            plt.ylabel('Dice')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(self.log_dir, 'dice_plot.png'))
            plt.close()
