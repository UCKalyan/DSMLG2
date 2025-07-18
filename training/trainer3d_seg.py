import tensorflow as tf
import math
import os
import matplotlib.pyplot as plt
from models.unet3d import UNET3D
from data.dataset_loader import BratsDataset3D
from training.metrics import (dice_coef, combined_loss, precision, iou, 
                              sensitivity, specificity, dice_coef_wt, 
                              dice_coef_tc, dice_coef_et)
from utils.logger import get_logger
from utils.helpers import ensure_dir

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
        
        # Calculate steps
        #steps_per_epoch = math.ceil(train_loader.dataset_size / self.config['batch_size'])
        #validation_steps = math.ceil(val_loader.dataset_size / self.config['batch_size'])

        if self.config['active_profile'] == 'cpu':
            steps_per_epoch = self.config['steps_per_epoch_3d']
            validation_steps = self.config['validation_steps_3d']
        else:
            steps_per_epoch = math.ceil(train_loader.dataset_size / self.config['batch_size'])
            validation_steps = math.ceil(val_loader.dataset_size / self.config['batch_size'])

        # Compile model
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.config['learning_rate']),
            loss=combined_loss,
            metrics=[
                dice_coef, iou, precision, sensitivity, specificity,
                dice_coef_wt, dice_coef_tc, dice_coef_et
            ]
        )

        # Training History Visualization Callback
        history_plot_callback = self.TrainingHistoryPlotter(log_dir='logs/unet3d_seg')

        # Callbacks
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint('unet3d_seg_best.keras', save_best_only=True, monitor='val_dice_coef', mode='max'),
            tf.keras.callbacks.EarlyStopping(patience=self.config['early_stopping_patience'], monitor='val_dice_coef', mode='max'),
            tf.keras.callbacks.TensorBoard(log_dir='logs/unet3d_seg'),
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
            # Plot loss
            plt.figure(figsize=(12, 5))
            plt.subplot(1, 2, 1)
            plt.plot(self.metrics['loss'], label='Training Loss')
            plt.plot(self.metrics['val_loss'], label='Validation Loss')
            plt.title('Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)

            # Plot Dice coefficient
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