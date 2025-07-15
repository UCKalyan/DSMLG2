import tensorflow as tf
import math
import os
import matplotlib.pyplot as plt
from models.classifier3d import Classifier3D
from data.dataset_loader import BratsDataset3D
from utils.logger import get_logger
from utils.helpers import ensure_dir

logger = get_logger(__name__)

class Trainer3DClassifier:
    def __init__(self, config):
        self.config = config
        self.model_builder = Classifier3D(config)
        self.model = self.model_builder.build_model()

    def train(self):
        logger.info("Starting 3D Classifier training...")

        # Datasets
        train_loader = BratsDataset3D(self.config, mode='train')
        val_loader = BratsDataset3D(self.config, mode='validation')

        train_dataset = train_loader.get_dataset(self.config['batch_size'])
        val_dataset = val_loader.get_dataset(self.config['batch_size'])

        # Calculate steps
        #steps_per_epoch = math.ceil(train_loader.dataset_size / self.config['batch_size'])
        #validation_steps = math.ceil(val_loader.dataset_size / self.config['batch_size'])
        steps_per_epoch = 50
        validation_steps = 10
        
        logger.info(f"Training with {steps_per_epoch} steps per epoch and {validation_steps} validation steps.")

        # Compile model
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.config['learning_rate']),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(name='roc_auc')]
        )

        # Training History Visualization Callback
        history_plot_callback = self.TrainingHistoryPlotter(log_dir='logs/classifier3d')

        # Callbacks
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint('classifier3d_best.keras', save_best_only=True, monitor='val_roc_auc', mode='max'),
            tf.keras.callbacks.EarlyStopping(patience=self.config['early_stopping_patience'], monitor='val_roc_auc', mode='max'),
            tf.keras.callbacks.TensorBoard(log_dir='logs/classifier3d'),
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
        logger.info("3D Classifier training finished.")

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
            plt.figure(figsize=(18, 5))
            plt.subplot(1, 3, 1)
            plt.plot(self.metrics['loss'], label='Training Loss')
            plt.plot(self.metrics['val_loss'], label='Validation Loss')
            plt.title('Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)

            # Plot accuracy
            plt.subplot(1, 3, 2)
            plt.plot(self.metrics['accuracy'], label='Training Accuracy')
            plt.plot(self.metrics['val_accuracy'], label='Validation Accuracy')
            plt.title('Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.grid(True)
            
            # Plot ROC AUC
            plt.subplot(1, 3, 3)
            plt.plot(self.metrics['roc_auc'], label='Training ROC AUC')
            plt.plot(self.metrics['val_roc_auc'], label='Validation ROC AUC')
            plt.title('ROC AUC')
            plt.xlabel('Epoch')
            plt.ylabel('AUC')
            plt.legend()
            plt.grid(True)

            plt.tight_layout()
            plt.savefig(os.path.join(self.log_dir, 'training_plots.png'))
            plt.close()
