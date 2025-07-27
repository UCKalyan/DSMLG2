import tensorflow as tf
import math
from models.unet2d import UNET2D
from data.dataset_loader import BratsDataset2D
from training.metrics import (get_custom_objects, dice_coef,  precision, iou, 
                              sensitivity, specificity, dice_coef_wt, create_w_t_e_loss,
                              dice_coef_tc, dice_coef_et,combined_weighted_loss, create_dice_focal_loss,create_weighted_categorical_crossentropy)
from utils.logger import get_logger
from utils.helpers import ensure_dir
import os
import matplotlib.pyplot as plt
# Import the ReduceLROnPlateau callback
from tensorflow.keras.callbacks import ReduceLROnPlateau #
from tensorflow.keras.optimizers.schedules import CosineDecayRestarts


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

        # Use steps from config
        steps_per_epoch = self.config['steps_per_epoch']
        validation_steps = self.config['validation_steps']
        steps_per_epoch = math.ceil(train_loader.dataset_size / self.config['batch_size'])
        validation_steps = math.ceil(val_loader.dataset_size / self.config['batch_size'])

        # --- FIX: Use the correct weighted loss function ---
        class_weights = self.config['class_weights'] # For BG, NCR/NET, ED, ET
        #loss_function = create_dice_focal_loss(class_weights,loss_factor=0.5)
        #loss_function = create_weighted_categorical_crossentropy(class_weights)
        #loss_function = create_weighted_categorical_crossentropy(class_weights)
        loss_function = create_w_t_e_loss(class_weights,wt_w=0.4, tc_w=0.3, et_w=0.3)
        #loss_function = combined_weighted_loss(class_weights)

        # ---------------------------------------------------

        # Use the CosineDecayRestarts scheduler
        # lr_schedule = CosineDecayRestarts(
        #     initial_learning_rate=5e-5,
        #     first_decay_steps=500,
        #     t_mul=2.0,
        #     m_mul=0.9,
        #     alpha=1e-6
        # )
        
        # lr_scheduler = ReduceLROnPlateau(
        #     monitor='val_loss', 
        #     factor=0.5, 
        #     patience=5, 
        #     min_lr=1e-6,
        #     verbose=1
        # )
        lr_scheduler = ReduceLROnPlateau(
            monitor=self.config['lr_plateau_monitor'], # CHANGE THIS from 'val_loss'
            factor=self.config['lr_plateau_factor'],    # Read from config
            patience=self.config['lr_plateau_patience'], # Read from config
            min_lr=self.config['lr_plateau_min_lr'],   # Read from config
            mode='max',       # Read from config
            verbose=1
        )
        

                # Load the best model from Stage 1
        # self.model = tf.keras.models.load_model(
        #     'unet2d_stage1_best.keras', 
        #     custom_objects=get_custom_objects() # Assumes get_custom_objects is in metrics.py
        # )

        # STAGE 2: Switch back to the Dice-focused loss
        #class_weights = self.config['class_weights']
        #loss_function = create_dice_focal_loss(class_weights, loss_factor=0.5) # 50% Dice
        #loss_function = create_w_t_e_loss(class_weights, wt_w=0.2, tc_w=0.4, et_w=0.4)
        # STAGE 2: Use a very low learning rate for fine-tuning
        # self.model.compile(
        #     optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), # Low LR
        #     loss=loss_function,
        #     metrics=[
        #         dice_coef, iou, precision, sensitivity, specificity,
        #         dice_coef_et, dice_coef_tc, dice_coef_wt
        #     ]
        # )
        # Compile model with all metrics
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.config['learning_rate']),
            loss=loss_function,
            metrics=[
                dice_coef, iou, precision, sensitivity, specificity,
                dice_coef_et, dice_coef_tc, dice_coef_wt
            ]
        )
        



        # Training History Visualization Callback
        history_plot_callback = self.TrainingHistoryPlotter(log_dir='logs/unet2d')


        # Callbacks - Remove ReduceLROnPlateau as CosineDecayRestarts is used
        callbacks = [
            #tf.keras.callbacks.ModelCheckpoint(
            #     'unet2d_best.keras', 
            #     save_best_only=True, 
            #     monitor=self.config['lr_plateau_monitor'], 
            #     mode='max'
            # ),
            # tf.keras.callbacks.EarlyStopping(
            #     monitor=self.config['lr_plateau_monitor'], 
            #     patience=15,
            #     verbose=1, 
            #     mode='max',
            #     restore_best_weights=True
            # ),
            # tf.keras.callbacks.ModelCheckpoint(
            # 'unet2d_best.keras', 
            #     save_best_only=True, 
            #     monitor='val_dice_coef_wt', # Experiment with monitoring TC
            #     mode='max'
            # ),
            # tf.keras.callbacks.EarlyStopping(
            #     monitor='val_dice_coef_wt', # Experiment with monitoring TC
            #     patience=15,
            #     verbose=1, 
            #     mode='max',
            #     restore_best_weights=True
            # ),  
            tf.keras.callbacks.ModelCheckpoint(
                'unet2d_best.keras', 
                save_best_only=True, 
                monitor=self.config['lr_plateau_monitor'], 
                mode='min' # Change this
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor=self.config['lr_plateau_monitor'], 
                patience=self.config['early_stopping_patience'], # Use value from config
                verbose=1, 
                mode='min', # Change this
                restore_best_weights=True
            ),
            tf.keras.callbacks.TensorBoard(log_dir='logs/unet2d'),
            history_plot_callback,
            lr_scheduler
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