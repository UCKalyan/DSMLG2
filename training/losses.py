import tensorflow as tf
from tensorflow.keras import backend as K

def dice_coefficient(y_true, y_pred, smooth=1e-6):
    """
    Calculates the Dice coefficient for multi-class segmentation.
    This is intended to be used as a metric, not a loss function.
    The Dice coefficient is calculated for all foreground classes and averaged.
    """
    # Ensure y_pred is in the correct format (softmax probabilities)
    y_pred = K.softmax(y_pred, axis=-1)
    
    # Skip the background class (channel 0) in Dice calculation
    y_true_foreground = y_true[..., 1:]
    y_pred_foreground = y_pred[..., 1:]
    
    # Flatten the tensors
    y_true_f = K.flatten(y_true_foreground)
    y_pred_f = K.flatten(y_pred_foreground)
    
    # Calculate intersection and union
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f) + K.sum(y_pred_f)
    
    # Calculate Dice coefficient
    dice = (2. * intersection + smooth) / (union + smooth)
    
    return dice

class MultiClassDiceLoss(tf.keras.losses.Loss):
    """
    Custom Keras loss function for multi-class Dice loss.
    Calculates the Dice loss for each foreground class and averages them.
    """
    # --- FIX 1: Update __init__ to accept **kwargs ---
    def __init__(self, smooth=1e-6, name="multi_class_dice_loss", **kwargs):
        # --- FIX 2: Pass kwargs to the parent constructor ---
        super().__init__(name=name, **kwargs)
        self.smooth = smooth

    def call(self, y_true, y_pred):
        """
        Args:
            y_true: Ground truth masks, one-hot encoded.
                    Shape: (batch_size, height, width, depth, num_classes)
            y_pred: Predicted masks (logits from the model).
                    Shape: (batch_size, height, width, depth, num_classes)
        """
        y_pred = K.softmax(y_pred, axis=-1)
        dice_coeff = dice_coefficient(y_true, y_pred, self.smooth)
        return 1 - dice_coeff

    # --- FIX 3: Implement get_config for proper serialization ---
    def get_config(self):
        config = super().get_config()
        config.update({"smooth": self.smooth})
        return config

class DiceCELoss(tf.keras.losses.Loss):
    """
    This class combines Multi-class Dice Loss and Categorical Cross-Entropy Loss,
    similar to the loss function used in the nnU-Net framework.
    Final Loss = (dice_weight * DiceLoss) + (ce_weight * CrossEntropyLoss)
    """
    # --- FIX 1: Update __init__ to accept **kwargs ---
    def __init__(self, ce_weight=1.0, dice_weight=1.0, smooth=1e-6, name="dice_ce_loss_nnunet", **kwargs):
        # --- FIX 2: Pass kwargs to the parent constructor ---
        super().__init__(name=name, **kwargs)
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth
        self.dice_loss = MultiClassDiceLoss(smooth=smooth)
        self.ce_loss = tf.keras.losses.CategoricalCrossentropy()

    def call(self, y_true, y_pred):
        """
        Args:
            y_true: Ground truth masks, one-hot encoded.
            y_pred: Predicted masks (logits from the model).
        """
        y_pred_softmax = K.softmax(y_pred, axis=-1)
        dice = self.dice_loss(y_true, y_pred_softmax)
        ce = self.ce_loss(y_true, y_pred_softmax)
        total_loss = (self.dice_weight * dice) + (self.ce_weight * ce)
        return total_loss

    # --- FIX 3: Implement get_config for proper serialization ---
    def get_config(self):
        config = super().get_config()
        config.update({
            "ce_weight": self.ce_weight,
            "dice_weight": self.dice_weight,
            "smooth": self.smooth
        })
        return config
