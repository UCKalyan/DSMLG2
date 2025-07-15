import tensorflow as tf
import numpy as np
from medpy.metric.binary import hd95

def dice_coef(y_true, y_pred, smooth=1e-6):
    """Computes the Dice coefficient for the whole tumor."""
    # y_true and y_pred are one-hot encoded.
    # We want to compare the foreground classes (anything that is not background)
    y_true_f = tf.keras.backend.flatten(y_true[..., 1:])
    y_pred_f = tf.keras.backend.flatten(y_pred[..., 1:])
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

def dice_coef_necrotic(y_true, y_pred):
    """Dice coefficient for the necrotic/non-enhancing core (label 1)."""
    return dice_coef(y_true[..., 1:2], y_pred[..., 1:2])

def dice_coef_edema(y_true, y_pred):
    """Dice coefficient for the peritumoral edema (label 2)."""
    return dice_coef(y_true[..., 2:3], y_pred[..., 2:3])

def dice_coef_enhancing(y_true, y_pred):
    """Dice coefficient for the GD-enhancing tumor (label 3 -> originally 4)."""
    # In our 4-class setup (BG, NCR, ED, ET), ET is at index 3
    return dice_coef(y_true[..., 3:4], y_pred[..., 3:4])

def dice_loss(y_true, y_pred):
    """Computes Dice loss."""
    return 1 - dice_coef(y_true, y_pred)

def combined_loss(y_true, y_pred):
    """Combines Dice loss and Categorical Cross-Entropy."""
    return dice_loss(y_true, y_pred) + tf.keras.losses.categorical_crossentropy(y_true, y_pred)

def iou(y_true, y_pred, smooth=1e-6):
    """Computes the Intersection over Union (IoU) or Jaccard Index."""
    y_true_f = tf.keras.backend.flatten(y_true[..., 1:])
    y_pred_f = tf.keras.backend.flatten(y_pred[..., 1:])
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    union = tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)

def precision(y_true, y_pred):
    """Computes precision."""
    true_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true[..., 1:] * y_pred[..., 1:], 0, 1)))
    predicted_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_pred[..., 1:], 0, 1)))
    return true_positives / (predicted_positives + tf.keras.backend.epsilon())

def recall(y_true, y_pred):
    """Computes recall."""
    true_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true[..., 1:] * y_pred[..., 1:], 0, 1)))
    possible_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true[..., 1:], 0, 1)))
    return true_positives / (possible_positives + tf.keras.backend.epsilon())

def sensitivity(y_true, y_pred):
    """Computes sensitivity (same as recall)."""
    return recall(y_true, y_pred)

def specificity(y_true, y_pred):
    """Computes specificity."""
    y_true_nobg = y_true[..., 1:]
    y_pred_nobg = y_pred[..., 1:]

    neg_y_true = 1 - y_true_nobg
    neg_y_pred = 1 - y_pred_nobg
    
    fp = tf.keras.backend.sum(neg_y_true * y_pred_nobg)
    tn = tf.keras.backend.sum(neg_y_true * neg_y_pred)
    
    specificity_val = tn / (tn + fp + tf.keras.backend.epsilon())
    return specificity_val

def calculate_hausdorff(y_true, y_pred):
    """
    Calculates the 95th percentile Hausdorff distance.
    Expects numpy arrays.
    """
    y_true_class = np.argmax(y_true, axis=-1)
    y_pred_class = np.argmax(y_pred, axis=-1)

    distances = []
    num_classes = y_true.shape[-1]
    for i in range(1, num_classes):
        true_mask = (y_true_class == i)
        pred_mask = (y_pred_class == i)
        if np.sum(true_mask) > 0 and np.sum(pred_mask) > 0:
            hd = hd95(pred_mask, true_mask)
            distances.append(hd)
        else:
            distances.append(np.nan)
    return np.nanmean(distances)
