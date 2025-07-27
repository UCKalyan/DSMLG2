import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np
from medpy.metric.binary import hd95


# =============================================================================
# CORE METRIC FUNCTIONS (NUMERICALLY STABLE & MIXED-PRECISION SAFE)
# =============================================================================

def dice_coef(y_true, y_pred, smooth=1e-6):
    """
    Computes a robust Dice coefficient for ALL FOREGROUND classes combined.
    This is effectively the same as dice_coef_wt.
    """
    y_true = tf.cast(y_true, y_pred.dtype)
    
    y_true_f = K.flatten(K.sum(y_true[..., 1:], axis=-1))
    y_pred_f = K.flatten(K.sum(y_pred[..., 1:], axis=-1))
    
    intersection = K.sum(y_true_f * y_pred_f)
    denominator = K.sum(y_true_f) + K.sum(y_pred_f)
    
    return (2. * intersection + smooth) / (denominator + smooth + K.epsilon())

def iou(y_true, y_pred, smooth=1e-6):
    """Computes Intersection over Union for all foreground classes."""
    y_true = tf.cast(y_true, y_pred.dtype)
    y_true_f = K.flatten(y_true[..., 1:])
    y_pred_f = K.flatten(y_pred[..., 1:])
    
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f) + K.sum(y_pred_f) - intersection
    
    return (intersection + smooth) / (union + smooth + K.epsilon())

def precision(y_true, y_pred):
    """Computes precision for all foreground classes."""
    y_true = tf.cast(y_true, y_pred.dtype)
    true_positives = K.sum(K.round(K.clip(y_true[..., 1:] * y_pred[..., 1:], 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred[..., 1:], 0, 1)))
    
    return true_positives / (predicted_positives + K.epsilon())

def sensitivity(y_true, y_pred): # Also known as recall
    """Computes sensitivity for all foreground classes."""
    y_true = tf.cast(y_true, y_pred.dtype)
    true_positives = K.sum(K.round(K.clip(y_true[..., 1:] * y_pred[..., 1:], 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true[..., 1:], 0, 1)))
    
    return true_positives / (possible_positives + K.epsilon())

# metrics.py

def specificity(y_true, y_pred):
    """Computes a numerically stable specificity."""
    y_true = tf.cast(y_true, y_pred.dtype)
    
    true_negatives = K.sum(K.round(K.clip(y_true[..., 0] * y_pred[..., 0], 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(y_true[..., 0], 0, 1)))
    
    return true_negatives / (possible_negatives + K.epsilon())



# =============================================================================
# BRATS-SPECIFIC METRICS (STABILIZED)
# =============================================================================

def dice_coef_wt(y_true, y_pred, smooth=1e-6):
    """Dice coefficient for the Whole Tumor (WT)."""
    y_true = tf.cast(y_true, y_pred.dtype)
    y_true_wt = K.sum(y_true[..., 1:], axis=-1)
    y_pred_wt = K.sum(y_pred[..., 1:], axis=-1)

    y_true_f = K.flatten(y_true_wt)
    y_pred_f = K.flatten(y_pred_wt)
    
    intersection = K.sum(y_true_f * y_pred_f)
    denominator = K.sum(y_true_f) + K.sum(y_pred_f)
    
    return (2. * intersection + smooth) / (denominator + smooth + K.epsilon())

def dice_coef_tc(y_true, y_pred, smooth=1e-6):
    """Dice coefficient for the Tumor Core (TC): NCR/NET (1) + ET (3)."""
    y_true = tf.cast(y_true, y_pred.dtype)
    y_true_tc = y_true[..., 1] + y_true[..., 3]
    y_pred_tc = y_pred[..., 1] + y_pred[..., 3]
    
    y_true_f = K.flatten(y_true_tc)
    y_pred_f = K.flatten(y_pred_tc)

    intersection = K.sum(y_true_f * y_pred_f)
    denominator = K.sum(y_true_f) + K.sum(y_pred_f)
    
    return (2. * intersection + smooth) / (denominator + smooth + K.epsilon())

def dice_coef_et(y_true, y_pred, smooth=1e-6):
    """Dice coefficient for the Enhancing Tumor (ET): ET (3)."""
    y_true = tf.cast(y_true, y_pred.dtype)
    y_true_f = K.flatten(y_true[..., 3])
    y_pred_f = K.flatten(y_pred[..., 3])

    intersection = K.sum(y_true_f * y_pred_f)
    denominator = K.sum(y_true_f) + K.sum(y_pred_f)
    
    return (2. * intersection + smooth) / (denominator + smooth + K.epsilon())


# =============================================================================
# LOSS FUNCTIONS
# =============================================================================

def weighted_dice_loss(y_true, y_pred, class_weights):
    """Calculates weighted Dice loss for each class channel."""
    y_true = tf.cast(y_true, y_pred.dtype)
    loss = 0.0
    num_classes = y_pred.shape[-1]
    
    for i in range(num_classes):
        # Use the internal, stable dice calculation for each channel
        y_true_channel = K.flatten(y_true[..., i])
        y_pred_channel = K.flatten(y_pred[..., i])
        intersection = K.sum(y_true_channel * y_pred_channel)
        denominator = K.sum(y_true_channel) + K.sum(y_pred_channel)
        dice = (2. * intersection + 1e-6) / (denominator + 1e-6 + K.epsilon())
        loss += (1.0 - dice) * class_weights[i]
        
    return loss / tf.cast(num_classes, tf.float32)

def combined_weighted_loss(class_weights):
    """Returns a combined loss function that uses the provided weights."""
    weights = tf.constant(class_weights, dtype=tf.float32)

    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, y_pred.dtype)
        focal_loss = tf.keras.losses.CategoricalFocalCrossentropy(from_logits=False)(y_true, y_pred)
        dice_loss = weighted_dice_loss(y_true, y_pred, weights)
        return focal_loss + dice_loss
        
    return loss


# =============================================================================
# UTILITY FOR LOADING MODELS
# =============================================================================

def get_custom_objects():
    """Returns a dictionary of all custom functions for loading Keras models."""
    return {
        #"combined_weighted_loss": combined_weighted_loss,
        "create_w_t_e_loss": create_w_t_e_loss,
        "dice_coef": dice_coef,
        "iou": iou,
        "precision": precision,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "dice_coef_wt": dice_coef_wt,
        "dice_coef_tc": dice_coef_tc,
        "dice_coef_et": dice_coef_et,
        "loss": create_w_t_e_loss

    }

# =============================================================================
# LOSS FUNCTIONS (REFINED)
# =============================================================================

def create_dice_focal_loss(class_weights, focal_gamma=2.0, loss_factor=0.5):
    """
    Creates a combined loss function that balances weighted Dice loss and Focal loss.
    This version uses a vectorized Dice loss to avoid TensorFlow graph errors.
    """
    class_weights_tensor = tf.constant(class_weights, dtype=tf.float32)

    # 💡 FIX: This function is now vectorized to avoid Python loops.
    def weighted_dice_loss(y_true, y_pred):
        """
        Calculates a vectorized, weighted Dice loss.
        """
        y_true = tf.cast(y_true, y_pred.dtype)
        smooth = 1e-6

        # Flatten the tensors and keep the class dimension
        # Shape becomes (batch_size * H * W, num_classes)
        y_true_f = K.reshape(y_true, (-1, tf.shape(y_true)[-1]))
        y_pred_f = K.reshape(y_pred, (-1, tf.shape(y_pred)[-1]))

        # Calculate intersection and union for each class, vectorized
        intersection = K.sum(y_true_f * y_pred_f, axis=0)
        denominator = K.sum(y_true_f, axis=0) + K.sum(y_pred_f, axis=0)

        # Calculate the Dice score for each class
        dice_per_class = (2. * intersection + smooth) / (denominator + smooth)

        # Apply the weights to the loss for each class
        loss_per_class = (1.0 - dice_per_class) * class_weights_tensor

        # Return the mean of the per-class losses
        return K.mean(loss_per_class)

    def loss(y_true, y_pred):
        """The combined loss function."""
        # Calculate Weighted Dice Loss using the new vectorized function
        dice_loss = weighted_dice_loss(y_true, y_pred)
        
        # Calculate weighted Focal Loss (from previous fix)
        y_true_focal = tf.cast(y_true, y_pred.dtype)
        focal_loss_fn = tf.keras.losses.CategoricalFocalCrossentropy(
            gamma=focal_gamma, 
            from_logits=False,
            reduction=tf.keras.losses.Reduction.NONE
        )
        per_pixel_focal_loss = focal_loss_fn(y_true_focal, y_pred)
        weights_map = tf.reduce_sum(y_true_focal * class_weights_tensor, axis=-1)
        weighted_focal_loss = per_pixel_focal_loss * weights_map
        focal_loss = tf.reduce_mean(weighted_focal_loss)
        
        # Combine the losses
        combined_loss = (loss_factor * dice_loss) + ((1 - loss_factor) * focal_loss)
        return combined_loss
        
    return loss

def hd95_metric(y_true, y_pred, spacing=(1.0, 1.0, 1.0)):
    """
    Calculates the 95th percentile Hausdorff Distance on 3D volumes.
    Expects batched, one-hot encoded numpy arrays.
    """
    y_true_np = y_true
    y_pred_np = y_pred

    hd_scores = []
    
    # Iterate over the batch (usually size 1 from the evaluator)
    for i in range(y_true_np.shape[0]):
        # Iterate through foreground classes (1, 2, 3), skipping background (0)
        for class_idx in range(1, y_true_np.shape[-1]):
            gt_mask = y_true_np[i, ..., class_idx].astype(bool)
            pred_mask = (y_pred_np[i, ..., class_idx] >= 0.5).astype(bool)

            # If both ground truth and prediction are empty for this class, skip
            if not np.any(gt_mask) and not np.any(pred_mask):
                continue
            
            # If one is empty but not the other, HD95 is undefined. Append NaN.
            if not np.any(gt_mask) or not np.any(pred_mask):
                hd_scores.append(np.nan)
                continue

            # Calculate HD95 for the 3D volume of this class
            try:
                # Ensure the mask is 3D, which it should be from the evaluator
                if gt_mask.ndim == 3:
                    # medpy expects (prediction, ground_truth)
                    hd95_val = hd95(pred_mask, gt_mask, voxelspacing=spacing)
                    
                    if not np.isinf(hd95_val):
                        hd_scores.append(hd95_val)
                    else:
                        hd_scores.append(np.nan) # Handle infinite values
                else:
                    # This case should not be reached if evaluator.py is working correctly
                    hd_scores.append(np.nan)
            except Exception:
                hd_scores.append(np.nan)

    # Return the average HD95 score, ignoring any NaNs
    if not hd_scores:
        return np.nan
    return np.nanmean(hd_scores)

# In training/metrics.py, add this function in the LOSS FUNCTIONS section

# In training/metrics.py, replace the old function with this one

def create_weighted_categorical_crossentropy(class_weights):
    """
    Creates a correctly implemented weighted categorical cross-entropy loss function.

    This function calculates the standard cross-entropy loss for each pixel and then
    multiplies that loss by the weight corresponding to the pixel's true class.

    Args:
        class_weights (list or tuple): A list of weights for each class.
                                       Example: [w_bg, w_class1, w_class2, w_class3]

    Returns:
        A callable loss function.
    """
    weights = tf.constant(class_weights, dtype=tf.float32)

    def loss(y_true, y_pred):
        """The actual loss calculation."""
        # Ensure y_true is the same data type as y_pred
        y_true = tf.cast(y_true, y_pred.dtype)

        # Calculate the standard categorical cross-entropy loss without reduction
        cce = tf.keras.losses.categorical_crossentropy(y_true, y_pred, from_logits=False)

        # Create a weight map by grabbing the weight for the true class of each pixel.
        # tf.reduce_sum(y_true * weights, axis=-1) effectively selects the weight
        # for the '1' in the one-hot encoded y_true vector.
        weight_map = tf.reduce_sum(y_true * weights, axis=-1)

        # Multiply the cross-entropy loss by the corresponding class weight.
        weighted_cce = cce * weight_map

        # Return the mean of the weighted loss.
        return tf.reduce_mean(weighted_cce)

    return loss

def create_w_t_e_loss1(class_weights, wt_w=0.33, tc_w=0.33, et_w=0.33):
    """
    Creates a loss function that is a weighted sum of Dice losses
    for Whole Tumor (WT), Tumor Core (TC), and Enhancing Tumor (ET).
    """
    
    def loss(y_true, y_pred):
        # Calculate Dice loss for each component
        # The (1 - dice_coef) turns the similarity metric into a loss to be minimized
        loss_wt = (1 - dice_coef_wt(y_true, y_pred)) * wt_w
        loss_tc = (1 - dice_coef_tc(y_true, y_pred)) * tc_w
        loss_et = (1 - dice_coef_et(y_true, y_pred)) * et_w
        
        return loss_wt + loss_tc + loss_et

    return loss

# In metrics.py, modify this function
def create_w_t_e_loss(class_weights, wt_w=0.25, tc_w=0.25, et_w=0.5): # Note the new default weights
    """
    Creates a loss function that is a weighted sum of Dice losses
    for Whole Tumor (WT), Tumor Core (TC), and Enhancing Tumor (ET).
    """
    
    def loss(y_true, y_pred):
        # Calculate Dice loss for each component
        # The (1 - dice_coef) turns the similarity metric into a loss to be minimized
        loss_wt = (1 - dice_coef_wt(y_true, y_pred)) * wt_w
        loss_tc = (1 - dice_coef_tc(y_true, y_pred)) * tc_w
        loss_et = (1 - dice_coef_et(y_true, y_pred)) * et_w
        
        return loss_wt + loss_tc + loss_et

    return loss