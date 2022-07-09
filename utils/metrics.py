import math

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.losses import *
from tensorflow.keras.callbacks import Callback
'''
    METRICS AND LOSS FUNCTIONS
'''
def Dice_Coef(spartial_axis = (1,2), smooth=1e-6):
    def Dice_Coef(y_true, y_pred):
        y_pred = tf.math.round(y_pred)
        tp = tf.math.reduce_sum(y_true * y_pred, axis=spartial_axis) # calculate True Positive
        fn = tf.math.reduce_sum(y_true * (1 - y_pred), axis=spartial_axis) # calculate False Negative
        fp = tf.math.reduce_sum((1 - y_true) * y_pred, axis=spartial_axis) # calculate False Positive 
        numerator = 2 * tp
        denominator = 2 * tp + fn + fp
        return tf.math.reduce_mean( (numerator + smooth) / (denominator + smooth) )
    return Dice_Coef

def dice_loss(spartial_axis=(0,1,2), smooth = 1e-6):
    def dice_loss(y_true, y_pred):
        tp = tf.math.reduce_sum(y_true * y_pred, axis=spartial_axis) # calculate True Positive
        fn = tf.math.reduce_sum(y_true * (1 - y_pred), axis=spartial_axis) # calculate False Negative
        fp = tf.math.reduce_sum((1 - y_true) * y_pred, axis=spartial_axis) # calculate False Positive
        numerator = 2 * tp + smooth
        denominator = 2 * tp + fn + fp + smooth
        return tf.math.reduce_mean(1 - numerator / denominator) # Average over classes
    return dice_loss

def bce_dice_loss(spartial_axis=(0,1,2), smooth = 1e-6):
    dice_loss_func = dice_loss(spartial_axis, smooth)
    def loss(y_true, y_pred):
        return binary_crossentropy(y_true, y_pred) + dice_loss_func(y_true, y_pred)
    return loss