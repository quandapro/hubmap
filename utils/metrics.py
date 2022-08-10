import math

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.losses import *
from tensorflow.keras.callbacks import Callback
'''
    METRICS AND LOSS FUNCTIONS
'''

class DiceCoef:
    def __init__(self, spartial_axis = (1,2), class_axis = None, from_logits = False, index = False, smooth=1e-6, name='Dice_Coef'):
        self.__name__ = name
        self.spartial_axis = spartial_axis
        self.class_axis = class_axis
        self.from_logits = from_logits
        self.index = index
        self.smooth = smooth
        
    def __call__(self, y_true, y_pred):
        if self.from_logits: # Apply sigmoid if from logits
            y_pred = K.sigmoid(y_pred)
            y_pred = tf.transpose(y_pred, perm=[0, 2, 3, 1])
        
        if self.index: # Convert to one-hot encoding
            y_true = tf.cast(y_true, dtype=tf.uint8)
            y_true = tf.one_hot(y_true, tf.shape(y_pred)[-1], dtype=tf.float32)

        y_pred = tf.math.round(y_pred)

        y_pred = tf.image.resize(y_pred, size=tf.shape(y_true)[1:3])

        # Apply class axis
        if self.class_axis is not None:
            y_true = y_true[..., self.class_axis]
            y_pred = y_pred[..., self.class_axis]

        # Compute dice coef
        tp = tf.math.reduce_sum(y_true * y_pred, axis=self.spartial_axis) # calculate True Positive
        fn = tf.math.reduce_sum(y_true * (1 - y_pred), axis=self.spartial_axis) # calculate False Negative
        fp = tf.math.reduce_sum((1 - y_true) * y_pred, axis=self.spartial_axis) # calculate False Positive 
        non_empty = tf.math.count_nonzero(tf.math.reduce_sum(y_true, axis=self.spartial_axis), dtype=tf.float32) # Ignore if y_true is empty
        numerator = 2 * tp
        denominator = 2 * tp + fn + fp
        return tf.math.reduce_sum( numerator / (denominator + self.smooth) ) / (non_empty + self.smooth)

class BceDiceLoss:
    def __init__(self, spartial_axis = (1,2), class_axis = None, from_logits = False, index = False, smooth=1e-6, name='loss'):
        self.__name__ = name
        self.spartial_axis = spartial_axis
        self.class_axis = class_axis
        self.from_logits = from_logits
        self.index = index
        self.smooth = smooth
        
    def __call__(self, y_true, y_pred):
        if self.from_logits: # Apply sigmoid if from logits
            y_pred = K.sigmoid(y_pred)
            y_pred = tf.transpose(y_pred, perm=[0, 2, 3, 1])

        if self.index: # Convert to one-hot encoding
            y_true = tf.cast(y_true, dtype=tf.uint8)
            y_true = tf.one_hot(y_true, tf.shape(y_pred)[-1], dtype=tf.float32)
        
        y_pred = tf.image.resize(y_pred, size=tf.shape(y_true)[1:3])

        # Apply class axis
        if self.class_axis is not None:
            y_true = y_true[..., self.class_axis]
            y_pred = y_pred[..., self.class_axis]

        # Compute dice coef
        tp = tf.math.reduce_sum(y_true * y_pred, axis=self.spartial_axis) # calculate True Positive
        fn = tf.math.reduce_sum(y_true * (1 - y_pred), axis=self.spartial_axis) # calculate False Negative
        fp = tf.math.reduce_sum((1 - y_true) * y_pred, axis=self.spartial_axis) # calculate False Positive 
        numerator = 2 * tp
        denominator = 2 * tp + fn + fp

        # Compute dice loss
        dice_loss = 1. - tf.math.reduce_mean( (numerator + self.smooth) / (denominator + self.smooth) )

        # Compute bce
        bce_loss = binary_crossentropy(y_true, y_pred)

        return bce_loss + dice_loss

class CceDiceLoss:
    def __init__(self, spartial_axis = (1,2), class_axis = None, from_logits = False, index = False, smooth=1e-6, name='loss'):
        self.__name__ = name
        self.spartial_axis = spartial_axis
        self.class_axis = class_axis
        self.from_logits = from_logits
        self.index = index
        self.smooth = smooth
        
    def __call__(self, y_true, y_pred):
        if self.from_logits: # Apply softmax if from logits
            y_pred = K.softmax(y_pred)
            y_pred = tf.transpose(y_pred, perm=[0, 2, 3, 1])

        if self.index: # Convert to one-hot encoding
            y_true = tf.cast(y_true, dtype=tf.uint8)
            y_true = tf.one_hot(y_true, tf.shape(y_pred)[-1], dtype=tf.float32)
        
        y_pred = tf.image.resize(y_pred, size=tf.shape(y_true)[1:3])

        # Apply class axis
        if self.class_axis is not None:
            y_true = y_true[..., self.class_axis]
            y_pred = y_pred[..., self.class_axis]

        # Compute dice coef
        tp = tf.math.reduce_sum(y_true * y_pred, axis=self.spartial_axis) # calculate True Positive
        fn = tf.math.reduce_sum(y_true * (1 - y_pred), axis=self.spartial_axis) # calculate False Negative
        fp = tf.math.reduce_sum((1 - y_true) * y_pred, axis=self.spartial_axis) # calculate False Positive 
        numerator = 2 * tp
        denominator = 2 * tp + fn + fp

        # Compute dice loss
        dice_loss = 1. - tf.math.reduce_mean( (numerator + self.smooth) / (denominator + self.smooth) )

        # Compute cce
        cce_loss = categorical_crossentropy(y_true, y_pred)

        return cce_loss + dice_loss
        