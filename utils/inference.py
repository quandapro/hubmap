import numpy as np
from scipy.ndimage.filters import gaussian_filter
import tqdm

from tensorflow.keras.callbacks import Callback
import tensorflow as tf

def dice_coef_numpy(y_true, y_pred, smooth = 1e-6):
    tp = np.sum(y_true * y_pred) # calculate True Positive
    fn = np.sum(y_true * (1 - y_pred)) # calculate False Negative
    fp = np.sum((1 - y_true) * y_pred) # calculate False Positive
    numerator = 2 * tp + smooth
    denominator = 2 * tp + fn + fp + smooth
    return np.mean(numerator / denominator)

from timeit import default_timer as timer

class CompetitionMetric(Callback):
    def __init__(self, validation_data, model_checkpoint, num_class = 1, period = 1, deep_supervision = True, patch_size = (512, 512)):
        super(Callback, self).__init__()
        
        self.validation_data = validation_data
        self.model_checkpoint = model_checkpoint
        self.best_validation_score = -np.inf
        self.deep_supervision = deep_supervision
        self.period = period
        self.num_class = num_class
        self.patch_size = patch_size
        self.stride = tuple([x // 2 for x in patch_size])
        self.gaussian_importance_map = self._get_gaussian(patch_size, num_class, sigma_scale = 1. / 8)

    def _get_gaussian(self, patch_size, num_class, sigma_scale):
        tmp = np.zeros(patch_size)
        center_coords = [i // 2 for i in patch_size]
        sigmas = [i * sigma_scale for i in patch_size]
        tmp[tuple(center_coords)] = 1
        gaussian_importance_map = gaussian_filter(tmp, sigmas, 0, mode='constant', cval=0)
        gaussian_importance_map = gaussian_importance_map / np.max(gaussian_importance_map) * 1
        gaussian_importance_map = gaussian_importance_map.astype(np.float32)

        # gaussian_importance_map cannot be 0, otherwise we may end up with nans!
        gaussian_importance_map[gaussian_importance_map == 0] = np.min(
            gaussian_importance_map[gaussian_importance_map != 0])

        result = np.zeros((*gaussian_importance_map.shape, num_class))
        for i in range(num_class):
            result[..., i] = gaussian_importance_map

        return result

        
    def sliding_window_inference(self, volume):
        '''
            Sliding window inference
            --------
            volume : numpy.ndarray
                Input 2D volume with shape = (1, Height, Width, 3)
            --------
            return : numpy.ndarray
                Output segmentation
        '''
        h, w = volume.shape[1:3]
        w_h, w_w = self.patch_size
        s_h, s_w = self.stride
        result = np.zeros((*volume.shape[:-1], self.num_class), dtype='float32')
        overlap = np.zeros((*volume.shape[:-1], self.num_class), dtype='float32')
        starting_points = [(x, y)  for x in set( list(range(0, h - w_h, s_h)) + [h - w_h] ) 
                                   for y in set( list(range(0, w - w_w, s_w)) + [w - w_w] )]

        patches = np.empty((len(starting_points), *self.patch_size, volume.shape[-1]), dtype='float32')
        for i, (x, y) in enumerate(starting_points):
            patches[i] = volume[:, x:x + w_h, y:y + w_w, :]

        y_pred = self.model.predict(patches, batch_size = 8)
        if self.deep_supervision:
            y_pred = y_pred[-1]
        
        for i in range(len(y_pred)):
            x, y = starting_points[i]
            result[:, x:x + w_h, y:y + w_w, :] += y_pred[i] * self.gaussian_importance_map
            overlap[:, x:x + w_h, y:y + w_w, :] += self.gaussian_importance_map

        assert np.sum(overlap == 0.) == 0, "Sliding window does not cover all volume"

        return result / overlap

    def on_epoch_end(self, epoch, logs={}):
        if (epoch + 1) % self.period == 0:
            start = timer()

            dice_coef = []
            for X, y_true in self.validation_data:
                non_empty = np.sum(y_true, axis=(0, 1, 2))
                non_empty = np.argmax(non_empty)
                y_pred = self.sliding_window_inference(X)
                # Thresholding
                y_pred = (y_pred > 0.5).astype('float32')
                dice_coef.append(dice_coef_numpy(y_true[..., non_empty], y_pred[..., non_empty]))

            mean_dice_coef = np.mean(dice_coef)

            print(f"val_dice_coef: {mean_dice_coef}")
            if mean_dice_coef > self.best_validation_score:
                print(f"Validation score improved from {self.best_validation_score} to {mean_dice_coef}. Saving model to {self.model_checkpoint}")
                self.model.save_weights(self.model_checkpoint)
                self.best_validation_score = mean_dice_coef
            else:
                print(f"Validation score does not improve from: {self.best_validation_score}")
            end = timer()
            print(f"Finished in {end - start}s")