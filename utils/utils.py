import os
import numpy as np
import random
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import Callback
import matplotlib.pyplot as plt

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def poly_scheduler(initial_lr, min_lr, no_of_epochs, exponent = 0.9):
    def scheduler(epoch, lr):
        return max(initial_lr * (1 - epoch / no_of_epochs)**exponent, min_lr)
    return scheduler

def cosine_scheduler(initial_lr, min_lr, epochs_per_cycle):
    def scheduler(epoch, lr):
        return min_lr + (initial_lr - min_lr) * (1 + np.cos(np.pi * (epoch % epochs_per_cycle) / epochs_per_cycle)) / 2
    return scheduler

class CustomModelCheckpoint(Callback):
    def __init__(self, ckpt, is_segformer, monitor, save_best_only):
        self.ckpt = ckpt
        self.is_segformer = is_segformer
        self.best_score = -np.inf
        self.monitor = monitor
        self.save_best_only = save_best_only

    def on_epoch_end(self, epoch, logs={}):
        ckpt = self.ckpt
        if self.is_segformer:
            ckpt = self.ckpt.split('.h5')[0]

        if not self.save_best_only:
            print(f"Saving model to {ckpt}")
            if self.is_segformer:
                self.model.save_pretrained(ckpt)
            else:
                self.model.save_weights(ckpt)

        elif logs[self.monitor] > self.best_score:
            print(f"{self.monitor} improved from {self.best_score} to {logs[self.monitor]}. Saving model to {ckpt}")
            if self.is_segformer:
                self.model.save_pretrained(ckpt)
            else:
                self.model.save_weights(ckpt)
            self.best_score = logs[self.monitor]