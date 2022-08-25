'''
    IMPORT LIBRARIES
'''
import albumentations as A
import argparse
import cv2
import gc
import glob
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
from sklearn.model_selection import StratifiedKFold
import os

import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import *
from tensorflow.keras.losses import *
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *

from utils.metrics import *
from utils.dataloader import DataLoader, SegFormerDataLoader, SSDataLoader
from utils.inference import CompetitionMetric
from utils.utils import *
from models.trans import TranSeg
import segmentation_models as sm

from transformers import TFSegformerForSemanticSegmentation, SegformerConfig

'''
    PARSE ARGUMENTS
'''
parser = argparse.ArgumentParser()
parser.add_argument("--backbone", type=str, help="Unet backbone", default="unet2d_ds")
parser.add_argument("--description", type=str, help="Model description", default="resize_baseline")
parser.add_argument("--batch", type=int, help="Batch size", default=16)
parser.add_argument("--datafolder", type=str, help="Data folder", default='data/npy_768x768')
parser.add_argument("--seed", type=int, help="Seed for random generator", default=2022)
parser.add_argument("--csv", type=str, help="Dataframe path", default='data/train.csv')
parser.add_argument("--trainsize", type=str, help="Training image size", default="768x768")
parser.add_argument("--fold", type=int, help="Number of folds", default=5)
parser.add_argument("--epoch", type=int, help="Number of epochs", default=100)
parser.add_argument("--mode", type=str, help="CV or full training", default="CV")
args = parser.parse_args()

'''
    CONFIGURATION
'''
RANDOM_SEED = args.seed
DATAFOLDER = args.datafolder
DF = pd.read_csv(args.csv)

MODEL_CHECKPOINTS_FOLDER = './model_checkpoint/'
MODEL_NAME = args.backbone
MODEL_DESC = args.description

TRAINING_SIZE = tuple([int(x) for x in args.trainsize.split("x")])
BATCH_SIZE = args.batch
KFOLD = args.fold
NUM_CLASSES = 3

MODE = args.mode

TRANSFORM = A.Compose([
    A.ShiftScaleRotate(shift_limit=0.0, scale_limit=0.5, rotate_limit=180, p=1.,
                        border_mode=cv2.BORDER_REFLECT),
    A.Cutout(num_holes=8, max_h_size=64, max_w_size=64, fill_value=0., always_apply=False, p=1.),
    A.OneOf([
        A.ElasticTransform(p=.3),
        A.GaussianBlur(p=.3),
        A.GaussNoise(p=.3),
        A.OpticalDistortion(p=0.3),
        A.GridDistortion(p=.1),
    ], p=1.),
], p=1.0)

initial_lr = 1e-3
min_lr = 1e-5
no_of_epochs = args.epoch

def augment(X):
    transformed = TRANSFORM(image=X)
    X = transformed["image"]
    return X

'''
    MAIN PROGRAM
'''
if __name__ == "__main__":
    # REPRODUCIBILITY
    seed_everything(RANDOM_SEED)

    print(f"Model name: {MODEL_NAME}. Description: {MODEL_DESC}")

    # Create folder for model checkpoints
    if not os.path.exists(f"{MODEL_CHECKPOINTS_FOLDER}/{MODEL_NAME}"):
        os.makedirs(f"{MODEL_CHECKPOINTS_FOLDER}/{MODEL_NAME}")
        
    if not os.path.exists(f"plot/{MODEL_NAME}"):
        os.makedirs(f"plot/{MODEL_NAME}")

    # SPLIT DATA INTO KFOLD
    skf = StratifiedKFold(n_splits=KFOLD)
    for fold, (_, val_idx) in enumerate(skf.split(X=DF, y=DF['organ']), 1):
        DF.loc[val_idx, 'fold'] = fold

    # This stores training and validation history
    hists = []

    # TRAINING FOR KFOLD
    for fold in range(1, KFOLD + 1):
        # Clear sessions and collect garbages
        K.clear_session()
        gc.collect()

        dataloader = SSDataLoader

        monitor = 'loss'

        # Enable XLA for performance
        tf.config.optimizer.set_jit(True)

        optimizer = Adam(learning_rate=initial_lr)


        model = TranSeg(num_classes = NUM_CLASSES, 
                        input_shape = (None, None, 3),
                        encoder_num_heads=[1, 2, 5, 8],
                        encoder_dims=[64, 128, 320, 512],
                        encoder_depth=[2, 2, 2, 2],
                        hidden_dropout=0.0,
                        attention_dropout=0.0,
                        drop_path=0.0,
                        decoder_dim=768,
                        patch_size = [7, 3, 3, 3],
                        stride = [4, 2, 2, 2],
                        sr = [8, 4, 2, 1],
                        activation = 'sigmoid')()   

        loss = SelfSupervisedLoss()
            
        model.compile(optimizer=optimizer, 
                      loss=loss)
        
        train_datagen = dataloader(DF["id"].values, DATAFOLDER, batch_size=BATCH_SIZE, shuffle=True, augment=augment)

        callbacks = [
            ModelCheckpoint(f'{MODEL_CHECKPOINTS_FOLDER}/{MODEL_NAME}/{MODEL_DESC}.h5', save_best_only=False, save_weights_only=True, verbose=1),
            LearningRateScheduler(schedule=poly_scheduler(initial_lr, min_lr, no_of_epochs), verbose=1),
            CSVLogger(f'{MODEL_CHECKPOINTS_FOLDER}/{MODEL_NAME}/{MODEL_DESC}_fold{fold}.csv', separator=",", append=False)
        ]
        hist = model.fit_generator(train_datagen, 
                                epochs=no_of_epochs, 
                                callbacks = callbacks,
                                verbose=2)
        hists.append(hist.history)
        break

    print("Done!")