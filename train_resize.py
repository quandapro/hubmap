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
from utils.dataloader import DataLoader, SegFormerDataLoader
from utils.inference import CompetitionMetric
from utils.utils import *
from models.unet import ResUnet2D, Unet2D
from models.custom_model import CustomModel
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
NUM_CLASSES = 1

MODE = args.mode

TRANSFORM = A.Compose([
    A.HorizontalFlip(),
    A.VerticalFlip(),
    A.RandomRotate90(),
    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.5, rotate_limit=180, p=0.9,
                        border_mode=cv2.BORDER_REFLECT),
    A.OneOf([
        A.HueSaturationValue(15,25,0),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3),
        A.RandomGamma(gamma_limit=(70, 150)),
        A.ChannelShuffle()
    ], p=1.),
    A.OneOf([
        A.ElasticTransform(p=.3),
        A.GaussianBlur(p=.3),
        A.GaussNoise(p=.3),
        A.OpticalDistortion(p=0.3),
        A.GridDistortion(p=.1),
    ], p=1.),
], p=1.0)

VALID_TRANSFORM = A.Compose([

])

CLASS_MAP = {'kidney' : 1,
             'largeintestine' : 2,
             'lung' : 3,
             'prostate' : 4,
             'spleen' : 5}

initial_lr = 3e-4
min_lr = 1e-4
no_of_epochs = args.epoch

def augment(X, y):
    transformed = TRANSFORM(image=X, mask=y)
    X = transformed["image"]
    y = transformed["mask"]
    return X, y

def valid_augment(X, y):
    transformed = VALID_TRANSFORM(image=X, mask=y)
    X = transformed["image"]
    y = transformed["mask"]
    return X, y

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

    DF["class"] = [CLASS_MAP[x] for x in DF["organ"].values]

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

        dataloader = DataLoader

        monitor = 'val_Dice_Coef'

        # Enable XLA for performance
        tf.config.optimizer.set_jit(True)

        optimizer = Adam(learning_rate=initial_lr)

        from_logits = False    
        index = False
        if MODEL_NAME == "segformer":
            model = TFSegformerForSemanticSegmentation.from_pretrained(
                "nvidia/mit-b3",
                num_labels=NUM_CLASSES + 1,
                ignore_mismatched_sizes=True
                )
            dataloader = SegFormerDataLoader
            from_logits = True
            index = True
                        
        else:
            model = sm.Unet(MODEL_NAME, input_shape=(None, None, 3), classes=NUM_CLASSES + 1, encoder_weights='imagenet', activation='sigmoid')
            monitor = 'val_Dice_Coef'

        loss = BceDiceLoss(spartial_axis=(1,2), class_axis=slice(0, NUM_CLASSES + 1), from_logits=from_logits, index = index)
        metrics = [DiceCoef(spartial_axis=(1,2), class_axis=slice(1, NUM_CLASSES + 1), from_logits=from_logits, index = index)]
            
        model.compile(optimizer=optimizer, 
                      loss=loss, 
                      metrics=metrics)
        
        if MODE == "CV":
            train_id, test_id = DF["id"][DF["fold"] != fold].values, DF["id"][DF["fold"] == fold].values

            # Train and test data generator
            train_datagen = dataloader(train_id, DATAFOLDER, batch_size=BATCH_SIZE, shuffle=True, augment=augment)
            test_datagen = dataloader(test_id, DATAFOLDER, batch_size=BATCH_SIZE, shuffle=False, augment=None)

            callbacks = [
                CustomModelCheckpoint(f'{MODEL_CHECKPOINTS_FOLDER}/{MODEL_NAME}/{MODEL_DESC}_fold{fold}.h5', monitor=monitor, is_segformer=(MODEL_NAME == 'segformer'), save_best_only=True),
                LearningRateScheduler(schedule=cosine_scheduler(initial_lr, min_lr, no_of_epochs), verbose=1),
                CSVLogger(f'{MODEL_CHECKPOINTS_FOLDER}/{MODEL_NAME}/{MODEL_DESC}_fold{fold}.csv', separator=",", append=False)
            ]
            hist = model.fit_generator(train_datagen, 
                                    epochs=no_of_epochs, 
                                    callbacks = callbacks,
                                    validation_data=test_datagen,
                                    verbose=2)
            hists.append(hist.history)
        else:
            train_datagen = dataloader(DF["id"].values, DATAFOLDER, batch_size=BATCH_SIZE, shuffle=True, augment=augment)

            callbacks = [
                CustomModelCheckpoint(f'{MODEL_CHECKPOINTS_FOLDER}/{MODEL_NAME}/{MODEL_DESC}.h5', monitor=monitor, is_segformer=(MODEL_NAME == 'segformer'), save_best_only=False),
                LearningRateScheduler(schedule=cosine_scheduler(initial_lr, min_lr, no_of_epochs), verbose=1),
                CSVLogger(f'{MODEL_CHECKPOINTS_FOLDER}/{MODEL_NAME}/{MODEL_DESC}_fold{fold}.csv', separator=",", append=False)
            ]
            hist = model.fit_generator(train_datagen, 
                                    epochs=no_of_epochs, 
                                    callbacks = callbacks,
                                    verbose=2)
            hists.append(hist.history)
            break

    if MODE == "CV":
        # PLOT TRAINING RESULTS
        val_Dice_Coef = []

        for i in range(1, KFOLD + 1):
            val_Dice_Coef.append(np.max(hists[i - 1][monitor]))

        print(val_Dice_Coef)
        print(f"{np.mean(val_Dice_Coef)} +- {np.std(val_Dice_Coef)}")
    print("Done!")