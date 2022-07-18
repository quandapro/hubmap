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
from tensorflow.keras.optimizers import *
import segmentation_models as sm

from utils.metrics import *
from utils.dataloader import DataLoader
from utils.inference import CompetitionMetric
from utils.utils import *
from models.unet import Unet2D

'''
    PARSE ARGUMENTS
'''
parser = argparse.ArgumentParser()
parser.add_argument("--backbone", type=str, help="Unet backbone", default="unet2d_ds")
parser.add_argument("--description", type=str, help="Model description", default="baseline")
parser.add_argument("--batch", type=int, help="Batch size", default=16)
parser.add_argument("--datafolder", type=str, help="Data folder", default='data/npy')
parser.add_argument("--seed", type=int, help="Seed for random generator", default=2022)
parser.add_argument("--csv", type=str, help="Dataframe path", default='data/train.csv')
parser.add_argument("--trainsize", type=str, help="Training image size", default="512x512")
parser.add_argument("--fold", type=int, help="Number of folds", default=5)
parser.add_argument("--epoch", type=int, help="Number of epochs", default=1000)
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
NUM_CLASSES = 5

TRANSFORM = A.Compose([
    A.HorizontalFlip(),
    A.VerticalFlip(),
    A.RandomRotate90(),
    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=15, p=0.9,
                        border_mode=cv2.BORDER_REFLECT),
    A.OneOf([
        A.ElasticTransform(p=.3),
        A.GaussianBlur(p=.3),
        A.GaussNoise(p=.3),
        A.OpticalDistortion(p=0.3),
        A.GridDistortion(p=.1),
    ], p=0.3),
    A.OneOf([
        A.HueSaturationValue(15,25,0),
        A.CLAHE(clip_limit=2),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3),
    ], p=0.3)
], p=1.0)

initial_lr = 1e-2
min_lr = 1e-4
no_of_epochs = args.epoch

def augment(X, y):
    h, w = X.shape[:2]
    # Random Crop that enforces non-empty mask 
    TRIES = 10
    DONE = False
    Cropped_X = Cropped_y = None
    for i in range(TRIES):
        start_x = np.random.randint(0, h - TRAINING_SIZE[0])
        start_y = np.random.randint(0, w - TRAINING_SIZE[0])

        Cropped_X = X[start_x:start_x + TRAINING_SIZE[0], start_y:start_y + TRAINING_SIZE[1]]
        Cropped_y = y[start_x:start_x + TRAINING_SIZE[0], start_y:start_y + TRAINING_SIZE[1]]

        # If mask is non-empty, we are done
        if np.sum(Cropped_y) > 0:
            break

    # Transform
    transformed = TRANSFORM(image=Cropped_X, mask=Cropped_y)
    Trans_X = transformed["image"]
    Trans_y = transformed["mask"]
    return Trans_X, Trans_y

'''
    MAIN PROGRAM
'''
if __name__ == "__main__":
    # REPRODUCIBILITY
    seed_everything(RANDOM_SEED)

    print(f"Model name: {MODEL_NAME}. Description: {MODEL_DESC}")
    class_map = ["large_bowel", "small_bowel", "stomach"]

    if not os.path.exists(f"{MODEL_CHECKPOINTS_FOLDER}/{MODEL_NAME}"):
        os.makedirs(f"{MODEL_CHECKPOINTS_FOLDER}/{MODEL_NAME}")
        
    if not os.path.exists(f"plot/{MODEL_NAME}"):
        os.makedirs(f"plot/{MODEL_NAME}")

    # SPLIT DATA INTO KFOLD
    skf = StratifiedKFold(n_splits=KFOLD)
    for fold, (_, val_idx) in enumerate(skf.split(X=DF, y=DF['organ']), 1):
        DF.loc[val_idx, 'fold'] = fold

    hists = []

    # TRAINING FOR KFOLD
    for fold in range(1, KFOLD + 1):
        # Clear sessions and collect garbages
        K.clear_session()
        gc.collect()

        # Enable XLA for performance
        tf.config.optimizer.set_jit(True)

        train_id, test_id = DF["id"][DF["fold"] != fold].values, DF["id"][DF["fold"] == fold].values

        train_datagen = DataLoader(train_id, DATAFOLDER, batch_size=BATCH_SIZE, shuffle=True, augment=augment)
        test_datagen = DataLoader(test_id, DATAFOLDER, batch_size=1, shuffle=False, augment=None)
        
        if MODEL_NAME == "unet2d_ds":
            model = Unet2D(num_classes=NUM_CLASSES, input_shape=(None, None, 3), deep_supervision=True)()
        else:
            model = sm.Unet(MODEL_NAME, input_shape=(None, None, 3), classes=NUM_CLASSES, encoder_weights='imagenet')
        
        optimizer = SGD(learning_rate=initial_lr, momentum=0.9, nesterov=True)

        model.compile(optimizer=optimizer, loss=bce_dice_loss(spartial_axis=(1, 2)), metrics=[Dice_Coef(spartial_axis=(1, 2))])
        
        callbacks = [
            CompetitionMetric(test_datagen, f'{MODEL_CHECKPOINTS_FOLDER}/{MODEL_NAME}/{MODEL_DESC}_fold{fold}.h5', num_class=NUM_CLASSES, period=25, patch_size=TRAINING_SIZE),
            LearningRateScheduler(schedule=poly_scheduler(initial_lr, no_of_epochs), verbose=1),
            CSVLogger(f'{MODEL_CHECKPOINTS_FOLDER}/{MODEL_NAME}/{MODEL_DESC}_fold{fold}.csv', separator=",", append=False)
        ]
        hist = model.fit_generator(train_datagen, 
                                epochs=no_of_epochs, 
                                callbacks = callbacks,
                                verbose=2)
        hists.append(callbacks[0].best_validation_score)
        break

    # PLOT TRAINING RESULTS
    val_Dice_Coef = []

    for i in range(1, KFOLD + 1):
        val_Dice_Coef.append(hists[i - 1])
        break

    print(val_Dice_Coef)
    print(f"{np.mean(val_Dice_Coef)} +- {np.std(val_Dice_Coef)}")
    print("Done!")