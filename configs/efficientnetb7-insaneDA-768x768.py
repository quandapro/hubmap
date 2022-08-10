import albumentations as A
import cv2

RANDOM_SEED       = 2022

MODEL_NAME        = "efficientnetb7"
MODEL_DESCRIPTION = "insaneDA_768x768"

DATAFOLDER        = "./data/npy_768x768"
TRAINING_SIZE     = (768, 768)

EPOCHS            = 100
BATCH_SIZE        = 4
KFOLD             = 5
NUM_CLASSES       = 1

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