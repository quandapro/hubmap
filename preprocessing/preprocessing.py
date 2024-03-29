import os
import glob
import numpy as np
import pandas as pd
import cv2

import sys
sys.path.append('../')

from utils.image import *

"""
    CONFIGURATION
"""
DATA_FOLDER = "../data/train_images"
IMAGE_SIZE = (768, 768)
DF = pd.read_csv("../data/train.csv")
OUTPUT_FOLDER = "../data/npy_768x768"

CLASS_MAP = {'kidney' : 1,
             'largeintestine' : 2,
             'lung' : 3,
             'prostate' : 4,
             'spleen' : 5}

if __name__ == '__main__':
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    for i in range(len(DF)):
        image_id = DF["id"][i]
        mask_rle = DF["rle"][i]
        organ    = DF["organ"][i]

        print(f"Processing image id {image_id}")

        image, shape = open_image(f"{DATA_FOLDER}/{image_id}.tiff")
        # mask = np.zeros((*shape[:2], len(CLASS_MAP) + 1), dtype='uint8')
        # mask[..., CLASS_MAP[organ]] = rle_decode(mask_rle, shape[:2])
        # mask[..., 0] = 1. - mask[..., CLASS_MAP[organ]]

        mask = np.zeros((*shape[:2], 1), dtype='uint8')
        mask[..., 0] = rle_decode(mask_rle, shape[:2])

        image = cv2.resize(image, IMAGE_SIZE, cv2.INTER_AREA)
        mask = cv2.resize(mask, IMAGE_SIZE, cv2.INTER_NEAREST)

        np.save(f"{OUTPUT_FOLDER}/{image_id}.npy", image)
        np.save(f"{OUTPUT_FOLDER}/{image_id}_mask.npy", mask)
    
    print("Done")