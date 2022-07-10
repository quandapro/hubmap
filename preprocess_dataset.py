import os
import glob
import numpy as np
import pandas as pd
import cv2

from utils.image import *

"""
    CONFIGURATION
"""
DATA_FOLDER = "./data/train_images"
DF = pd.read_csv("./data/train.csv")
OUTPUT_FOLDER = "./data/npy"

CLASS_MAP = {'kidney' : 0,
             'largeintestine' : 1,
             'lung' : 2,
             'prostate' : 3,
             'spleen' : 4}

if __name__ == '__main__':
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    for i in range(len(DF)):
        image_id = DF["id"][i]
        mask_rle = DF["rle"][i]
        organ    = DF["organ"][i]

        print(f"Processing image id {image_id}")

        image, shape = open_image(f"{DATA_FOLDER}/{image_id}.tiff")
        mask = np.zeros((*shape[:2], len(CLASS_MAP)), dtype='uint8')
        mask[..., CLASS_MAP[organ]] = rle_decode(mask_rle, shape[:2])
        np.save(f"{OUTPUT_FOLDER}/{image_id}.npy", image)
        np.save(f"{OUTPUT_FOLDER}/{image_id}_mask.npy", mask)
    
    print("Done")