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
OUTPUT_FOLDER = "./data/npy_tile_512x512"
TILE_SIZE = (512, 512)
STRIDE = (256, 256)

CLASS_MAP = {'kidney' : 0,
             'largeintestine' : 1,
             'lung' : 2,
             'prostate' : 3,
             'spleen' : 4}

def center_padding(image, desired_shape):
    h, w = image.shape[:2]
    result = np.zeros((*desired_shape, image.shape[-1]), dtype='float32')
    padding_h = (desired_shape[0] - h) // 2
    padding_w = (desired_shape[1] - w) // 2
    result[padding_h:padding_h + h, padding_w:padding_w + w, :] = image
    return result

def make_divisible(nums, divide=512):
    return [num + (divide - num % divide) for num in nums]

def save_tile(image, mask, image_id):
    h, w = image.shape[:2]
    t_h, t_w = TILE_SIZE
    s_h, s_w = STRIDE
    desired_shape = make_divisible([h, w], TILE_SIZE[0])
    print(f"New image shape for {image_id}: {desired_shape}")
    new_h, new_w = desired_shape
    image = center_padding(image, desired_shape)
    mask = center_padding(mask, desired_shape)
    starting_points = [(x, y) for x in set( list(range(0, new_h - t_h, s_h)) + [new_h - t_h])
                            for y in set( list(range(0, new_w - t_w, s_w)) + [new_w - t_w])]
    for (x, y) in starting_points:
        patch_image = image[x:x + t_h, y:y + t_w]
        patch_mask  = mask[x:x + t_h, y:y + t_w]
        np.save(f"{OUTPUT_FOLDER}/{image_id}_{x}_{y}.npy", patch_image)
        np.save(f"{OUTPUT_FOLDER}/{image_id}_{x}_{y}_mask.npy", patch_mask)

if __name__ == '__main__':
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    for i in range(len(DF)):
        image_id = DF["id"][i]
        mask_rle = DF["rle"][i]
        organ    = DF["organ"][i]

        print(f"Processing image id {image_id}")

        image, shape = open_image(f"{DATA_FOLDER}/{image_id}.tiff")
        mask = np.zeros((*shape[:2], 1), dtype='uint8')
        mask[..., 0] = rle_decode(mask_rle, shape[:2])
        save_tile(image, mask, image_id)
    
    print("Done")