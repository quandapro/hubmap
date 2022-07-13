import cv2
import numpy as np
import tifffile as tiff

'''
    Image helper functions
''' 
def open_image(path):
    image = tiff.imread(path)
    shape = image.shape
    return image, shape

def rle_decode(mask_rle, shape):
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T
