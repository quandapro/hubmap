import numpy as np
import glob
import os
from tensorflow.keras.utils import Sequence

class DataLoader(Sequence):
    def __init__(self, train_ids, datafolder, batch_size, shuffle, augment):
        self.train_ids = train_ids
        self.datafolder = datafolder
        self.batch_size = batch_size 
        indices = np.arange(len(train_ids))
        if shuffle:
            np.random.shuffle(indices)
        self.indices = indices
        self.augment = augment
        self.shuffle = shuffle

    def get_id_from_folder(self, datafolder, train_ids):
        image_files = os.listdir(datafolder)
        image_files = [x for x in image_files if 'mask' not in x]
        image_ids = [x.split('.')[0] for x in image_files]
        image_ids = [x for x in image_ids if x.split("_")[0] in train_ids]
        return image_ids
        
    def load_data(self, train_id):
        X = np.load(f"{self.datafolder}/{train_id}.npy")
        y = np.load(f"{self.datafolder}/{train_id}_mask.npy")
        return X, y

    def __len__(self):
        if len(self.indices) % self.batch_size == 0 or self.shuffle:
            return len(self.indices) // self.batch_size
        return len(self.indices) // self.batch_size + 1

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __getitem__(self, index):
        indices = self.indices[index * self.batch_size : (index + 1) * self.batch_size]
        X = []
        y = []
        for i in range(len(indices)):
            train_id = self.train_ids[indices[i]]
            image, mask = self.load_data(train_id)
            if self.augment is not None:
                image, mask = self.augment(image, mask)
            X.append(image)
            y.append(mask)
        X = np.asarray(X, dtype='float32')
        y = np.asarray(y, dtype='float32')
        if len(X.shape) < 4: # Add channel axis if needed
            X = np.expand_dims(X, axis=-1)
        if len(y.shape) < 4: # Add channel axis if needed
            y = np.expand_dims(y, axis=-1)

        return X / 255., y 

class SegFormerDataLoader(Sequence):
    def __init__(self, train_ids, datafolder, batch_size, shuffle, augment):
        self.train_ids = train_ids
        self.datafolder = datafolder
        self.batch_size = batch_size 
        indices = np.arange(len(train_ids))
        if shuffle:
            np.random.shuffle(indices)
        self.indices = indices
        self.augment = augment
        self.shuffle = shuffle

    def get_id_from_folder(self, datafolder, train_ids):
        image_files = os.listdir(datafolder)
        image_files = [x for x in image_files if 'mask' not in x]
        image_ids = [x.split('.')[0] for x in image_files]
        image_ids = [x for x in image_ids if x.split("_")[0] in train_ids]
        return image_ids
        
    def load_data(self, train_id):
        X = np.load(f"{self.datafolder}/{train_id}.npy")
        y = np.load(f"{self.datafolder}/{train_id}_mask.npy")
        return X, y

    def __len__(self):
        if len(self.indices) % self.batch_size == 0 or self.shuffle:
            return len(self.indices) // self.batch_size
        return len(self.indices) // self.batch_size + 1

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __getitem__(self, index):
        indices = self.indices[index * self.batch_size : (index + 1) * self.batch_size]
        X = []
        y = []
        for i in range(len(indices)):
            train_id = self.train_ids[indices[i]]
            image, mask = self.load_data(train_id)
            if self.augment is not None:
                image, mask = self.augment(image, mask)
            X.append(image)
            y.append(mask)
        X = np.asarray(X, dtype='float32')
        y = np.asarray(y, dtype='float32')
        if len(X.shape) < 4: # Add channel axis if needed
            X = np.expand_dims(X, axis=-1)
        if len(y.shape) < 4: # Add channel axis if needed
            y = np.expand_dims(y, axis=-1)

        X = X.transpose(0, 3, 1, 2)
        # Change from one-hot encode to index
        y = np.argmax(y, axis=-1)
        return { 'pixel_values' : X / 255., 'labels' : y}