import numpy as np
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
        return X / 255., y 