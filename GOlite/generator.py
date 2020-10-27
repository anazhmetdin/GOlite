import numpy as np
import keras
from sklearn.model_selection import train_test_split


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, list_IDs, labels, dim, label_dim, batch_size=32,
                 n_channels=1, shuffle=True, trainSize=1):
        'Initialization'
        self.dim = dim
        self.label_dim = label_dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.trainSize = trainSize
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle is True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'
        # Initialization
        X = np.empty([*self.dim, self.n_channels], dtype=float)
        y = np.empty([*self.label_dim], dtype=float)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            temp = np.load(ID)
            temp = temp.reshape(*temp.shape, self.n_channels)
            X = temp
            # Store class
            y = np.load(self.labels[ID])
            x_train, x_test, y_train, y_test = train_test_split(X,
                                                                y,
                                                                train_size=self.trainSize,
                                                                random_state=42)
        return x_train, y_train
