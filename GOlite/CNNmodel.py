from keras.models import Sequential
from keras.layers import Dense, Flatten
from GOlite.generator import DataGenerator
from sklearn.model_selection import train_test_split
from random import randint
import numpy as np
from keras import backend
import glob


class CNNmodel():
    def __init__(self, dPrefix, lPrefix, dim, label_dim, val,
                 filters, filterSize, method='CN', params="121"):
        self.model = None
        self.filters = filters
        self.filterSize = [int(i) for i in filterSize.split(',')]
        self.dPrefix = dPrefix
        self.lPrefix = lPrefix
        self.dim = [int(i) for i in dim.split(',')]
        self.label_dim = [int(i) for i in label_dim.split(',')]
        self.val = val
        self.method = method
        self.params = params
        self.list_IDs = dict()
        self.list_IDs['train'] = list()
        self.list_IDs['validation'] = list()
        self.labels = dict()
        self.generate_dicts()
        self.build_model()

    def fbeta(y_true, y_pred, beta=2):
        # clip predictions
        y_pred = backend.clip(y_pred, 0, 1)
        # calculate elements
        tp = backend.sum(backend.round(backend.clip(y_true * y_pred, 0, 1)), axis=1)
        fp = backend.sum(backend.round(backend.clip(y_pred - y_true, 0, 1)), axis=1)
        fn = backend.sum(backend.round(backend.clip(y_true - y_pred, 0, 1)), axis=1)
        # calculate precision
        p = tp / (tp + fp + backend.epsilon())
        # calculate recall
        r = tp / (tp + fn + backend.epsilon())
        # calculate fbeta, averaged across each class
        bb = beta ** 2
        fbeta_score = backend.mean((1 + bb) * (p * r) / (bb * p + r + backend.epsilon()))
        return fbeta_score

    def build_model(self):
        if self.method == "CN":
            self.build_model_CNN()
        elif self.method == "DN":
            self.build_model_DenseNet()

    def build_model_CNN(self):
        self.model = Sequential()
        a = self.filterSize[0]
        step = self.filterSize[2]
        b = self.filterSize[1]
        if len(self.dim) == 2:
            from keras.layers import Conv1D as conv
            from keras.layers import GlobalMaxPooling1D as GlobalMaxPooling
            from keras.layers import MaxPooling1D as MaxPooling
        elif len(self.dim) == 3:
            from keras.layers import Conv2D as conv
            from keras.layers import GlobalMaxPooling2D as GlobalMaxPooling
            from keras.layers import MaxPooling2D as MaxPooling
        self.model.add(conv(filters=self.filters, kernel_size=a,
                       activation='relu',
                       input_shape=(*self.dim[1:], 1)))
        for size in range(a+step, b+1, step):
            self.model.add(conv(filters=self.filters, kernel_size=size,
                                activation='relu'))
            if (size-a) % 3 == 0 and size != b:
                self.model.add(MaxPooling(padding="same"))
        self.model.add(GlobalMaxPooling())
        if len(self.dim) == 3:
            self.model.add(Flatten())
        self.model.add(Dense(self.label_dim[1], activation='sigmoid'))
        print(self.model.summary())
        self.model.compile(loss='binary_crossentropy', optimizer='adam',
                           metrics=['AUC'])

    def build_model_DenseNet(self):
        if self.params == "121":
            from keras.applications import DenseNet121 as DenseNet
        elif self.params == "169":
            from keras.applications import DenseNet169 as DenseNet
        elif self.params == "201":
            from keras.applications import DenseNet201 as DenseNet
        self.model = DenseNet(
            include_top=True,
            weights=None,
            input_tensor=None,
            input_shape=None,
            pooling="max",
            classes=self.label_dim[1],
        )
        self.model.compile(loss='binary_crossentropy', optimizer='adam',
                           metrics=['AUC'])

    def generate_dicts(self):
        iList = sorted(glob.glob(self.dPrefix))
        oList = sorted(glob.glob(self.lPrefix))
        for i in range(int(self.val*len(iList)), len(iList)):
            self.list_IDs['train'].append(iList[i])
            self.labels[iList[i]] = oList[i]
        for i in range(int(self.val*len(iList))-1):
            self.list_IDs['validation'].append(iList[i])
            self.labels[iList[i]] = oList[i]

    def fit_model_bitByBit(self, filepath, batch_size=10, epochs=13,
                           trainSize=0.2):
        results = []
        Tlen = len(self.list_IDs["train"])
        Vlen = len(self.list_IDs["validation"])
        indxs = np.arange(0, Tlen)
        for j in range(epochs):
            print("epoch", j+1, "/", epochs)
            np.random.shuffle(indxs)
            for i in range(batch_size):
                print("\tbatch", i+1, "/", batch_size)
                x_train = np.load(self.list_IDs['train'][indxs[i]])
                x_train = x_train.reshape([*x_train.shape, 1])
                y_train = np.load(self.labels[self.list_IDs['train'][indxs[i]]])
                if trainSize != 1:
                    x_train, x_test, y_train, y_test = train_test_split(x_train,
                                                                        y_train,
                                                                        train_size=trainSize,
                                                                        random_state=42)
                if i == 0:
                    results = self.model.train_on_batch(x_train, y_train,
                                                        return_dict=True,
                                                        reset_metrics=True)
                else:
                    results = self.model.train_on_batch(x_train, y_train,
                                                        return_dict=True,
                                                        reset_metrics=False)
                print("\t\ttraining:", results)

            indxs = np.arange(0, Vlen-1)
            np.random.shuffle(indxs)
            for i in range(10):
                x_test = np.load(self.list_IDs['validation'][i])
                x_test = x_test.reshape([*x_test.shape, 1])
                y_test = np.load(self.labels[self.list_IDs['validation'][i]])

                if i == 0:
                    results = self.model.test_on_batch(x_test, y_test,
                                                       return_dict=True,
                                                       reset_metrics=True)
                else:
                    results = self.model.test_on_batch(x_test, y_test,
                                                       return_dict=True,
                                                       reset_metrics=False)
            print("\t\tvalidation:", results)
            if self.method == "CN":
                self.model.save(filepath+"_"+self.method+"_"+str(j)+".tf")
            elif self.method == "DN":
                self.model.save(filepath+"_"+self.method+"_"+self.params+"_"+str(j)+".tf")

    def fit_model_generator(self, batch_size=10, epochs=13, trainSize=0.2):
        # Parameters
        params = {'dim': tuple(self.dim),
                  'label_dim': tuple(self.label_dim),
                  'batch_size': batch_size,
                  'n_channels': 1,
                  'shuffle': True,
                  'trainSize': trainSize}
        # Datasets
        partition = self.list_IDs
        labels = self.labels
        # Generators
        training_generator = DataGenerator(partition['train'], labels,
                                           **params)
        params['trainSize'] = 1
        validation_generator = DataGenerator(partition['validation'],
                                             labels, **params)
        self.model.fit(x=training_generator,
                       validation_data=validation_generator, epochs=epochs)
