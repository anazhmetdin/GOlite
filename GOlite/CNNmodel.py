from keras.models import Sequential
from keras.layers import Conv1D, Dense, GlobalMaxPooling1D, MaxPooling1D
from GOlite.generator import DataGenerator
from keras import backend
import glob


class CNNmodel():
    def __init__(self, dPrefix, lPrefix, dim, label_dim, batchS, val,
                 filters, filterSize):
        self.model = Sequential()
        self.filters = filters
        self.filterSize = [int(i) for i in filterSize.split(',')]
        self.dPrefix = dPrefix
        self.lPrefix = lPrefix
        self.dim = [int(i) for i in dim.split(',')]
        self.label_dim = [int(i) for i in label_dim.split(',')]
        self.batchS = batchS
        self.val = val
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
        a = self.filterSize[0]
        step = self.filterSize[2]
        b = self.filterSize[1]
        self.model.add(Conv1D(filters=self.filters, kernel_size=a,
                       strides=self.batchS, activation='relu',
                       input_shape=(self.dim[1], 1)))
        for size in range(a+step, b+1, step):
            self.model.add(Conv1D(filters=self.filters, kernel_size=size,
                                  activation='relu'))
            if (size-a) % 3 == 0 and size != b:
                self.model.add(MaxPooling1D(padding="same"))
            if size == b:
                self.model.add(GlobalMaxPooling1D())
        self.model.add(Dense(self.label_dim[1], activation='softmax'))
        print(self.model.summary())
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

    def fit_model(self, testSize=0.1):
        # Parameters
        params = {'dim': tuple(self.dim),
                  'label_dim': tuple(self.label_dim),
                  'batch_size': self.batchS,
                  'n_channels': 1,
                  'shuffle': True}
        # Datasets
        partition = self.list_IDs
        labels = self.labels
        # Generators
        training_generator = DataGenerator(partition['train'], labels,
                                           **params)
        validation_generator = DataGenerator(partition['validation'],
                                             labels, **params)
        self.model.fit(x=training_generator,
                       validation_data=validation_generator)
