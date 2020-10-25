from keras.models import Sequential
from keras.layers import Conv1D
from keras.layers import Dense
from GOlite.generator import DataGenerator
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

    def build_model(self):
        a = self.filterSize[0]
        step = self.filterSize[1]
        b = self.filterSize[2]
        print(self.filters, type(self.filters))
        print(a, type(a))
        print(self.batchS, type(self.batchS))
        print(self.dim[1], type(self.dim[1]))
        self.model.add(Conv1D(filters=self.filters, kernel_size=a,
                       strides=self.batchS, activation='relu',
                       input_shape=(None, self.dim[1])))
        for size in range(a+step, b+1, step):
            self.model.add(Conv1D(filters=self.filters, kernel_size=size,
                                  activation='relu'))
        self.model.add(Dense(self.label_dim[1], activation='sigmoid'))
        self.model.compile(optimizer='adam',
                           loss='categorical_crossentropy',
                           metrics=['categorical_crossentropy'])

    def generate_dicts(self):
        iList = glob.glob(self.dPrefix)
        oList = glob.glob(self.lPrefix)
        for i in range(int(self.val*len(iList))-1):
            self.list_IDs['train'].append(iList[i])
            self.labels[iList[i]] = oList[i]
        for i in range(int(self.val*len(iList)), len(iList)):
            self.list_IDs['train'].append(iList[i])
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
                       validation_data=validation_generator,
                       use_multiprocessing=True, workers=6)
