from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.models import load_model
from GOlite.generator import DataGenerator
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pickle
from random import randint
import numpy as np
from keras import backend
import glob


class CNNmodel():
    def __init__(self, dPrefix, lPrefix, dim, label_dim, val,
                 filters, filterSize, method='CN', params="121", model=""):
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
        self.t_History = {"loss": [], "auc": [], 'MSE': [],
                          'cat_acc': [], 'cat_crossE': []}
        self.v_History = {"loss": [], "auc": [], 'MSE': [],
                          'cat_acc': [], 'cat_crossE': []}
        self.generate_dicts()
        if model == "":
            self.build_model()
        else:
            print(model)
            self.model = load_model(model)
            history = model[:model.rfind("_")]
            print(history)
            with open(history+"_t_history", 'rb') as pickle_file:
                self.t_History = pickle.load(pickle_file)
            with open(history+"_v_history", 'rb') as pickle_file:
                self.v_History = pickle.load(pickle_file)
            self.model.compile(loss='binary_crossentropy', optimizer='adam',
                               metrics=["AUC", "MSE", 'categorical_accuracy',
                                        'categorical_crossentropy'])

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
            input_shape = tuple([self.dim[1], 1])
            from keras.layers import Conv1D as conv
            from keras.layers import GlobalMaxPooling1D as GlobalMaxPooling
            from keras.layers import MaxPooling1D as MaxPooling
        elif len(self.dim) == 4:
            input_shape = tuple([224, 224, 3])
            from keras.layers import Conv2D as conv
            from keras.layers import GlobalMaxPooling2D as GlobalMaxPooling
            from keras.layers import MaxPooling2D as MaxPooling
        if len(self.dim) == 3:
            input_shape = tuple([*self.dim, 1])
            from keras.layers import Conv2D as conv
            from keras.layers import GlobalMaxPooling2D as GlobalMaxPooling
            from keras.layers import MaxPooling2D as MaxPooling
        self.model.add(conv(filters=self.filters, kernel_size=a,
                       activation='relu',
                       input_shape=input_shape))
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
                           metrics=["AUC", "MSE", 'categorical_accuracy',
                                    'categorical_crossentropy'])

    def build_model_DenseNet(self):
        if self.params == "121":
            from keras.applications import DenseNet121 as DenseNet
        elif self.params == "169":
            from keras.applications import DenseNet169 as DenseNet
        elif self.params == "201":
            from keras.applications import DenseNet201 as DenseNet
        self.model = DenseNet(
            include_top=True,
            classes=self.label_dim[1],
        )
        self.model.compile(loss='binary_crossentropy', optimizer='adam',
                           metrics=["AUC", "MSE", 'categorical_accuracy',
                                    'categorical_crossentropy'])

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
                x_train = np.load(self.list_IDs['train'][indxs[i]])
                if self.method == "DN" or len(self.dim) == 4:
                    from cv2 import resize, INTER_AREA
                    temp = np.zeros((x_train.shape[0], 224, 224, 3))
                    for k in range(x_train.shape[0]):
                        x = resize(x_train[k:k+1:].reshape((x_train.shape[1],
                                                            x_train.shape[1],
                                                            3)),
                                   (224, 224), interpolation=INTER_AREA)
                        temp[k:k+1:] = x
                    x_train = temp
                if len(x_train.shape) != 4:
                    x_train = x_train.reshape([*x_train.shape, 1])
                y_train = np.load(self.labels[self.list_IDs['train'][indxs[i]]])
                if trainSize != 1:
                    x_train, x_test, y_train, y_test = train_test_split(x_train,
                                                                        y_train,
                                                                        train_size=trainSize,
                                                                        random_state=42)
                # if i == 0:
                #     results = self.model.train_on_batch(x_train, y_train,
                #                                         return_dict=True,
                #                                         reset_metrics=True)
                # else:
                results = self.model.train_on_batch(x_train, y_train,
                                                    return_dict=True,
                                                    reset_metrics=False)
                auc = results['auc']
                loss = results['loss']
                mse = results['MSE']
                acc = results['categorical_accuracy']
                crossE = results['categorical_crossentropy']

                progress = str(i+1)+"/"+str(batch_size)
                stats = " Loss: "+str(loss)+" AUC: "+str(auc)
                print("\r\tbatch " + progress + stats, end='')

            self.t_History['auc'].append(auc)
            self.t_History['loss'].append(loss)
            self.t_History['MSE'].append(mse)
            self.t_History['cat_acc'].append(acc)
            self.t_History['cat_crossE'].append(crossE)

            indxs1 = np.arange(0, Vlen-1)
            np.random.shuffle(indxs1)
            for l in range(int(0.5*len(indxs1))):
                x_test = np.load(self.list_IDs['validation'][indxs1[l]])

                if self.method == "DN" or len(self.dim) == 4:
                    from cv2 import resize, INTER_AREA
                    temp = np.zeros((x_test.shape[0], 224, 224, 3))
                    for k in range(x_test.shape[0]):
                        x = resize(x_test[k:k+1:].reshape((x_test.shape[1],
                                                            x_test.shape[1],
                                                            3)),
                                   (224, 224), interpolation=INTER_AREA)
                        temp[k:k+1:] = x
                    x_test = temp

                if len(x_train.shape) != 4:
                    x_test = x_test.reshape([*x_test.shape, 1])
                y_test = np.load(self.labels[self.list_IDs['validation'][indxs1[l]]])

                # if i == 0:
                #     results = self.model.test_on_batch(x_test, y_test,
                #                                        return_dict=True,
                #                                        reset_metrics=True)
                # else:
                results = self.model.test_on_batch(x_test, y_test,
                                                   return_dict=True,
                                                   reset_metrics=False)
            auc = results['auc']
            loss = results['loss']
            mse = results['MSE']
            acc = results['categorical_accuracy']
            crossE = results['categorical_crossentropy']

            stats = " Loss: "+str(loss)+" AUC: "+str(auc)
            print("\n\t\tvalidation:" + stats)

            self.v_History['auc'].append(auc)
            self.v_History['loss'].append(loss)
            self.v_History['MSE'].append(mse)
            self.v_History['cat_acc'].append(acc)
            self.v_History['cat_crossE'].append(crossE)

            if self.method == "CN":
                method_name = self.method
            elif self.method == "DN":
                method_name = self.method+"_"+self.params

            self.model.save(filepath+"_"+method_name+"_"+str(j))

            for metric in self.t_History:
                plt.plot(self.t_History[metric],
                         label=metric+' (training data)')
                plt.plot(self.v_History[metric],
                         label=metric + ' (validation data)')
                plt.title(metric + ' for ' + method_name)
                plt.ylabel(metric + ' value')
                plt.xlabel('No. epoch')
                plt.legend(loc="upper left")
                plt.savefig(filepath+"_"+method_name+"_"+metric+".png")
                plt.clf()

            with open(filepath+"_"+method_name+"_"+"t_history", 'wb') as file_pi:
                pickle.dump(self.t_History, file_pi)
            with open(filepath+"_"+method_name+"_"+"v_history", 'wb') as file_pi:
                pickle.dump(self.v_History, file_pi)

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
