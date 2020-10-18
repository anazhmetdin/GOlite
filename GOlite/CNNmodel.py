from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv1D
from keras.layers import Dense


class CNNmodel():
    def __init__(self, filters, filterSize):
        self.model = Sequential()
        self.filters = filters
        self.fiterSize = filterSize.split(',')
        self.build_model()
        self.X_train = []
        self.X_test = []
        self.y_train = []
        self.y_test = []

    def build_model(self):
        a = self.filterSize[0]
        step = self.filterSize[1]
        b = self.filterSize[2]
        self.model.add(Conv1D(filters=self.filters, kernel_size=a,
                              activation='relu', input_shape=(None, 21, 2000)))
        for size in range(a+step, b+1, step):
            self.model.add(Conv1D(filters=self.filters, kernel_size=size,
                                  activation='relu'))
        self.model.add(Dense(32, activation='sigmoid'))
        self.model.compile(optimizer='adam',
                           loss='sparse_categorical_crossentropy',
                           metrics=['sparse_categorical_crossentropy'])

    def setup_data(self, data, testSize=0.1):
        self.X_train, self.X_test,
        self.y_train, self. y_test = train_test_split(seqData, functionData,
                                                      test_size=testSize,
                                                      random_state=42)
