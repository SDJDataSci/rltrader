import numpy as np
from keras.models import Sequential
from keras.layers import Activation, LSTM, Dense, BatchNormalization
from keras.optimizers import sgd, Adam


class PolicyNetwork:
    def __init__(self, input_dim=0, output_dim=0, lr=0.01):
        self.input_dim = input_dim
        self.lr = lr
        self.model = Sequential()
        self.model.add(LSTM(512, input_shape=(1, input_dim),
                            return_sequences=True, stateful=False, dropout=0.7))
        self.model.add(Dense(512, activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(LSTM(512, return_sequences=True, stateful=False, dropout=0.7))
        self.model.add(Dense(512, activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(LSTM(512, return_sequences=True, stateful=False, dropout=0.7))
        self.model.add(Dense(512, activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(LSTM(512, return_sequences=True, stateful=False, dropout=0.7))
        self.model.add(Dense(512, activation='sigmoid'))
        self.model.add(BatchNormalization())
        self.model.add(LSTM(256, return_sequences=True, stateful=False, dropout=0.7))
        self.model.add(Dense(256, activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(LSTM(256, return_sequences=False, stateful=False, dropout=0.7))
        self.model.add(Dense(256, activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(Dense(output_dim))
        self.model.add(Activation('sigmoid'))
        self.model.compile(optimizer=Adam(lr=lr), loss='mse')
        self.prob = None


        # self.model.add(LSTM(256, input_shape=(1, input_dim),
        #                     return_sequences=True, stateful=False, dropout=0.5))
        # self.model.add(Dense(256, activation='relu'))
        # self.model.add(BatchNormalization())
        # self.model.add(LSTM(128, return_sequences=True, stateful=False, dropout=0.6))
        # self.model.add(Dense(128, activation='relu'))
        # self.model.add(BatchNormalization())
        # self.model.add(LSTM(128, return_sequences=True, stateful=False, dropout=0.7))
        # self.model.add(Dense(64, activation='relu'))
        # self.model.add(BatchNormalization())
        # self.model.add(LSTM(64, return_sequences=False, stateful=False))
        # self.model.add(BatchNormalization())


    def reset(self):
        self.prob = None

    def predict(self, sample):
        self.prob = self.model.predict(np.array(sample).reshape((1, -1, self.input_dim)))[0]
        return self.prob

    def train_on_batch(self, x, y):
        return self.model.train_on_batch(x, y)

    def save_model(self, model_path):
        if model_path is not None and self.model is not None:
            self.model.save_weights(model_path, overwrite=True)

    def load_model(self, model_path):
        if model_path is not None:
            self.model.load_weights(model_path)