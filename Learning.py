import pandas as pd
from DataProcessing import processing
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from keras.optimizers import Adam
import tensorflow as tf
import os
import kagglehub 
from keras.models import Sequential
from keras import layers

class deep_learning():

    def __init__(self, processor:processing):
        self.scaled_features_train, self.scaled_features_test, self.labels_train, self.labels_test, self.transformer = processor.preprocess_data()

        self.model = Sequential()
        
        self.model.add(layers.Input(shape=[self.scaled_features_train.shape[1]]))
        self.model.add(layers.Dense(16, activation='relu'))
        self.model.add(layers.Dense(128, activation='relu'))
        self.model.add(layers.Dense(32, activation='relu'))
        self.model.add(layers.Dense(1, activation='relu'))
        
        self.optimizer = Adam(learning_rate=0.01)
        print(self.model.summary())

    def learn(self, epochs=100):
        self.model.compile(loss='mse', optimizer=self.optimizer, metrics=['mae'])
        history = self.model.fit(self.scaled_features_train, self.labels_train, epochs=epochs, verbose=1)
        metrics = self.get_metrics()
        print(metrics)
        return history
    
    
    def get_metrics(self):
        mse , mae = self.model.evaluate(self.scaled_features_test, self.labels_test, verbose=0)
        return mse, mae
    

    
    def predict(self , input:pd.Series):
        #input = self.transformer.transform(input)
        #print(input.shape)
        return self.model.predict(input)


