"""
Created on Wed Jul 01 2020
Updated on Wed Jul 04 2020

Initial implementation of autokeras for model optimization
"""

import os
import numpy as np
import pandas as pd
import autokeras as ak
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, LSTM, Flatten
from tensorflow.keras import backend as K
import sklearn
from sklearn.preprocessing import power_transform
from sklearn.metrics import mean_absolute_error
from sklearn.datasets import fetch_california_housing
import pmdarima as pm
from pmdarima.metrics import smape


class ImageClassification():
    def train(inData):
        (x_train, y_train), (x_test, y_test) = inData
        
        # Initialize the image classifier.
        input_node = ak.ImageInput()
        output_node = ak.ImageBlock(
            # Only search ResNet architectures.
            block_type="resnet",
            # Normalize the dataset.
            normalize=True,
            # Do not do data augmentation.
            augment=False,
        )(input_node)
        output_node = ak.ClassificationHead()(output_node)
        model = ak.AutoModel(
            inputs=input_node,
            outputs=output_node,
            overwrite=True,
            max_trials=1)
        
        # Feed the image classifier with training data.
        model.fit(
            x_train,
            y_train,
            # Split the training data and use the last 15% as validation data.
            validation_split=0.15,
            epochs=10)
        
        # Predict with the best model.
        predicted_y = model.predict(x_test)
        print(predicted_y)
        
        # Evaluate the best model with testing data.
        print(model.evaluate(x_test, y_test))
        
        return model
        

class ImageRegression():
    def train(inData):
        (x_train, y_train), (x_test, y_test) = inData
        
        # Initialize the image regressor.
        input_node = ak.ImageInput()
        output_node = ak.ImageBlock(
            # Only search ResNet architectures.
            block_type="resnet",
            # Normalize the dataset.
            normalize=False,
            # Do not do data augmentation.
            augment=False,
        )(input_node)
        output_node = ak.RegressionHead()(output_node)
        model = ak.AutoModel(
            inputs=input_node,
            outputs=output_node,
            overwrite=True,
            max_trials=1)
        
        # Feed the image regressor with training data.
        model.fit(
            x_train,
            y_train,
            # Split the training data and use the last 15% as validation data.
            validation_split=0.15,
            epochs=2)
        
        # Predict with the best model.
        predicted_y = model.predict(x_test)
        print(predicted_y)
        
        # Evaluate the best model with testing data.
        print(model.evaluate(x_test, y_test))
        
        return model
        

class TextClassification():
    def train(inData):
        index_offset = 3  # word index offset
        (x_train, y_train), (x_test, y_test) = inData
        y_train = y_train.reshape(-1, 1)
        y_test = y_test.reshape(-1, 1)
        # Prepare the dictionary of index to word.
        word_to_id = imdb.get_word_index()
        word_to_id = {k: (v + index_offset) for k, v in word_to_id.items()}
        word_to_id["<PAD>"] = 0
        word_to_id["<START>"] = 1
        word_to_id["<UNK>"] = 2
        id_to_word = {value: key for key, value in word_to_id.items()}
        # Convert the word indices to words.
        x_train = list(map(lambda sentence: ' '.join(
            id_to_word[i] for i in sentence), x_train))
        x_test = list(map(lambda sentence: ' '.join(
            id_to_word[i] for i in sentence), x_test))
        x_train = np.array(x_train, dtype=np.str)
        x_test = np.array(x_test, dtype=np.str)
        
        # Initialize the text classifier.
        input_node = ak.TextInput()
        output_node = ak.TextBlock(vectorizer='ngram')(input_node)
        output_node = ak.ClassificationHead()(output_node)
        model = ak.AutoModel(
            inputs=input_node,
            outputs=output_node,
            overwrite=True,
            max_trials=1)

        # Feed the text classifier with training data.
        model.fit(x_train,
            y_train,
            # Split the training data and use the last 15% as validation data.
            validation_split=0.15,
            epochs=2)
        
        # Predict with the best model.
        predicted_y = model.predict(x_test)
        
        # Evaluate the best model with testing data.
        print(model.evaluate(x_test, y_test))
        
        return model
        

class TextRegression():
    def train(inData):
        index_offset = 3  # word index offset
        (x_train, y_train), (x_test, y_test) = inData
        y_train = y_train.reshape(-1, 1)
        y_test = y_test.reshape(-1, 1)
        # Prepare the dictionary of index to word.
        word_to_id = imdb.get_word_index()
        word_to_id = {k: (v + index_offset) for k, v in word_to_id.items()}
        word_to_id["<PAD>"] = 0
        word_to_id["<START>"] = 1
        word_to_id["<UNK>"] = 2
        id_to_word = {value: key for key, value in word_to_id.items()}
        # Convert the word indices to words.
        x_train = list(map(lambda sentence: ' '.join(
            id_to_word[i] for i in sentence), x_train))
        x_test = list(map(lambda sentence: ' '.join(
            id_to_word[i] for i in sentence), x_test))
        x_train = np.array(x_train, dtype=np.str)
        x_test = np.array(x_test, dtype=np.str)
        
        # Initialize the text regressor.
        input_node = ak.TextInput()
        output_node = ak.TextBlock(vectorizer='ngram')(input_node)
        output_node = ak.RegressionHead()(output_node)
        model = ak.AutoModel(
            inputs=input_node,
            outputs=output_node,
            overwrite=True,
            max_trials=1)
        
        # Feed the text regressor with training data.
        model.fit(x_train,
            y_train,
            # Split the training data and use the last 15% as validation data.
            validation_split=0.15,
            epochs=2)
        
        # Predict with the best model.
        predicted_y = model.predict(x_test)
        
        # Evaluate the best model with testing data.
        print(model.evaluate(x_test, y_test))
        
        return model
        

class StructuredClassification():
    def train(df):
        train_size = int(df.shape[0] * 0.9)
        df[:train_size].to_csv('train.csv', index=False)
        df[train_size:].to_csv('eval.csv', index=False)
        train_file_path = 'train.csv'
        test_file_path = 'eval.csv'
        
        # x_train as pandas.DataFrame, y_train as pandas.Series
        x_train = pd.read_csv(train_file_path)
        print(type(x_train)) # pandas.DataFrame
        y_train = x_train.pop('Class')
        print(type(y_train)) # pandas.Series
        
        # You can also use pandas.DataFrame for y_train.
        y_train = pd.DataFrame(y_train)
        print(type(y_train)) # pandas.DataFrame
        
        # You can also use numpy.ndarray for x_train and y_train.
        x_train = x_train.to_numpy().astype(np.unicode)
        y_train = y_train.to_numpy()
        print(type(x_train)) # numpy.ndarray
        print(type(y_train)) # numpy.ndarray
        
        # Preparing testing data.
        x_test = pd.read_csv(test_file_path)
        y_test = x_test.pop('Class')
        
        # Initialize the structured data classifier.
        input_node = ak.StructuredDataInput()
        output_node = ak.StructuredDataBlock(categorical_encoding=True)(input_node)
        output_node = ak.ClassificationHead()(output_node)
        model = ak.AutoModel(
            inputs=input_node, 
            outputs=output_node, 
            overwrite=True,
            max_trials=3)
        
        # Feed the structured data classifier with training data.
        model.fit(x_train,
            y_train,
            # Split the training data and use the last 15% as validation data.
            validation_split=0.15,
            epochs=10)
        
        # Predict with the best model.
        predicted_y = model.predict(test_file_path)
        
        # Evaluate the best model with testing data.
        print(model.evaluate(test_file_path, 'Class'))
        
        return model
        

class StructuredRegression():
    def train(df):
        train_size = int(df.shape[0] * 0.9)
        df[:train_size].to_csv('train.csv', index=False)
        df[train_size:].to_csv('eval.csv', index=False)
        train_file_path = 'train.csv'
        test_file_path = 'eval.csv'
        
        # x_train as pandas.DataFrame, y_train as pandas.Series
        x_train = pd.read_csv(train_file_path)
        print(type(x_train)) # pandas.DataFrame
        y_train = x_train.pop('Value')
        print(type(y_train)) # pandas.Series
        
        # You can also use pandas.DataFrame for y_train.
        y_train = pd.DataFrame(y_train)
        print(type(y_train)) # pandas.DataFrame
        
        # You can also use numpy.ndarray for x_train and y_train.
        x_train = x_train.to_numpy().astype(np.unicode)
        y_train = y_train.to_numpy()
        print(type(x_train)) # numpy.ndarray
        print(type(y_train)) # numpy.ndarray
        
        # Preparing testing data.
        x_test = pd.read_csv(test_file_path)
        y_test = x_test.pop('Value')
                
        # Initialize the structured data regressor.
        input_node = ak.StructuredDataInput()
        output_node = ak.StructuredDataBlock(categorical_encoding=True)(input_node)
        output_node = ak.RegressionHead()(output_node)
        model = ak.AutoModel(
            inputs=input_node, 
            outputs=output_node, 
            overwrite=True,
            max_trials=3)
        
        # Feed the structured data regressor with training data.
        model.fit(x_train,
            y_train,
            # Split the training data and use the last 15% as validation data.
            validation_split=0.15,
            epochs=10)
       
        # Predict with the best model.
        predicted_y = model.predict(test_file_path)
        
        # Evaluate the best model with testing data.
        print(model.evaluate(test_file_path, 'Value'))
        
        return model
    
class VanillaLSTM():
    def train(df): 
        df=df.drop(columns=['Date Time'])
        
        #User has to specify feature columns, I chose columns 4 through 7
        X=df.iloc[:, 3:7]
        X=np.array(X)
        
        #Similarly the user has to choose output column, I chose the 2nd one since it's the Temperature(°C)
        y=df.iloc[:,[1]]
        y=np.array(y)

        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2, random_state=3)
        X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
        X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
        y_train=y_train.reshape(-1, 1)
        y_test=y_test.reshape(-1, 1)

        print(X_train.shape)
        print(y_train.shape)
        print(X_test.shape)
        print(y_test.shape)

        model = keras.Sequential()
        model.add(LSTM(64, activation='tanh', return_sequences=True))
        model.add(LSTM(32, activation='tanh', return_sequences=True))
        model.add(LSTM(64, activation='tanh', return_sequences=True))
        model.add(LSTM(32, activation='tanh', return_sequences=True))
        model.add(LSTM(64, activation='tanh', return_sequences=True))
        model.add(LSTM(32, activation='tanh', return_sequences=True))
        model.add(LSTM(64, activation='tanh', return_sequences=True))
        model.add(LSTM(32, activation='tanh', return_sequences=True))
        model.add(Dense(64, activation='relu'))
        model.add(Flatten())
        model.add(Dense(1, activation='linear'))

        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        model.fit(X_train, y_train, batch_size=256, epochs=15, validation_data=(X_test, y_test))
        print(model.evaluate(X_test, y_test))
        
        return model

class AutoARIMA():
    def train(df):
        #Get a dataset. This is Microsoft stock data.
        df = df.drop(columns=['Date', 'Volume', 'OpenInt'])
        
        #Dataset shape is now (7983,4)
        print(df.shape)
        
        #define the series to be forecasted (user specified)
        y = df['High']
        y = np.array(y)
        y = y.reshape(-1,1)
        
        #exog represents the exogeneous variables (user specified)
        exog = df[['Open','Low','Close']]
        exog = np.array(exog)
        
        #Box-Cox transform on y and exog
        y = power_transform(y, method='box-cox')
        exog = power_transform(exog, method='box-cox')
        
        y_train, y_test = pm.model_selection.train_test_split(y, test_size=0.2)
        exog_train, exog_test = pm.model_selection.train_test_split(exog, test_size=0.2)
        
        model = pm.auto_arima(y_train, exog_train, start_p=1, d=None, start_q=1, information_criterion='aic', 
                                   maxiter=100, method='lbfgs', test='kpss', stepwise=True)
        
 
        forecasts = model.predict(y_test.shape[0], exog_test)
        
        error = smape(y_test, forecasts)
        mae = mean_absolute_error(y_test, forecasts)
        print("Symmetric Mean Absolute Percentage Error: ", error)
        print("Mean Absolute Error: ", mae)
        
        return model
