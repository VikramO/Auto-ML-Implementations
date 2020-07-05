"""
Created on Wed Jul 01 2020
Updated on Wed Jul 04 2020

Initial implementation of autokeras for model optimization
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.datasets import mnist, imdb
import autokeras as ak
from sklearn.datasets import fetch_california_housing
import os
import pmdarima as pm
from pmdarima.model_selection import train_test_split
from pmdarima.preprocessing import BoxCoxEndogTransformer
from pmdarima.metrics import smape
from sklearn.preprocessing import power_transform
from sklearn.metrics import mean_absolute_error


class ImageClassification():
    def train():
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        
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
        clf = ak.AutoModel(
            inputs=input_node,
            outputs=output_node,
            overwrite=True,
            max_trials=1)
        
        # Feed the image classifier with training data.
        clf.fit(
            x_train,
            y_train,
            # Split the training data and use the last 15% as validation data.
            validation_split=0.15,
            epochs=10,
        )
        
        # Predict with the best model.
        predicted_y = clf.predict(x_test)
        print(predicted_y)
        
        # Evaluate the best model with testing data.
        print(clf.evaluate(x_test, y_test))
        

class ImageRegression():
    def train():
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        
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
        reg = ak.AutoModel(
            inputs=input_node,
            outputs=output_node,
            overwrite=True,
            max_trials=1)
        
        # Feed the image regressor with training data.
        reg.fit(
            x_train,
            y_train,
            # Split the training data and use the last 15% as validation data.
            validation_split=0.15,
            epochs=2,
        )
        
        # Predict with the best model.
        predicted_y = reg.predict(x_test)
        print(predicted_y)
        
        # Evaluate the best model with testing data.
        print(reg.evaluate(x_test, y_test))
        

class TextClassification():
    def train():
        # Load the integer sequence the IMDB dataset with Keras.
        index_offset = 3  # word index offset
        (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=1000,
                                                              index_from=index_offset)
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
        clf = ak.AutoModel(
            inputs=input_node,
            outputs=output_node,
            overwrite=True,
            max_trials=1)
        clf.fit(x_train, y_train, epochs=2)

        # Feed the text classifier with training data.
        clf.fit(x_train,
            y_train,
            # Split the training data and use the last 15% as validation data.
            validation_split=0.15)
        
        # Predict with the best model.
        predicted_y = clf.predict(x_test)
        
        # Evaluate the best model with testing data.
        print(clf.evaluate(x_test, y_test))
        

class TextRegression():
    def train():
        # Load the integer sequence the IMDB dataset with Keras.
        index_offset = 3  # word index offset
        (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=1000,
                                                              index_from=index_offset)
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
        reg = ak.AutoModel(
            inputs=input_node,
            outputs=output_node,
            overwrite=True,
            max_trials=1)
        reg.fit(x_train, y_train, epochs=2)
        
        # Feed the text regressor with training data.
        reg.fit(x_train,
            y_train,
            # Split the training data and use the last 15% as validation data.
            validation_split=0.15)
        
        # Predict with the best model.
        predicted_y = reg.predict(x_test)
        
        # Evaluate the best model with testing data.
        print(reg.evaluate(x_test, y_test))
        

class StructuredClassification():
    def train():
        TRAIN_DATA_URL = "https://storage.googleapis.com/tf-datasets/titanic/train.csv"
        TEST_DATA_URL = "https://storage.googleapis.com/tf-datasets/titanic/eval.csv"
        
        train_file_path = tf.keras.utils.get_file("train.csv", TRAIN_DATA_URL)
        test_file_path = tf.keras.utils.get_file("eval.csv", TEST_DATA_URL)
        
        # x_train as pandas.DataFrame, y_train as pandas.Series
        x_train = pd.read_csv(train_file_path)
        print(type(x_train)) # pandas.DataFrame
        y_train = x_train.pop('survived')
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
        y_test = x_test.pop('survived')
        
        # Initialize the structured data classifier.
        input_node = ak.StructuredDataInput()
        output_node = ak.StructuredDataBlock(categorical_encoding=True)(input_node)
        output_node = ak.ClassificationHead()(output_node)
        clf = ak.AutoModel(
            inputs=input_node, 
            outputs=output_node, 
            overwrite=True,
            max_trials=3)
        
        # Feed the structured data classifier with training data.
        clf.fit(x_train,
            y_train,
            # Split the training data and use the last 15% as validation data.
            validation_split=0.15,
            epochs=10)
        
        # Predict with the best model.
        predicted_y = clf.predict(test_file_path)
        
        # Evaluate the best model with testing data.
        print(clf.evaluate(test_file_path, 'survived'))
        

class StructuredRegression():
    def train():
        house_dataset = fetch_california_housing()
        df = pd.DataFrame(
            np.concatenate((
                house_dataset.data, 
                house_dataset.target.reshape(-1,1)),
                axis=1),
            columns=house_dataset.feature_names + ['Price'])
        train_size = int(df.shape[0] * 0.9)
        df[:train_size].to_csv('train.csv', index=False)
        df[train_size:].to_csv('eval.csv', index=False)
        train_file_path = 'train.csv'
        test_file_path = 'eval.csv'
        
        # x_train as pandas.DataFrame, y_train as pandas.Series
        x_train = pd.read_csv(train_file_path)
        print(type(x_train)) # pandas.DataFrame
        y_train = x_train.pop('Price')
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
        y_test = x_test.pop('Price')
                
        # Initialize the structured data regressor.
        input_node = ak.StructuredDataInput()
        output_node = ak.StructuredDataBlock(categorical_encoding=True)(input_node)
        output_node = ak.RegressionHead()(output_node)
        reg = ak.AutoModel(
            inputs=input_node, 
            outputs=output_node, 
            overwrite=True,
            max_trials=3)
        
        # Feed the structured data regressor with training data.
        reg.fit(x_train,
            y_train,
            # Split the training data and use the last 15% as validation data.
            validation_split=0.15,
            epochs=10)
       
        # Predict with the best model.
        predicted_y = reg.predict(test_file_path)
        
        # Evaluate the best model with testing data.
        print(reg.evaluate(test_file_path, 'Price'))
    
class VanillaLSTM():
    def train(): 
        #Get a dataset
        zip_path = keras.utils.get_file(
        origin='https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip',
        fname='jena_climate_2009_2016.csv.zip',
        extract=True)
        csv_path, _ = os.path.splitext(zip_path)
        df = pd.read_csv(csv_path)
        df=df.drop(columns=['Date Time'])
        
        #User has to specify feature columns, I chose columns 4 through 7
        X=df.iloc[:, 3:7]
        X=np.array(X)
        
        #Similarly the user has to choose output column, I chose the 2nd one since it's the Temperature(°C)
        y=df.iloc[:,[1]]
        y=np.array(y)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)
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

class AutoARIMA():
    def train(self):
        #Get a dataset. This is Microsoft stock data.
        df = pm.datasets.load_msft()
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
        
        y_train, y_test = train_test_split(y, test_size=0.2)
        exog_train, exog_test = train_test_split(exog, test_size=0.2)
        
        arima = pm.auto_arima(y_train, exog_train, start_p=1, d=None, start_q=1, information_criterion='aic', 
                                   maxiter=100, method='lbfgs', test='kpss', stepwise=True)
        
 
        forecasts = arima.predict(y_test.shape[0], exog_test)
        
        error = smape(y_test, forecasts)
        mae = mean_absolute_error(y_test, forecasts)
        print("Symmetric Mean Absolute Percentage Error: ", error)
        print("Mean Absolute Error: ", mae)
