# Importing functions and classes we'll use

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import SimpleRNN, LSTM, GRU, Dropout, Flatten
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from warnings import catch_warnings
from warnings import filterwarnings
import keras
import sys
import scipy.stats
import json
import numpy.fft
import time
from decimal import Decimal
import math
from math import sqrt
import seaborn as sns
import optuna
from optuna.samplers import TPESampler
from keras.callbacks import EarlyStopping

from .ASAP import *
from .esn import *


def sarima_multistep_forecast(history, config, window_size, n_steps):
    order, sorder, trend = config
    new_hist = history[:]
    yhat = []
    total_train_time = 0
    total_prediction_time = 0
    # define model
    for i in range(n_steps):

      model = SARIMAX(new_hist[-window_size:], order=order, seasonal_order=sorder, trend=trend,
                    enforce_stationarity=False, enforce_invertibility=False)

      # Record the starting time to train the model
      training_start_time = time.time()

      # fit model
      model_fit = model.fit(disp=False)

      # Record the end time from training the model
      training_end_time = time.time()
      elapsed_training_time = training_end_time - training_start_time
      total_train_time += elapsed_training_time

      # make multistep forecast
      # yhat = model_fit.forecast(steps=n_steps)

      # Record the starting time to generate predictions
      predictions_start_time = time.time()

      prediction = model_fit.predict(start=len(history[-window_size:]), end=len(history[-window_size:]))

      # Record the end time from generate predictions
      predictions_end_time = time.time()
      elapsed_predicting_time = predictions_end_time - predictions_start_time
      total_prediction_time += elapsed_predicting_time

      yhat = np.append(yhat, prediction)
      new_hist = np.append(new_hist, prediction)
      new_hist = new_hist[1:]
    return yhat, total_train_time, total_prediction_time


def create_multistep_dataset(data, n_input, n_out=1):
    X, y = list(), list()
    in_start = 0
    # step over the entire history one time step at a time
    for _ in range(len(data)):
        # define the end of the input sequence
        in_end = in_start + n_input
        out_end = in_end + n_out
        # ensure we have enough data for this instance
        if out_end <= len(data):
            x_input = data[in_start:in_end]
            X.append(x_input)
            y.append(data[in_end:out_end])
        # move along one time step
        in_start += 1
    return np.array(X), np.array(y)

def predict_using_models(models, X, model='all'):
    n = len(models)  # Nombre de modèles

    # Initialisation d'une matrice pour stocker les prédictions
    predictions = np.zeros((X.shape[0], n))

    # Record the starting time to generate predictions
    predictions_start_time = time.time()

    if model == 'esn':
      for i, model in enumerate(models):
        # Effectue les prédictions pour le modèle i
        y_pred = model.predict(X)
        predictions[:, i] = np.squeeze(y_pred)
    else :

      for i, model in enumerate(models):
          # Effectue les prédictions pour le modèle i
          y_pred = model.predict(X, verbose=0)
          predictions[:, i] = np.squeeze(y_pred)

    # Record the ending time of generating predictions
    predictions_end_time = time.time()
    predictions_elapsed_time = predictions_end_time - predictions_start_time

    return predictions, predictions_elapsed_time

def find_best_model(data, selected_columns, horizon, TRIALS):

    if horizon==1 and len(selected_columns)==1: # 1-step univariate
        
        def objective(trial):
            # define search space for hyperparameters
            forecasting_model = trial.suggest_categorical('forecasting_model', ['RNN', 'LSTM', 'GRU', 'ESN', 'ESN', 'ARIMA'])

            # In case the model is LSTM, GRU or RNN
            if forecasting_model in ["LSTM", "GRU", "RNN"]:

                variable = data[[selected_columns[0]]]
                variable_dataset = variable.values
                window_size, slide_size = smooth_ASAP(variable_dataset, resolution=50)

                denoised_variable_dataset = moving_average(variable_dataset, window_size)

                # split into train and test sets
                train_size = int(len(denoised_variable_dataset) * 0.9)
                test_size = len(denoised_variable_dataset) - train_size
                # train, test = denoised_variable_dataset[0:train_size], denoised_variable_dataset[train_size:]
                # Here we will use only the last 1000 observations to find the best model and params (800 for train and 200 for test)
                train, test = denoised_variable_dataset[-1000:-200], denoised_variable_dataset[-200:]
                

                num_hidden_layers = trial.suggest_int('num_hidden_layers', 1, 5)
                look_back = trial.suggest_int('look_back', 10, 150)
                learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-1)
                batch_size = trial.suggest_int('batch_size', 10, 150)
                epochs = 3#trial.suggest_int('epochs', 10, 100)

                # reshape into X=t and Y=t+1
                trainX, trainY = create_multistep_dataset(train, look_back, 1)
                validX, validY = create_multistep_dataset(test, look_back, 1)

                # reshape input to be [samples, time steps, features]
                trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
                validX = np.reshape(validX, (validX.shape[0], 1, validX.shape[1]))
                print(trainX.shape)

                # Crée et entraîne le modèle pour l'horizon de prévision i
                model = Sequential()
                for i in range(num_hidden_layers):
                    num_units = trial.suggest_int(f'units_layer_{i}', 8, 256, log=True)
                    return_sequences = (i < num_hidden_layers - 1)
                    if forecasting_model=="RNN":
                        model.add(SimpleRNN(units=num_units, return_sequences=return_sequences))
                    elif forecasting_model=="LSTM":
                        model.add(LSTM(units=num_units, return_sequences=return_sequences))
                    elif forecasting_model=="GRU":
                        model.add(GRU(units=num_units, return_sequences=return_sequences))
                model.add(Dense(1))
                optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
                model.compile(loss='mean_squared_error', optimizer=optimizer)
                model.fit(trainX, trainY, epochs=epochs, batch_size=batch_size, verbose=0)
                validPredict = model.predict(validX, verbose=0)

                # calculate root mean squared error
                validScore = np.sqrt(mean_squared_error(validY, validPredict))

                return validScore
            
            elif forecasting_model == "ESN":
                variable = data[[selected_columns[0]]]
                variable_dataset = variable.values
                window_size, slide_size = smooth_ASAP(variable_dataset, resolution=50)

                denoised_variable_dataset = moving_average(variable_dataset, window_size)

                # split into train and test sets
                train_size = int(len(denoised_variable_dataset) * 0.9)
                test_size = len(denoised_variable_dataset) - train_size
                # train, test = denoised_variable_dataset[0:train_size], denoised_variable_dataset[train_size:]
                # Here we will use only the last 1000 observations to find the best model and params (800 for train and 200 for test)
                train, test = denoised_variable_dataset[-1000:-200], denoised_variable_dataset[-200:]
                
                n_reservoir = trial.suggest_int('n_reservoir', 10, 1000)   # -
                sparsity = trial.suggest_categorical('sparsity', [0.01, 0.1, 0.2, 0.3, 0.4, 0.5])   # -
                spectral_radius = trial.suggest_categorical('spectral_radius', [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.25, 10.0])   # - spectral radius of W
                noise = trial.suggest_categorical('noise', [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009])       # - Noise Set
                look_back = trial.suggest_int('look_back', 10, 150)
                # reshape into X=t and Y=t+1
                trainX, trainY = create_multistep_dataset(train, look_back, 1)
                validX, validY = create_multistep_dataset(test, look_back, 1)

                # reshape input to be [samples, time steps, features]
                trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1]))
                validX = np.reshape(validX, (validX.shape[0], validX.shape[1]))

                # Build and fit the ESN model

                model = ESN(n_inputs = look_back,
                            n_outputs = 1,
                            n_reservoir = n_reservoir,
                            sparsity=sparsity,
                            random_state=1234,
                            spectral_radius=spectral_radius,
                            noise = noise,
                            teacher_scaling = 10)

                # Train and test our model
                pred_train = model.fit(trainX, trainY)
                predictions = model.predict(validX)
                predictions = np.array(predictions)

                # Evaluate the model on the validation set
                val_loss = np.sqrt(mean_squared_error(predictions, validY))

                return val_loss
            
            else:
                variable = data[[selected_columns[0]]]
                variable_dataset = variable.values
                window_size, slide_size = smooth_ASAP(variable_dataset, resolution=50)

                denoised_variable_dataset = moving_average(variable_dataset, window_size)

                # split into train and test sets
                train_size = int(len(denoised_variable_dataset) * 0.9)
                test_size = len(denoised_variable_dataset) - train_size
                # train, test = denoised_variable_dataset[0:train_size], denoised_variable_dataset[train_size:]
                # Here we will use only the last 1000 observations to find the best model and params (800 for train and 200 for test)
                train, test = denoised_variable_dataset[-1000:-200], denoised_variable_dataset[-200:]
                
                # define search space for hyperparameters
                p = trial.suggest_int('p', 0, 3)
                d = trial.suggest_int('d', 0, 2)
                q = trial.suggest_int('q', 0, 3)
                
                window_length = trial.suggest_int('window_length', 5, 150)

                order = (p, d, q)
                seasonal_order = (0, 0, 0, 0)

                cfg = (order, seasonal_order, 'c')

                predictions = []
                ys = []
                # seed history with training dataset
                history = []
                history.extend(train)

                training_time = 0
                prediction_time = 0


                # Record the starting time to generate predictions
                predictions_start_time = time.time()

                # step over each time-step in the test set
                # for i = 0
                # fit model and make forecast for history
                yhat,total_train_time, total_prediction_time = sarima_multistep_forecast(np.array(history), cfg, window_length, horizon)
                training_time += total_train_time
                prediction_time += total_prediction_time
                # store forecast in list of predictions
                predictions.append(yhat)
                ys.append(test[:horizon])
                # add actual observation to history for the next loop
                history.extend(test[:horizon])
                for i in range(horizon, len(test)):
                    # fit model and make forecast for history
                    yhat, total_train_time, total_prediction_time = sarima_multistep_forecast(np.array(history), cfg, window_length, horizon)
                    # store forecast in list of predictions
                    predictions.append(yhat)
                    ys.append(test[i:i+horizon])
                    # add actual observation to history for the next loop
                    history.append(test[i])

                    training_time += total_train_time
                    prediction_time += total_prediction_time

                # Record the ending time of generating predictions
                predictions_end_time = time.time()
                predictions_elapsed_time = predictions_end_time - predictions_start_time

                # estimate prediction error
                ys_converted = [array.tolist() for array in ys if len(array) == horizon]
                predictions_converted = [array.tolist() for array in predictions]

                testRMSE = np.sqrt(mean_squared_error(ys_converted, predictions_converted[:len(ys_converted)]))
                testMAE = mean_absolute_error(ys_converted, predictions_converted[:len(ys_converted)])

                return testRMSE

        study = optuna.create_study(direction='minimize', sampler=TPESampler())

        # Record the starting time to generate predictions
        start_time = time.time()

        study.optimize(objective, n_trials=TRIALS, n_jobs=-1)

        # Record the ending time
        end_time = time.time()
        elapsed_time = end_time - start_time

        print('done')
        print("Model HyperParameters Tuning Elapsed Time : %.5f" % (elapsed_time), "seconds")

        best_params = study.best_params
        best_error = study.best_value
    
    # I am here Now !

    elif horizon>1 and len(selected_columns)==1: # N-step univariate

        # Define the objective function for Optuna optimization
        def objective(trial):
            
            forecasting_model = trial.suggest_categorical('forecasting_model', ['RNN', 'LSTM', 'GRU', 'ESN', 'ESN', 'ARIMA'])
            
            if forecasting_model in ["RNN", "LSTM", "GRU"]:

                forecasting_strategy = "MIMO"

                variable = data[[selected_columns[0]]]
                variable_dataset = variable.values
                window_size, slide_size = smooth_ASAP(variable_dataset, resolution=50)

                denoised_variable_dataset = moving_average(variable_dataset, window_size)

                # split into train and test sets
                train_size = int(len(denoised_variable_dataset) * 0.9)
                test_size = len(denoised_variable_dataset) - train_size
                train, test = denoised_variable_dataset[-1000:-200], denoised_variable_dataset[-200:]
                
                # define search space for hyperparameters
                look_back = trial.suggest_int('look_back', 10, 150)
                num_hidden_layers = trial.suggest_int('num_hidden_layers', 1, 10)

                learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-1)
                batch_size = trial.suggest_int('batch_size', 10, 150)
                epochs = 3

                # reshape into X=t and Y=t+1
                trainX, trainY = create_multistep_dataset(train, look_back, horizon)
                validX, validY = create_multistep_dataset(test, look_back, horizon)

                # reshape input to be [samples, time steps, features]
                trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
                validX = np.reshape(validX, (validX.shape[0], 1, validX.shape[1]))
                print(trainX.shape)

                # Crée et entraîne le modèle pour l'horizon de prévision i
                model = Sequential()
                for i in range(num_hidden_layers):
                    num_units = trial.suggest_int(f'units_layer_{i}', 8, 256, log=True)
                    return_sequences = (i < num_hidden_layers - 1)
                    if forecasting_model=="RNN":
                        model.add(SimpleRNN(units=num_units, return_sequences=return_sequences))
                    elif forecasting_model=="LSTM":
                        model.add(LSTM(units=num_units, return_sequences=return_sequences))
                    elif forecasting_model=="GRU":
                        model.add(GRU(units=num_units, return_sequences=return_sequences))
                model.add(Dense(horizon))
                optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
                model.compile(loss='mean_squared_error', optimizer=optimizer)
                model.fit(trainX, trainY, epochs=epochs, batch_size=batch_size, verbose=0)
                validPredict = model.predict(validX, verbose=0)

                # calculate root mean squared error
                validScore = np.sqrt(mean_squared_error(validY, validPredict))

                return validScore
            
            elif forecasting_model=="ESN":
            
                forecasting_strategy = "Recursive"
                def esn_recursive_strategy(model, X_row, n_steps):
                    forecasts = []

                    for i in range(n_steps):
                        forecast = model.predict(np.array([X_row]))
                        forecasts.append(forecast[0, 0])
                        X_row = X_row.tolist()
                        X_row.append(forecast[0, 0])
                        X_row = X_row[1:]
                        X_row = np.array(X_row)
                    return forecasts
                def esn_make_predictions(model, X, n_steps):
                    predictions = []
                    for i in range(len(X)):
                        row_forecasts = esn_recursive_strategy(model, X[i, :], n_steps)
                        predictions.append(row_forecasts)
                    return predictions
                
                variable = data[[selected_columns[0]]]
                variable_dataset = variable.values
                window_size, slide_size = smooth_ASAP(variable_dataset, resolution=50)

                denoised_variable_dataset = moving_average(variable_dataset, window_size)

                # split into train and test sets
                train_size = int(len(denoised_variable_dataset) * 0.9)
                test_size = len(denoised_variable_dataset) - train_size
                train, test = denoised_variable_dataset[-1000:-200], denoised_variable_dataset[-200:]

                n_reservoir = trial.suggest_int('n_reservoir', 10, 1000)   # -
                sparsity = trial.suggest_categorical('sparsity', [0.01, 0.1, 0.2, 0.3, 0.4, 0.5])   # -
                spectral_radius = trial.suggest_categorical('spectral_radius', [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.25, 10.0])   # - spectral radius of W
                noise = trial.suggest_categorical('noise', [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009])       # - Noise Set
                look_back = trial.suggest_int('look_back', 10, 150)

                # reshape into X=t and Y=t+1
                trainX, trainY = create_multistep_dataset(train, look_back, 1)
                testX, testY = create_multistep_dataset(test, look_back, 1)

                # reshape input to be [samples, time steps, features]
                trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1]))
                testX = np.reshape(testX, (testX.shape[0], testX.shape[1]))

                # Build and fit the ESN model

                model = ESN(n_inputs = look_back,
                            n_outputs = 1,
                            n_reservoir = n_reservoir,
                            sparsity=sparsity,
                            random_state=1234,
                            spectral_radius=spectral_radius,
                            noise = noise,
                            teacher_scaling = 10)

                # Train and test our model
                pred_train = model.fit(trainX, trainY)
                
                predictions_start_time = time.time()

                testPredict = esn_make_predictions(model, testX, horizon)

                # Record the ending time of generating predictions
                predictions_end_time = time.time()
                predictions_elapsed_time = predictions_end_time - predictions_start_time

                testPredict = np.array(testPredict)
                _, new_testY = create_multistep_dataset(test, look_back, horizon)

                testRMSE = np.sqrt(mean_squared_error(new_testY, testPredict[:len(new_testY), :]))
                testMAE = mean_absolute_error(new_testY, testPredict[:len(new_testY), :])

                # Evaluate the model on the validation set
                val_loss = testRMSE

                return val_loss
            
            else :
                variable = data[[selected_columns[0]]]
                variable_dataset = variable.values
                window_size, slide_size = smooth_ASAP(variable_dataset, resolution=50)

                denoised_variable_dataset = moving_average(variable_dataset, window_size)

                # split into train and test sets
                train_size = int(len(denoised_variable_dataset) * 0.9)
                test_size = len(denoised_variable_dataset) - train_size
                # train, test = denoised_variable_dataset[0:train_size], denoised_variable_dataset[train_size:]
                # Here we will use only the last 1000 observations to find the best model and params (800 for train and 200 for test)
                train, test = denoised_variable_dataset[-1000:-200], denoised_variable_dataset[-200:]
                
                # define search space for hyperparameters
                p = trial.suggest_int('p', 0, 3)
                d = trial.suggest_int('d', 0, 2)
                q = trial.suggest_int('q', 0, 3)
                
                window_length = trial.suggest_int('window_length', 5, 150)

                order = (p, d, q)
                seasonal_order = (0, 0, 0, 0)

                cfg = (order, seasonal_order, 'c')

                predictions = []
                ys = []
                # seed history with training dataset
                history = []
                history.extend(train)

                training_time = 0
                prediction_time = 0


                # Record the starting time to generate predictions
                predictions_start_time = time.time()

                # step over each time-step in the test set
                # for i = 0
                # fit model and make forecast for history
                yhat,total_train_time, total_prediction_time = sarima_multistep_forecast(np.array(history), cfg, window_length, horizon)
                training_time += total_train_time
                prediction_time += total_prediction_time
                # store forecast in list of predictions
                predictions.append(yhat)
                ys.append(test[:horizon])
                # add actual observation to history for the next loop
                history.extend(test[:horizon])
                for i in range(horizon, len(test)):
                    # fit model and make forecast for history
                    yhat, total_train_time, total_prediction_time = sarima_multistep_forecast(np.array(history), cfg, window_length, horizon)
                    # store forecast in list of predictions
                    predictions.append(yhat)
                    ys.append(test[i:i+horizon])
                    # add actual observation to history for the next loop
                    history.append(test[i])

                    training_time += total_train_time
                    prediction_time += total_prediction_time

                # Record the ending time of generating predictions
                predictions_end_time = time.time()
                predictions_elapsed_time = predictions_end_time - predictions_start_time

                # estimate prediction error
                ys_converted = [array.tolist() for array in ys if len(array) == horizon]
                predictions_converted = [array.tolist() for array in predictions]

                testRMSE = np.sqrt(mean_squared_error(ys_converted, predictions_converted[:len(ys_converted)]))
                testMAE = mean_absolute_error(ys_converted, predictions_converted[:len(ys_converted)])

                return testRMSE

        # Create the Optuna study
        study = optuna.create_study(direction='minimize')

        # Record the starting time to generate predictions
        start_time = time.time()

        # Run the optimization
        study.optimize(objective, n_trials=TRIALS)

        # Record the ending time
        end_time = time.time()
        elapsed_time = end_time - start_time

        print('done')
        print("Model HyperParameters Tuning Elapsed Time : %.5f" % (elapsed_time), "seconds")

        # Print the best parameters and corresponding loss
        best_params = study.best_params
        best_loss = study.best_value


    # ARIMA Model don't need training
    if best_params["forecasting_model"] == "ARIMA":
        return None, best_params, None

    # Training the model, is needed

    if horizon==1 and len(selected_columns)==1: # 1-step univariate
        if best_params["forecasting_model"] in ["RNN", "LSTM", "GRU"]:

            variable = data[[selected_columns[0]]]
            variable_dataset = variable.values
            window_size, slide_size = smooth_ASAP(variable_dataset, resolution=50)

            denoised_variable_dataset = moving_average(variable_dataset, window_size)
            
            # define hyperparameters
            look_back = best_params["look_back"]
            num_hidden_layers = best_params["num_hidden_layers"]

            learning_rate = best_params["learning_rate"]
            batch_size = best_params["batch_size"]
            # epochs = best_params["epochs"]
            # reshape into X=t and Y=t+1
            trainX, trainY = create_multistep_dataset(denoised_variable_dataset, look_back, 1)

            # reshape input to be [samples, time steps, features]
            trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
            # Crée et entraîne le modèle pour l'horizon de prévision i
            model = Sequential()
            for i in range(num_hidden_layers):
                num_units = best_params[f'units_layer_{i}']
                return_sequences = (i < num_hidden_layers - 1)
                if best_params["forecasting_model"]=="RNN":
                    model.add(SimpleRNN(units=num_units, return_sequences=return_sequences))
                elif best_params["forecasting_model"]=="LSTM":
                    model.add(LSTM(units=num_units, return_sequences=return_sequences))
                elif best_params["forecasting_model"]=="GRU":
                    model.add(GRU(units=num_units, return_sequences=return_sequences))
            model.add(Dense(1))
            optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
            model.compile(loss='mean_squared_error', optimizer=optimizer)

            # Record the starting time to train the model
            training_start_time = time.time()

            early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
            # Train our model
            model.fit(trainX, trainY, epochs=100, batch_size=batch_size, verbose=0, callbacks=[early_stopping], validation_split=0.1)
            
            # Record the ending time
            training_end_time = time.time()
            training_elapsed_time = training_end_time - training_start_time
            
            return model, best_params, training_elapsed_time
        
        elif best_params["forecasting_model"] == "ESN":

            variable = data[[selected_columns[0]]]
            variable_dataset = variable.values
            window_size, slide_size = smooth_ASAP(variable_dataset, resolution=50)

            denoised_variable_dataset = moving_average(variable_dataset, window_size)

            # Define the hyperparameters
            n_reservoir = best_params["n_reservoir"]   # -
            sparsity = best_params["sparsity"]   # -
            spectral_radius = best_params["spectral_radius"]   # - spectral radius of W
            noise = best_params["noise"]   # - Noise Set
            look_back = best_params["look_back"]

            # reshape into X=t and Y=t+1
            trainX, trainY = create_multistep_dataset(denoised_variable_dataset, look_back, 1)

            # reshape input to be [samples, time steps, features]
            trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1]))

            # Build and fit the ESN model

            model = ESN(n_inputs = look_back,
                        n_outputs = 1,
                        n_reservoir = n_reservoir,
                        sparsity=sparsity,
                        random_state=1234,
                        spectral_radius=spectral_radius,
                        noise = noise,
                        teacher_scaling = 10)
            

            # Record the starting time to train the model
            training_start_time = time.time()

            # Train our model
            pred_train = model.fit(trainX, trainY)
            
            # Record the ending time
            training_end_time = time.time()
            training_elapsed_time = training_end_time - training_start_time

            return model, best_params, training_elapsed_time

    elif horizon>1 and len(selected_columns)==1: # N-step univariate
        if best_params["forecasting_model"] in ["RNN", "LSTM", "GRU"]:
            variable = data[[selected_columns[0]]]
            variable_dataset = variable.values
            window_size, slide_size = smooth_ASAP(variable_dataset, resolution=50)

            denoised_variable_dataset = moving_average(variable_dataset, window_size)

            # define hyperparameters
            look_back = best_params["look_back"]
            num_hidden_layers = best_params["num_hidden_layers"]

            learning_rate = best_params["learning_rate"]
            batch_size = best_params["batch_size"]
            # epochs = best_params["epochs"]

            # reshape into X=t and Y=t+1
            trainX, trainY = create_multistep_dataset(denoised_variable_dataset, look_back, horizon)

            # reshape input to be [samples, time steps, features]
            trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))

            # Crée et entraîne le modèle pour l'horizon de prévision i
            model = Sequential()
            for i in range(num_hidden_layers):
                num_units = best_params[f'units_layer_{i}']
                return_sequences = (i < num_hidden_layers - 1)
                if best_params["forecasting_model"]=="RNN":
                    model.add(SimpleRNN(units=num_units, return_sequences=return_sequences))
                elif best_params["forecasting_model"]=="LSTM":
                    model.add(LSTM(units=num_units, return_sequences=return_sequences))
                elif best_params["forecasting_model"]=="GRU":
                    model.add(GRU(units=num_units, return_sequences=return_sequences))
            model.add(Dense(horizon))
            optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
            model.compile(loss='mean_squared_error', optimizer=optimizer)

            # Record the starting time to train the model
            training_start_time = time.time()
            
            early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
            # Train our model
            model.fit(trainX, trainY, epochs=100, batch_size=batch_size, verbose=0, callbacks=[early_stopping], validation_split=0.1)
            
            # Record the ending time
            training_end_time = time.time()
            training_elapsed_time = training_end_time - training_start_time

            return model, best_params, training_elapsed_time
        
        if best_params["forecasting_model"] == "ESN":

            def esn_recursive_strategy(model, X_row, n_steps):
                forecasts = []

                for i in range(n_steps):
                    forecast = model.predict(np.array([X_row]))
                    forecasts.append(forecast[0, 0])
                    X_row = X_row.tolist()
                    X_row.append(forecast[0, 0])
                    X_row = X_row[1:]
                    X_row = np.array(X_row)
                return forecasts
            def esn_make_predictions(model, X, n_steps):
                predictions = []
                for i in range(len(X)):
                    row_forecasts = esn_recursive_strategy(model, X[i, :], n_steps)
                    predictions.append(row_forecasts)
                return predictions

            variable = data[[selected_columns[0]]]
            variable_dataset = variable.values
            window_size, slide_size = smooth_ASAP(variable_dataset, resolution=50)

            denoised_variable_dataset = moving_average(variable_dataset, window_size)

            # Define the hyperparameters
            n_reservoir = best_params["n_reservoir"]   # -
            sparsity = best_params["sparsity"]   # -
            spectral_radius = best_params["spectral_radius"]   # - spectral radius of W
            noise = best_params["noise"]   # - Noise Set
            look_back = best_params["look_back"]

            # reshape into X=t and Y=t+1
            trainX, trainY = create_multistep_dataset(denoised_variable_dataset, look_back, 1)

            # reshape input to be [samples, time steps, features]
            trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1]))

            # Build and fit the ESN model

            model = ESN(n_inputs = look_back,
                        n_outputs = 1,
                        n_reservoir = n_reservoir,
                        sparsity=sparsity,
                        random_state=1234,
                        spectral_radius=spectral_radius,
                        noise = noise,
                        teacher_scaling = 10)

            # Record the starting time to train the model
            training_start_time = time.time()

            # Train our model
            pred_train = model.fit(trainX, trainY)
            
            # Record the ending time
            training_end_time = time.time()
            training_elapsed_time = training_end_time - training_start_time

            return model, best_params, training_elapsed_time

        
