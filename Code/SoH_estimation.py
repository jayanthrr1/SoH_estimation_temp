#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 09:54:25 2024

@author: jay
"""

#%% Import libraries and change directory
import numpy as np

import matplotlib.pyplot as plt
# import pickle5 as pickle
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
import os
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import (
    MaxAbsScaler,
    MinMaxScaler,
    StandardScaler,
)

import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.callbacks import EarlyStopping

# Set the plotting environment font.
plt.rcParams['font.family'] = 'Arial'

#Locate the file
full_path = os.path.realpath(__file__)
#Change the working directory
path = os.path.dirname(full_path)
os.chdir(path)

# Import helper functions
from pulse_current import create_pulse_dataset
from constant_current import create_cc_dataset
from Rapid_SoH_functions import calculate_prediction_error, plot_correlation_auto_sizing

# Plot settings
plotting = True

# SPECIFY THE INPUT TYPE HERE.
# Possible options:
    # pulse : uses voltage response corresponding to pulse current
    # cc : uses voltage reponse corresponding to constant current
    
input_type = 'cc'

#%% Get data based on input type

# Possible output options (resistance take the form 'r' followed by the current [C3, C4, C5] and SoC [30,50,70] ) :
    # soh
    # r_C3_30soc
    # r_C3_50soc
    # r_C3_70soc
    # r_C4_30soc
    # r_C4_50soc
    # r_C4_70soc
    # r_C5_30soc
    # r_C5_50soc
    # r_C5_70soc

# default is to inlude all the outputs    
outputs = ['soh', 'r_C3_30soc', 'r_C3_50soc', 'r_C3_70soc', \
                  'r_C4_30soc', 'r_C4_50soc', 'r_C4_70soc', \
                  'r_C5_30soc', 'r_C5_50soc', 'r_C5_70soc']
    

# Pulse inputs
# Possible input options (Inputs take the form 'pulseV' followed by the current [C3, C4, C5] and SoC [30,50,70] ) : 
    # pulseV_C3_30soc
    # pulseV_C3_50soc
    # pulseV_C3_70soc
    # pulseV_C4_30soc
    # pulseV_C4_50soc
    # pulseV_C4_70soc
    # pulseV_C5_30soc
    # pulseV_C5_50soc
    # pulseV_C5_70soc
    
# CC inputs
# Possible parameters to tune:
    # soc1 : starting soc (default = 0.2)
    # soc2 : ending soc (default = 0.7)
    # lookback : time of the section (defaults = 600)
    # soh_limit : soh limit (defaults = 0.8)
    

if input_type == 'pulse':    
    # modify this list to choose the sepcific inputs; default is to include all of them    
    inputs = ['pulseV_C3_30soc', 'pulseV_C3_50soc', 'pulseV_C3_70soc', \
              'pulseV_C4_30soc', 'pulseV_C4_50soc', 'pulseV_C4_70soc', \
              'pulseV_C5_30soc', 'pulseV_C5_50soc', 'pulseV_C5_70soc']

    dataset = create_pulse_dataset(inputs = inputs, outputs = outputs, soh_limit = 0.8)

elif input_type == 'cc':
    # modify the function parameters to choose specific sections; for default values leave them unchanged
    dataset = create_cc_dataset(soc1 = 0.2, soc2 = 0.7, lookback = 600,  soh_limit = 0.8, outputs = outputs)

#######################################################################################################################################################
#%% Define model

def create_model(input_dim, hidden_layers, neurons, activation, optimizer):
    model = keras.models.Sequential()
    model.add(Dense(units=neurons, activation=activation, input_dim=input_dim))

    for i in range(hidden_layers):
        model.add(Dense(units=neurons, activation=activation))

    model.add(Dense(10))
    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.mean_squared_error,
                  metrics=['accuracy'])
    
    return model

stop = EarlyStopping(monitor='loss', min_delta=0, 
                     patience=5, verbose=1, mode='auto',
                     baseline=None, restore_best_weights=True)

settings = {'hidden_layers': 5, 'neurons': 500, 'activation': 'relu', 'optimizer': 'adam', 'batch_size': 100, 'nb_epoch': 500}

# Find GPU
# gpus = tf.config.list_physical_devices('GPU')
# print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

#%% Model training

# Get cell names
cell_numbers = np.unique(dataset['stacked_cell_numbers'])

consolidated_results = {
     'mape_train_dummy':[],
     'rmse_train_dummy':[],
     'mape_test_dummy':[],
     'rmse_test_dummy':[],
     
     'mape_train_pinn':[],
     'rmse_train_pinn':[],
     'mape_test_pinn':[],
     'rmse_test_pinn':[],
     
     'y_train_true':[],
     'y_train_pred':[],
     'y_test_true':[],
     'y_test_pred':[],
     
     
     'r2_train_pinn':[],
     'r2_test_pinn':[],
     }

# Specify cross validation function
kf = KFold(n_splits=5, shuffle=True, random_state=1)
 
# Looping over each fold
i = 0 
for train_cells, test_cells in kf.split(cell_numbers):
    # print(train_cells)
    i = i + 1
    print('Running fold {}'.format(i))
    
    # train, test, prediction changes for each fold and are stored temporarily in fold_vars
    fold_vars = {}
    
    fold_vars['train_idxs'] = np.where(dataset['stacked_cell_numbers'] == train_cells)[0]
    fold_vars['test_idxs'] = np.where(dataset['stacked_cell_numbers'] == test_cells)[0]
   
    # Create the inputs
    # fold_vars['X_scaler'] = MinMaxScaler(feature_range=(0, 1))
    fold_vars['X_scaler'] = StandardScaler()

    
    #Scaling by rows
    fold_vars['X_train'] = dataset['stacked_relative_inputs'][fold_vars['train_idxs']]
    fold_vars['X_test'] = dataset['stacked_relative_inputs'][fold_vars['test_idxs']]
    fold_vars['X_train_sc'] = fold_vars['X_scaler'].fit_transform(fold_vars['X_train'])
    fold_vars['X_test_sc'] = fold_vars['X_scaler'].transform(fold_vars['X_test'])
    
    
    # Create the targets
    # fold_vars['y_scaler'] = MinMaxScaler(feature_range=(0, 1))
    fold_vars['y_scaler'] = StandardScaler()
    fold_vars['y_train'] = dataset['stacked_outputs'][fold_vars['train_idxs']]
    fold_vars['y_test'] = dataset['stacked_outputs'][fold_vars['test_idxs']]
    
    # Scaled outputes
    fold_vars['y_train_sc'] = fold_vars['y_scaler'].fit_transform(fold_vars['y_train'])
    fold_vars['y_test_sc'] = fold_vars['y_scaler'].transform(fold_vars['y_test'])
    
    
    # # Specify training and test data
    # trainX = dataset['X_train_sc']
    # testX = dataset['X_test_sc']
    # y_sc = dataset['y_train_sc']
    
   
    # Training ################################################################
   
    # Dummy Model Training Data
    fold_vars['y_train_true'] = fold_vars['y_train']
    fold_vars['y_train_pred'] = np.tile(np.mean(fold_vars['y_train_true'], axis=0), (len(fold_vars['y_train_true']),1))
    fold_vars['mape_train_dummy'], fold_vars['rmse_train_dummy'] = calculate_prediction_error(fold_vars['y_train_pred'], fold_vars['y_train_true'])
    
    # Dummy Model Test Data
    fold_vars['y_test_true'] = fold_vars['y_test']
    fold_vars['y_test_pred'] = np.tile(np.mean(fold_vars['y_train_true'], axis=0), (len(fold_vars['y_test_true']),1)) # y_pred is still mean of training data
    fold_vars['mape_test_dummy'], fold_vars['rmse_test_dummy'] = calculate_prediction_error(fold_vars['y_test_pred'], fold_vars['y_test_true'])
    
    
    # Keras sequential 
    model = create_model(input_dim = np.shape(fold_vars['X_train'])[1], hidden_layers = settings['hidden_layers'], 
                         neurons = settings['neurons'], 
                         activation = settings['activation'], 
                         optimizer = settings['optimizer'])

    
    history = model.fit(fold_vars['X_train_sc'], fold_vars['y_train_sc'],
                        validation_data=(fold_vars['X_test_sc'], fold_vars['y_test_sc']),
                        epochs=settings['nb_epoch'],
                        batch_size=settings['batch_size'],
                        callbacks=stop,
                        verbose = 0)
    
    # Extract training and test loss from history
    train_loss = history.history['loss']
    test_loss = history.history['val_loss']
    epochs = range(1, len(train_loss) + 1)
    
    #plotting the loss vs epoch
    plt.plot(epochs, train_loss, 'b', label='Training loss')
    plt.plot(epochs, test_loss, 'r', label='Validation loss')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
    
    
    
    # Calculate the overall error for this fold  ##############################
    fold_vars['y_train_pred'] = fold_vars['y_scaler'].inverse_transform(model.predict(fold_vars['X_train_sc']))
    
    if plotting:
        
        plot_correlation_auto_sizing(fold_vars['y_train_pred'][:,0], fold_vars['y_train_true'][:,0], 'Predicted Capacity [Ah]', 'True Capacity [Ah]')
        
        # plot_correlation_auto_sizing(fold_vars['y_train_pred'][:,1] * 1000, fold_vars['y_train_true'][:,1] * 1000, 'Predicted R_2A_30SOC [mOhm]', 'True R_2A_30SOC [mOhm]')
        # plot_correlation_auto_sizing(fold_vars['y_train_pred'][:,2] * 1000, fold_vars['y_train_true'][:,2] * 1000, 'Predicted R_2A_50SOC [mOhm]', 'True R_2A_50SOC [mOhm]')
        # plot_correlation_auto_sizing(fold_vars['y_train_pred'][:,3] * 1000, fold_vars['y_train_true'][:,3] * 1000, 'Predicted R_2A_70SOC [mOhm]', 'True R_2A_70SOC [mOhm]')
        
        # plot_correlation_auto_sizing(fold_vars['y_train_pred'][:,4] * 1000, fold_vars['y_train_true'][:,4] * 1000, 'Predicted R_4A_30SOC [mOhm]', 'True R_4A_30SOC [mOhm]')
        # plot_correlation_auto_sizing(fold_vars['y_train_pred'][:,5] * 1000, fold_vars['y_train_true'][:,5] * 1000, 'Predicted R_4A_50SOC [mOhm]', 'True R_4A_50SOC [mOhm]')
        # plot_correlation_auto_sizing(fold_vars['y_train_pred'][:,6] * 1000, fold_vars['y_train_true'][:,6] * 1000, 'Predicted R_4A_70SOC [mOhm]', 'True R_4A_70SOC [mOhm]')
        
        # plot_correlation_auto_sizing(fold_vars['y_train_pred'][:,7] * 1000, fold_vars['y_train_true'][:,4] * 1000, 'Predicted R_6A_30SOC [mOhm]', 'True R_6A_30SOC [mOhm]')
        # plot_correlation_auto_sizing(fold_vars['y_train_pred'][:,8] * 1000, fold_vars['y_train_true'][:,5] * 1000, 'Predicted R_6A_50SOC [mOhm]', 'True R_6A_50SOC [mOhm]')
        # plot_correlation_auto_sizing(fold_vars['y_train_pred'][:,9] * 1000, fold_vars['y_train_true'][:,6] * 1000, 'Predicted R_6A_70SOC [mOhm]', 'True R_6A_70SOC [mOhm]')
    
    
    fold_vars['y_test_pred'] = fold_vars['y_scaler'].inverse_transform(model.predict(fold_vars['X_test_sc']))
    
    if plotting:
        plot_correlation_auto_sizing(fold_vars['y_test_pred'][:,0], fold_vars['y_test_true'][:,0], 'Predicted Capacity [Ah]', 'True Capacity [Ah]', cmap = 'Reds')
        
        # plot_correlation_auto_sizing(fold_vars['y_test_pred'][:,1] * 1000, fold_vars['y_test_true'][:,1] * 1000, 'Predicted R_2A_30SOC [mOhm]', 'True R_2A_30SOC [mOhm]', cmap = 'Reds')
        # plot_correlation_auto_sizing(fold_vars['y_test_pred'][:,2] * 1000, fold_vars['y_test_true'][:,2] * 1000, 'Predicted R_2A_50SOC [mOhm]', 'True R_2A_50SOC [mOhm]', cmap = 'Reds')
        # plot_correlation_auto_sizing(fold_vars['y_test_pred'][:,3] * 1000, fold_vars['y_test_true'][:,3] * 1000, 'Predicted R_2A_70SOC [mOhm]', 'True R_2A_70SOC [mOhm]', cmap = 'Reds')
        
        # plot_correlation_auto_sizing(fold_vars['y_test_pred'][:,4] * 1000, fold_vars['y_test_true'][:,4] * 1000, 'Predicted R_4A_30SOC [mOhm]', 'True R_4A_30SOC [mOhm]', cmap = 'Reds')
        # plot_correlation_auto_sizing(fold_vars['y_test_pred'][:,5] * 1000, fold_vars['y_test_true'][:,5] * 1000, 'Predicted R_4A_50SOC [mOhm]', 'True R_4A_50SOC [mOhm]', cmap = 'Reds')
        # plot_correlation_auto_sizing(fold_vars['y_test_pred'][:,6] * 1000, fold_vars['y_test_true'][:,6] * 1000, 'Predicted R_4A_70SOC [mOhm]', 'True R_4A_70SOC [mOhm]', cmap = 'Reds')
        
        # plot_correlation_auto_sizing(fold_vars['y_test_pred'][:,7] * 1000, fold_vars['y_test_true'][:,4] * 1000, 'Predicted R_6A_30SOC [mOhm]', 'True R_6A_30SOC [mOhm]', cmap = 'Reds')
        # plot_correlation_auto_sizing(fold_vars['y_test_pred'][:,8] * 1000, fold_vars['y_test_true'][:,5] * 1000, 'Predicted R_6A_50SOC [mOhm]', 'True R_6A_50SOC [mOhm]', cmap = 'Reds')
        # plot_correlation_auto_sizing(fold_vars['y_test_pred'][:,9] * 1000, fold_vars['y_test_true'][:,6] * 1000, 'Predicted R_6A_70SOC [mOhm]', 'True R_6A_70SOC [mOhm]', cmap = 'Reds')
    
    fold_vars['mape_train_pinn'], fold_vars['rmse_train_pinn'] = calculate_prediction_error(fold_vars['y_train_pred'], fold_vars['y_train_true'])
    fold_vars['mape_test_pinn'], fold_vars['rmse_test_pinn'] = calculate_prediction_error(fold_vars['y_test_pred'], fold_vars['y_test_true'])   
    
    
    # # Errors in prediction of SOH
    print('MAPE of fold {} : {}'.format(i , fold_vars['mape_test_pinn'][0]) )
    print('RMSE of fold {} : {}'.format(i , fold_vars['rmse_test_pinn'][0]) )
    
    consolidated_results['mape_train_dummy'].append(fold_vars['mape_train_dummy'])
    consolidated_results['rmse_train_dummy'].append(fold_vars['rmse_train_dummy'])
    consolidated_results['mape_test_dummy'].append(fold_vars['mape_test_dummy'])
    consolidated_results['rmse_test_dummy'].append(fold_vars['rmse_test_dummy'])
    
    consolidated_results['mape_train_pinn'].append(fold_vars['mape_train_pinn'])
    consolidated_results['rmse_train_pinn'].append(fold_vars['rmse_train_pinn'])
    consolidated_results['mape_test_pinn'].append(fold_vars['mape_test_pinn'])
    consolidated_results['rmse_test_pinn'].append(fold_vars['rmse_test_pinn'])
    
    consolidated_results['y_train_true'].append(fold_vars['y_train_true'])
    consolidated_results['y_train_pred'].append(fold_vars['y_train_pred'])
    consolidated_results['y_test_true'].append(fold_vars['y_test_true'])
    consolidated_results['y_test_pred'].append(fold_vars['y_test_pred'])
    
    consolidated_results['r2_train_pinn'].append(r2_score(fold_vars['y_train_true'], fold_vars['y_train_pred']))
    consolidated_results['r2_test_pinn'].append(r2_score(fold_vars['y_test_true'], fold_vars['y_test_pred']))
    
print('MAPE_dummy \t', *np.mean(consolidated_results['mape_test_dummy'], axis = 0).tolist(), sep='\t')
print('MAPE_dummy_std \t', *np.std(consolidated_results['mape_test_dummy'], axis = 0).tolist(), sep='\t')
print('MAPE_PINN \t', *np.mean(consolidated_results['mape_test_pinn'], axis = 0).tolist(), sep='\t')
print('MAPE_PINN_std \t', *np.std(consolidated_results['mape_test_pinn'], axis = 0).tolist(), sep='\t')

print('RMSE_dummy \t', *np.mean(consolidated_results['rmse_test_dummy'], axis = 0).tolist(), sep='\t')
print('RMSE_dummy_std \t', *np.std(consolidated_results['rmse_test_dummy'], axis = 0).tolist(), sep='\t')
print('RMSE_PINN \t', *np.mean(consolidated_results['rmse_test_pinn'], axis = 0).tolist(), sep='\t')
print('RMSE_PINN_std \t', *np.std(consolidated_results['rmse_test_pinn'], axis = 0).tolist(), sep='\t')

print('r2_train_pinn \t', np.mean( consolidated_results['r2_train_pinn'] ) )
print('r2_test_pinn \t', np.mean( consolidated_results['r2_test_pinn'] ) )

#Save the model
# date = datetime.now().strftime("%d%b%Y %Hh%Mm")
# joblib.dump(model,"Saved Models/Pulses/KerasSeq_" + date + ".pkl")
# model.save("Saved Models/Pulses/KerasSeq_02Nov2023.h5")
