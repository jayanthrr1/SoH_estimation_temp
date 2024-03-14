#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 09:54:25 2024

@author: jay
"""


import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from scipy.stats import pearsonr, spearmanr

#%% Plotting functions

# Function to calculate Mean Absolute Percentage Error (MAPE) and Root Mean Squared Error (RMSE)
def calculate_prediction_error(y_pred, y_true):
    mape = []
    rmse = []
    for i in range(0,np.size(y_pred, axis=1)):
        mape.append(np.round(mean_absolute_percentage_error(y_true[:,i], y_pred[:,i]) * 100, 3))
        rmse.append(np.round(mean_squared_error(y_true[:,i], y_pred[:,i], squared=False), 4))
    mape = np.vstack(mape)
    rmse = np.vstack(rmse)
    return mape.reshape(-1), rmse.reshape(-1)


# Function to plot correlation between features and targets, and display MAPE and RMSE
def plot_correlation_auto_sizing(feature, target, xlabel, ylabel, cmap='Blues'):
    pearson_corrs = (np.round(pearsonr(feature, target)[0] ** 2, 3))
    spearman_corrs = (np.round(spearmanr(feature, target)[0], 3))
    
    mape = np.round(mean_absolute_percentage_error(target, feature) * 100, 3)
    rmse = np.round(mean_squared_error(target, feature, squared=False), 3)
    
    fig, ax = plt.subplots(nrows=1, ncols=1, sharex=False, sharey=False, figsize=(4,4), dpi=300, squeeze=True)
    fig.tight_layout(pad=0)
    title_pad = 10
    fontsize = 14
    a = 0
    ax.scatter(feature, target, c=target, s=25, marker='o', cmap=cmap, edgecolors='k')
    ax.plot(np.linspace(np.min(target), np.max(target), 100), np.linspace(np.min(target), np.max(target), 100), linewidth=1, color='k')
    ax.set_aspect('equal', adjustable='datalim')
    # ax.set_xlim([np.min(target)-(0.01*np.min(target)), np.max(target)+(0.01*np.max(target))])
    # ax.set_ylim([np.min(target)-(0.01*np.min(target)), np.max(target)+(0.01*np.max(target))])
    matplotlib.rc('xtick', labelsize=fontsize-2)
    matplotlib.rc('ytick', labelsize=fontsize-2)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    # ax.set_title(title + '\n$\rho=${}  S: {}'.format(pearson_corrs, spearman_corrs), fontsize=fontsize)
    ax.set_title(r'$R^2={}$  MAPE: {}  RMSE: {}'.format(pearson_corrs, mape, rmse), fontsize=fontsize)
    plt.show()
    return