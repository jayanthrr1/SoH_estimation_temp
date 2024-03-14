#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 10:01:04 2024

@author: jay
"""

import pandas as pd
import numpy as np

#%% Get pulse data

# Function to get pulse data of all cells along with identifiers
def get_pulse_data(soh_limit = 0.8):
    
    #read voltage profiles and resistances of all the cells
    pulse_voltage_profiles_dic = pd.read_pickle(r'../Rawdata/pulse_voltage_profiles.pkl')
    pulse_resistances_dic = pd.read_pickle(r'../Rawdata/pulse_resistances.pkl')

    #Correcting the cell names to remove spaces and proper capitalization
    pulse_voltage_profiles_dic = { k.replace(' ', ''): v for k, v in pulse_voltage_profiles_dic.items() }
    pulse_voltage_profiles_dic = { k.replace('s', 'S'): v for k, v in pulse_voltage_profiles_dic.items() }

    cell_names = list(pulse_voltage_profiles_dic.keys())
    # cell_names = list(pulse_resistances_dic.keys())
    dataset_ = {}
    r_C3_30soc = []
    r_C3_50soc = []
    r_C3_70soc = []
    
    r_C4_30soc = []
    r_C4_50soc = []
    r_C4_70soc = []
    
    r_C5_30soc = []
    r_C5_50soc = []
    r_C5_70soc = []
    
    soh = []
    
    pulseV_C3_30soc = []
    pulseV_C3_50soc = []
    pulseV_C3_70soc = []
    
    pulseV_C4_30soc = []
    pulseV_C4_50soc = []
    pulseV_C4_70soc = []
    
    pulseV_C5_30soc = []
    pulseV_C5_50soc = []
    pulseV_C5_70soc = []
    
    cell_ID = []
    cell_numbers = []
    
    # Append the cell info and voltage profiles for each cell
    for cell_num, cell_name in enumerate(cell_names):
        
        initial_capacity = pulse_resistances_dic[cell_name]['C4']['30SOC']['dchg_cap'][0]
        
        # gets all RPT for each cell
        # RPTs = np.arange(0, len(pulse_voltage_profiles_dic[cell_name]['C4']['50SOC']), 1)
        
        # gets all RPT within specifed soh limit
        RPTs = len([ item for item in pulse_resistances_dic[cell_name]['C4']['30SOC']['dchg_cap'] if item >= initial_capacity * soh_limit ])
        RPTs = np.arange(0, RPTs, 1) # converting to range
        
        # each RPT row must have a corresponding cell_ID and cell_number
        cell_ID.append(np.repeat(cell_name, len(RPTs)))
        cell_numbers.append(np.repeat(cell_num, len(RPTs)))
        
        # append the voltage profiles for each RPT of one cell
        for RPT in RPTs:
            pulseV_C3_30soc.append(pulse_voltage_profiles_dic[cell_name]['C3']['30SOC'][RPT])
            pulseV_C3_50soc.append(pulse_voltage_profiles_dic[cell_name]['C3']['50SOC'][RPT])
            pulseV_C3_70soc.append(pulse_voltage_profiles_dic[cell_name]['C3']['70SOC'][RPT])
            
            pulseV_C4_30soc.append(pulse_voltage_profiles_dic[cell_name]['C4']['30SOC'][RPT])
            pulseV_C4_50soc.append(pulse_voltage_profiles_dic[cell_name]['C4']['50SOC'][RPT])
            pulseV_C4_70soc.append(pulse_voltage_profiles_dic[cell_name]['C4']['70SOC'][RPT])
            
            pulseV_C5_30soc.append(pulse_voltage_profiles_dic[cell_name]['C5']['30SOC'][RPT])
            pulseV_C5_50soc.append(pulse_voltage_profiles_dic[cell_name]['C5']['50SOC'][RPT])
            pulseV_C5_70soc.append(pulse_voltage_profiles_dic[cell_name]['C5']['70SOC'][RPT])
            
            r_C3_30soc.append(pulse_resistances_dic[cell_name]['C3']['30SOC']['resistance'][RPT])
            r_C3_50soc.append(pulse_resistances_dic[cell_name]['C3']['50SOC']['resistance'][RPT])
            r_C3_70soc.append(pulse_resistances_dic[cell_name]['C3']['70SOC']['resistance'][RPT])
            
            r_C4_30soc.append(pulse_resistances_dic[cell_name]['C4']['30SOC']['resistance'][RPT])
            r_C4_50soc.append(pulse_resistances_dic[cell_name]['C4']['50SOC']['resistance'][RPT])
            r_C4_70soc.append(pulse_resistances_dic[cell_name]['C4']['70SOC']['resistance'][RPT])
            
            r_C5_30soc.append(pulse_resistances_dic[cell_name]['C5']['30SOC']['resistance'][RPT])
            r_C5_50soc.append(pulse_resistances_dic[cell_name]['C5']['50SOC']['resistance'][RPT])
            r_C5_70soc.append(pulse_resistances_dic[cell_name]['C5']['70SOC']['resistance'][RPT])
            
            
            soh.append(pulse_resistances_dic[cell_name]['C4']['50SOC']['dchg_cap'][RPT])
    
    cell_ID = np.hstack(cell_ID)        
    cell_ID = cell_ID.reshape(-1,1)
    
    cell_numbers = np.hstack(cell_numbers)        
    cell_numbers = cell_numbers.reshape(-1,1)
    
    r_C3_30soc = np.hstack(r_C3_30soc)
    r_C3_30soc = r_C3_30soc.reshape(-1,1)
    r_C3_50soc = np.hstack(r_C3_50soc)
    r_C3_50soc = r_C3_50soc.reshape(-1,1)
    r_C3_70soc = np.hstack(r_C3_70soc)
    r_C3_70soc = r_C3_70soc.reshape(-1,1)
    
    r_C4_30soc = np.hstack(r_C4_30soc)
    r_C4_30soc = r_C4_30soc.reshape(-1,1)
    r_C4_50soc = np.hstack(r_C4_50soc)
    r_C4_50soc = r_C4_50soc.reshape(-1,1)
    r_C4_70soc = np.hstack(r_C4_70soc)
    r_C4_70soc = r_C4_70soc.reshape(-1,1)
    
    r_C5_30soc = np.hstack(r_C5_30soc)
    r_C5_30soc = r_C5_30soc.reshape(-1,1)
    r_C5_50soc = np.hstack(r_C5_50soc)
    r_C5_50soc = r_C5_50soc.reshape(-1,1)
    r_C5_70soc = np.hstack(r_C5_70soc)
    r_C5_70soc = r_C5_70soc.reshape(-1,1)
    
    soh = np.hstack(soh)
    soh = soh.reshape(-1,1)
    
    pulseV_C3_30soc = np.vstack(pulseV_C3_30soc)
    pulseV_C3_50soc = np.vstack(pulseV_C3_50soc)
    pulseV_C3_70soc = np.vstack(pulseV_C3_70soc)
    
    pulseV_C4_30soc = np.vstack(pulseV_C4_30soc)
    pulseV_C4_50soc = np.vstack(pulseV_C4_50soc)
    pulseV_C4_70soc = np.vstack(pulseV_C4_70soc)
    
    pulseV_C5_30soc = np.vstack(pulseV_C5_30soc)
    pulseV_C5_50soc = np.vstack(pulseV_C5_50soc)
    pulseV_C5_70soc = np.vstack(pulseV_C5_70soc)
    
    
    cell_names = np.hstack(cell_names)
    cell_names = cell_names.reshape(-1,1)
    
    dataset_['cell_ID'] = cell_ID
    dataset_['cell_numbers'] = cell_numbers
    dataset_['cell_names'] = cell_names
    
    dataset_['pulseV_C3_30soc'] = pulseV_C3_30soc
    dataset_['pulseV_C3_50soc'] = pulseV_C3_50soc
    dataset_['pulseV_C3_70soc'] = pulseV_C3_70soc
    
    dataset_['pulseV_C4_30soc'] = pulseV_C4_30soc
    dataset_['pulseV_C4_50soc'] = pulseV_C4_50soc
    dataset_['pulseV_C4_70soc'] = pulseV_C4_70soc
    
    dataset_['pulseV_C5_30soc'] = pulseV_C5_30soc
    dataset_['pulseV_C5_50soc'] = pulseV_C5_50soc
    dataset_['pulseV_C5_70soc'] = pulseV_C5_70soc
    
    dataset_['r_C3_30soc'] = r_C3_30soc
    dataset_['r_C3_50soc'] = r_C3_50soc
    dataset_['r_C3_70soc'] = r_C3_70soc
    
    dataset_['r_C4_30soc'] = r_C4_30soc
    dataset_['r_C4_50soc'] = r_C4_50soc
    dataset_['r_C4_70soc'] = r_C4_70soc
    
    dataset_['r_C5_30soc'] = r_C5_30soc
    dataset_['r_C5_50soc'] = r_C5_50soc
    dataset_['r_C5_70soc'] = r_C5_70soc
    
    dataset_['soh'] = soh
    
    return dataset_
    

inputs = ['pulseV_C3_30soc', 'pulseV_C3_50soc', 'pulseV_C3_70soc', 'pulseV_C4_30soc', 'pulseV_C4_50soc', 'pulseV_C4_70soc', 'pulseV_C5_30soc', 'pulseV_C5_50soc', 'pulseV_C5_70soc']
outputs = ['soh', 'r_C3_30soc', 'r_C3_50soc', 'r_C3_70soc', 'r_C4_30soc', 'r_C4_50soc', 'r_C4_70soc','r_C5_30soc', 'r_C5_50soc', 'r_C5_70soc']

# Function to prepare the dataset for modeling
def create_pulse_dataset(inputs = [], outputs = [], soh_limit = 0.8):
    dataset_ = get_pulse_data(soh_limit)
    dataset = {}
    dataset['stacked_inputs'] = np.vstack([dataset_[i] for i in inputs])
    
    dataset_['outputs'] = np.hstack([dataset_[i] for i in outputs])
    
    n_inputs = len(inputs)
    
    dataset['stacked_outputs'] = np.vstack([dataset_['outputs']] * n_inputs)
    dataset['stacked_cell_ID'] = np.vstack([dataset_['cell_ID']] * n_inputs)
    dataset['stacked_cell_numbers'] = np.vstack([dataset_['cell_numbers']] * n_inputs)
    
    dataset['stacked_relative_inputs'] = []
    for i in dataset['stacked_inputs']:
        dataset['stacked_relative_inputs'].append(i - i[0])
    dataset['stacked_relative_inputs'] = np.vstack(dataset['stacked_relative_inputs'])
    
    return dataset