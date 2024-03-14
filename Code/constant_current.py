#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 15:47:50 2024

@author: jay
"""

import numpy as np
import pandas as pd

#%% Fetch the capacity and resistances data

def get_cell_state(soh_limit = 0.8):
    pulse_resistances_dic = pd.read_pickle(r'../Rawdata/pulse_resistances.pkl')
        
    # Aging data is present in the pulse resistance dic. Fetch the aging data and calc RPT at SOH limit
    aging_dic = {}
    end_rpt = []
    cell_names = list(pulse_resistances_dic.keys())
    for cell_name in cell_names:
        aging_dic[cell_name] = pulse_resistances_dic[cell_name]['C4']['30SOC']['dchg_cap']
        
        end_rpt.append([i for i in range(len(aging_dic[cell_name])) if aging_dic[cell_name][i] >= soh_limit*aging_dic[cell_name][0]][-1])
        
    return pulse_resistances_dic, aging_dic, end_rpt


#%% Fetch the voltage profiles

def get_voltage_profiles():
    charge_voltage_profiles_dic = pd.read_pickle('../Rawdata/' + 'charge_voltage_profiles.pkl')

    #Correcting the keys (cell names) to remove spaces and proper capitalization
    charge_voltage_profiles_dic = { k.replace(' ', ''): v for k, v in charge_voltage_profiles_dic.items() }
    charge_voltage_profiles_dic = { k.replace('s', 'S'): v for k, v in charge_voltage_profiles_dic.items() }
    
    cell_names = list(charge_voltage_profiles_dic.keys())

    # Truncating all the rpt size to 3200 (smallest rpt size in epSanyo025)
    #This code only cuts off some of the CV portion and will not affect the subvectors    
    for cell_name in cell_names:
        for rpt in np.arange(0, len(charge_voltage_profiles_dic[cell_name]['charge'])):
            # if len(charge_voltage_profiles_dic[cell_name]['charge'][rpt]) > 7000:
            #     print(cell_name, rpt)
            charge_voltage_profiles_dic[cell_name]['charge'][rpt] = charge_voltage_profiles_dic[cell_name]['charge'][rpt][:3200]
   
    return charge_voltage_profiles_dic, cell_names


# # Plotting the voltage profile of all the rpts for each cell
# for cell_name in cell_names:
#     for item in charge_voltage_profiles_dic[cell_name]['charge']:
#       plt.plot(item)
#     plt.show()
#     plt.clf()

#%% Calculate OCV
# ocv_soc = pd.read_excel('ocv.xlsx')

# Calculate OCV by calculating average of the first discharge of all the cells


def getOCV():
    #Calculate OCV and SOC of the NMC cells 
    #Using the first RPT of all the cells and averaging them     
    
    charge_voltage_profiles_dic = get_voltage_profiles()[0]
    cell_names = get_voltage_profiles()[1]
    
    ocv_soc_df = pd.DataFrame()
    for cell_name in cell_names:
        voltage0 = pd.Series(charge_voltage_profiles_dic[cell_name]['charge'][0], name= cell_name)
        ocv_soc_df = pd.concat((ocv_soc_df,voltage0.rename(cell_name)), ignore_index=True, axis = 1)
    
    
    ocv_mean = ocv_soc_df.mean(axis = 1).values
    ocv_sd = ocv_soc_df.std(axis = 1).values
    
    # #Plotting the OCV
    # plt.plot(np.arange(0,3200), ocv_mean, 'k-')
    # plt.fill_between(np.arange(0,3200), ocv_mean-ocv_sd, ocv_mean+ocv_sd, alpha=0.2, edgecolor='#1B2ACC', facecolor='#089FFF',
    #     linewidth=4, antialiased=True)
    # plt.show()
    # plt.clf()
    
    ocv_soc_df['ocv_mean'] = ocv_mean.tolist()
    ocv_soc_df['ocv_sd'] = ocv_sd.tolist()
    #Min max scaling the voltage to get SOC
    ocv_soc_df['soc'] = ( ocv_soc_df['ocv_mean'] - ocv_soc_df['ocv_mean'].min() )/ ( ocv_soc_df['ocv_mean'].max() - ocv_soc_df['ocv_mean'].min() )
    
    return ocv_soc_df


#%% Functions to clip voltage data and create sub vectors

#Clips the voltage data to SOCs
def extract_cc_v_curves_raw(soc1, soc2, ocv_soc, v_data, end_rpt_80):
    
    ocv_soc_df = getOCV()
    
    # Extract the voltages to select the data
    v1 = ocv_soc_df['ocv_mean'][np.where(ocv_soc_df['soc'] >= soc1)[0][0]]
    v2 = ocv_soc_df['ocv_mean'][np.where(ocv_soc_df['soc'] >= soc2)[0][0]]
    
    cc_qv_curves = []
    # group_id = []
    cell_names = list(v_data.keys())
    for i in range(0,len(v_data)):
        for j in range(0, int(end_rpt_80[i])):
            # group_id.append(og_data['groups'][i])
            v_chg_vec = v_data[cell_names[i]]['charge'][j]
            
            idxs = (v_chg_vec >= v1) * (v_chg_vec <= v2)
            v_chg_vec_clip = v_chg_vec[idxs]
            cc_qv_curves.append(v_chg_vec_clip.tolist())
    # cc_qv_curves = np.vstack(cc_qv_curves)
    # group_id = np.vstack(group_id)
    
    return cc_qv_curves

# Use this to transform the voltage vector into a dataset of N subvectors 
def create_subvecs(cc_qv_dataset, lookback, stride):
    """Transform a time series into a dataset of many equal length samples
    """
    X = []
    if len(cc_qv_dataset) < lookback:
        print('Lookback is longer than V profile. Current V profile length : ' , len(cc_qv_dataset))
        
    for i in range(len(cc_qv_dataset)-lookback):
        feature = cc_qv_dataset[i:i+lookback]
        X.append(feature)
    X = np.vstack(X)
    return X[0::stride,:]


# Extract the other datset
def generate_target_values(pulse_resistances_dic, end_rpt_80):
    
    target_dataset = {'cell_id' : [],
                      
                      'soh' : [], 
                      
                      'r_C3_30soc' : [],
                      'r_C3_50soc' : [],
                      'r_C3_70soc' : [],
                    
                      'r_C4_30soc' : [],
                      'r_C4_50soc' : [],
                      'r_C4_70soc' : [],
                    
                      'r_C5_30soc' : [],
                      'r_C5_50soc' : [],
                      'r_C5_70soc' : [],
                      }
     
    for i,j in enumerate(pulse_resistances_dic):
        for k in range(end_rpt_80[i]):
            
            target_dataset['cell_id'].append(i) #cell_ids are integers from 0 to 47
            # target_dataset['cell_id'].append(j) # cell_ids are strings such as 'epSanyo002'
            
            target_dataset['soh'].append(pulse_resistances_dic[j]['C3']['30SOC']['dchg_cap'][k])
            target_dataset['r_C3_30soc'].append(pulse_resistances_dic[j]['C3']['30SOC']['resistance'][k])
            target_dataset['r_C3_50soc'].append(pulse_resistances_dic[j]['C3']['50SOC']['resistance'][k])
            target_dataset['r_C3_70soc'].append(pulse_resistances_dic[j]['C3']['70SOC']['resistance'][k])
            target_dataset['r_C4_30soc'].append(pulse_resistances_dic[j]['C4']['30SOC']['resistance'][k])
            target_dataset['r_C4_50soc'].append(pulse_resistances_dic[j]['C4']['50SOC']['resistance'][k])
            target_dataset['r_C4_70soc'].append(pulse_resistances_dic[j]['C4']['70SOC']['resistance'][k])
            target_dataset['r_C5_30soc'].append(pulse_resistances_dic[j]['C5']['30SOC']['resistance'][k])
            target_dataset['r_C5_50soc'].append(pulse_resistances_dic[j]['C5']['50SOC']['resistance'][k])
            target_dataset['r_C5_70soc'].append(pulse_resistances_dic[j]['C5']['70SOC']['resistance'][k])
   
    return target_dataset
    



# # Set the parameters for the model.
# lower_soc, minutes = np.mgrid[20:61:10, 3:15.5:2]
# x_mesh = np.vstack((lower_soc.flatten(), minutes.flatten())).T
def get_cc_data(soc1, soc2, lookback, stride, ocv_soc, v_data, end_rpt, target_dataset):
    
    ocv_soc_df = getOCV()
    
    cc_qv_curves = extract_cc_v_curves_raw(soc1, soc2, ocv_soc_df, v_data, end_rpt)
    # dataset['cc_qv_curves_padded'] = pad_sequences(cc_qv_curves, padding='post', dtype=float, maxlen=3000) # Pad the sequences with zeros at the end.
    
    
    # Take the extracted CC curves and further extract sub-vecs of equal length
    # lookback = 7 * 60 # 5 mins of 1 Hz voltage data
    # stride = int(lookback / 2)
    cc_subvecs = []
    cell_ids = []
    
    soh = []
    r_C3_30soc = []
    r_C3_50soc = []
    r_C3_70soc = []
    r_C4_30soc = []
    r_C4_50soc = []
    r_C4_70soc = []
    r_C5_30soc = []
    r_C5_50soc = []
    r_C5_70soc = []
    
    for i in range(0,len(cc_qv_curves)):
        v_chg_subvecs = create_subvecs(cc_qv_curves[i], lookback, stride) #subvecs of each cell
        cc_subvecs.append(v_chg_subvecs) #subvecs of all cells
        cell_ids.append(np.ones(len(v_chg_subvecs)) * target_dataset['cell_id'][i])
        
        soh.append(np.ones(len(v_chg_subvecs)) * target_dataset['soh'][i])
        r_C3_30soc.append(np.ones(len(v_chg_subvecs)) * target_dataset['r_C3_30soc'][i])
        r_C3_50soc.append(np.ones(len(v_chg_subvecs)) * target_dataset['r_C3_50soc'][i])
        r_C3_70soc.append(np.ones(len(v_chg_subvecs)) * target_dataset['r_C3_70soc'][i])
        r_C4_30soc.append(np.ones(len(v_chg_subvecs)) * target_dataset['r_C4_30soc'][i])
        r_C4_50soc.append(np.ones(len(v_chg_subvecs)) * target_dataset['r_C4_50soc'][i])
        r_C4_70soc.append(np.ones(len(v_chg_subvecs)) * target_dataset['r_C4_70soc'][i])
        r_C5_30soc.append(np.ones(len(v_chg_subvecs)) * target_dataset['r_C5_30soc'][i])
        r_C5_50soc.append(np.ones(len(v_chg_subvecs)) * target_dataset['r_C5_50soc'][i])
        r_C5_70soc.append(np.ones(len(v_chg_subvecs)) * target_dataset['r_C5_70soc'][i])
    
    cc_subvecs = np.vstack(cc_subvecs)
    cell_ids = np.hstack(cell_ids).reshape(-1,1)
    soh = np.hstack(soh).reshape(-1,1)
    r_C3_30soc = np.hstack(r_C3_30soc).reshape(-1,1)
    r_C3_50soc = np.hstack(r_C3_50soc).reshape(-1,1)
    r_C3_70soc = np.hstack(r_C3_70soc).reshape(-1,1)
    r_C4_30soc = np.hstack(r_C4_30soc).reshape(-1,1)
    r_C4_50soc = np.hstack(r_C4_50soc).reshape(-1,1)
    r_C4_70soc = np.hstack(r_C4_70soc).reshape(-1,1)
    r_C5_30soc = np.hstack(r_C5_30soc).reshape(-1,1)
    r_C5_50soc = np.hstack(r_C5_50soc).reshape(-1,1)
    r_C5_70soc = np.hstack(r_C5_70soc).reshape(-1,1)    
    
    # Form dataset and downsample
    dwn_samp = 1 # 5x
    dataset2 = {
        'cc_subvecs':cc_subvecs[0::dwn_samp,:],
        'cell_ids':cell_ids[0::dwn_samp,:],
        'soh':soh[0::dwn_samp,:],
        'r_C3_30soc':r_C3_30soc[0::dwn_samp,:],
        'r_C3_50soc':r_C3_50soc[0::dwn_samp,:],
        'r_C3_70soc':r_C3_70soc[0::dwn_samp,:],
        'r_C4_30soc':r_C4_30soc[0::dwn_samp,:],
        'r_C4_50soc':r_C4_50soc[0::dwn_samp,:],
        'r_C4_70soc':r_C4_70soc[0::dwn_samp,:],
        'r_C5_30soc':r_C5_30soc[0::dwn_samp,:],
        'r_C5_50soc':r_C5_50soc[0::dwn_samp,:],
        'r_C5_70soc':r_C5_70soc[0::dwn_samp,:],
        }
    
    return dataset2

#%% Create partial curves dataset

def create_cc_dataset(soc1 = 0.2, soc2 = 0.7, lookback = 600,  soh_limit = 0.8, outputs = []):
    
    stride = int(0.3*lookback)
    
    ocv_soc_df = getOCV()
    pulse_resistances_dic = get_cell_state(soh_limit = soh_limit)[0]
    end_rpt = get_cell_state(soh_limit = soh_limit)[2]

    target_dataset = generate_target_values(pulse_resistances_dic, end_rpt)

    # Path to the new voltage data files
    v_data =  get_voltage_profiles()[0]

    dataset_ = get_cc_data(soc1, soc2, lookback, stride, ocv_soc_df, v_data, end_rpt, target_dataset)
    
    dataset = {}
    dataset['stacked_cell_numbers'] = dataset_['cell_ids']
    dataset['stacked_inputs'] = dataset_['cc_subvecs']
    dataset['stacked_outputs'] = np.hstack([dataset_[i] for i in outputs])

    dataset['stacked_relative_inputs'] = []
    for i in dataset['stacked_inputs']:
        dataset['stacked_relative_inputs'].append(i - i[0])
    dataset['stacked_relative_inputs'] = np.vstack(dataset['stacked_relative_inputs'])
    
    return dataset