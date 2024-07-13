'''
This script performs warning analysis on the selected near-crashes in 100Car NDS data to 
determine conflicts and generate warning analysis results. The analysis is performed using
different surrogate metrics of conflicts: PSD, DRAC, TTC, and the unified metric proposed in our study.
The results are saved in CSV files for further analysis.
'''

import sys
import time as systime
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
sys.path.append('./')
from Demonstration.demonstration_utils import *


# Set input and output paths
path_input = './Data/InputData/'
path_output = './Data/OutputData/'


# Load data
events = pd.read_hdf(path_input+'100Car/HundredCar_NearCrashes.h5', key='data')
meta = pd.read_csv(path_input+'100Car/HundredCar_metadata_NearCrashes.csv')
proximity_phi = pd.read_csv(path_output+'conflict_probability/proximity_phi.csv')

# PSD
demonstration = []
used_time = []
for psd_threshold in tqdm(np.round(np.arange(0.04,4.,0.04),2), desc='PSD'):
    current_time = systime.time()
    meta = warning(events, meta, [psd_threshold], 'PSD')
    used_time.append(systime.time()-current_time)
    meta['threshold'] = psd_threshold
    demonstration.append(meta)
demonstration = pd.concat(demonstration).reset_index(drop=True)
demonstration.to_csv(path_output+'conflict_probability/PSD_warning.csv', index=False)
print(f'PSD: {np.mean(used_time):.4f} seconds per threshold') # 0.6942s

# DRAC
demonstration = []
used_time = []
for drac_threshold in tqdm(np.round(np.arange(0.05,5.,0.05),2), desc='DRAC'):
    current_time = systime.time()
    meta = warning(events, meta, [drac_threshold], 'DRAC')
    used_time.append(systime.time()-current_time)
    meta['threshold'] = drac_threshold
    demonstration.append(meta)
demonstration = pd.concat(demonstration).reset_index(drop=True)
demonstration.to_csv(path_output+'conflict_probability/DRAC_warning.csv', index=False)
print(f'DRAC: {np.mean(used_time):.4f} seconds per threshold') # 0.9196s

# TTC
demonstration = []
used_time = []
for ttc_threshold in tqdm(np.round(np.arange(0.2,20.,0.2),1), desc='TTC'):
    current_time = systime.time()
    meta = warning(events, meta, [ttc_threshold], 'TTC')
    used_time.append(systime.time()-current_time)
    meta['threshold'] = ttc_threshold
    demonstration.append(meta)
demonstration = pd.concat(demonstration).reset_index(drop=True)
demonstration.to_csv(path_output+'conflict_probability/TTC_warning.csv', index=False)
print(f'TTC: {np.mean(used_time):.4f} seconds per threshold') # 0.9706s

# Unified
demonstration = []
used_time = []
n_list = np.arange(2,101)
for n in tqdm(n_list, desc='Unified'):
    current_time = systime.time()
    meta = warning(events, meta, [n, proximity_phi], 'Unified')
    used_time.append(systime.time()-current_time)
    meta['threshold'] = n
    demonstration.append(meta)
demonstration = pd.concat(demonstration).reset_index(drop=True)
demonstration.to_csv(path_output+'conflict_probability/Unified_warning.csv', index=False)
print(f'Unified: {np.mean(used_time):.4f} seconds per threshold') # 0.5434s
