'''
This script is used for collision warnings under the optimal thresholds found previously.
'''

import sys
import pandas as pd
sys.path.append('./')
from Demonstration.demonstration_utils import *

# Set the input and output paths
path_input = './Data/InputData/'
path_output = './Data/OutputData/'

# With optimal threshold
## TTC*=4.0, DRAC*=0.45, PSD*=0.64, Unified*=13
thresholds = [4.0, 0.45, 0.64, 13]

## Load data
events = pd.read_hdf(path_input+'100Car/HundredCar_NearCrashes.h5', key='data')
meta = pd.read_csv(path_input+'100Car/HundredCar_metadata_NearCrashes.csv').rename(columns={'webfileid':'trip_id'})
proximity_phi = pd.read_csv(path_output+'conflict_probability/proximity_phi.csv')

## Generate warning results
for threshold, conflict_indicator in zip(thresholds, ['TTC', 'DRAC', 'PSD', 'Unified']):
    print(f'Processing with threshold = {threshold} and conflict indicator = {conflict_indicator} ...')
    if conflict_indicator=='Unified':
        meta, data = warning(events, meta, [threshold, proximity_phi], conflict_indicator, record_data=True)
    else:
        meta, data = warning(events, meta, [threshold], conflict_indicator, record_data=True)
    meta['threshold'] = threshold
    meta['timeliness'] = meta['moment'] - meta['first warning']
    meta.to_csv(path_output+'conflict_probability/optimal_warning/'+conflict_indicator+'_NearCrashes.csv', index=False)
    if conflict_indicator!='Unified':
        data = data.rename(columns={'indicator_value': conflict_indicator})
        events = events.merge(data[['trip_id','time',conflict_indicator]], on=['trip_id', 'time'], how='left')
    else:
        events = events.merge(data[['trip_id','time','probability']], on=['trip_id', 'time'], how='left')
events.to_hdf(path_output+'conflict_probability/optimal_warning/NearCrashes.h5', key='data', mode='w')
print(events.columns)
