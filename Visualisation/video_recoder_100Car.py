'''
This script produce images for the dynamic visualisation of near-crashes that
have at least one metric among DRAC, TTC, and Unified fail to indicate a conflict.
'''

import os
import sys
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
font = {'family' : 'Arial',
        'size'   : 9}
plt.rc('font', **font)
plt.rcParams['mathtext.fontset'] = 'stix' # dejavuserif
sys.path.append('./')
from DataProcessing.utils.coortrans import coortrans
coortrans = coortrans()
from visual_utils import *
from GaussianProcessRegression.training_utils import *

# Set input and output paths
path_raw = './Data/RawData/'
path_processed = './Data/ProcessedData/'
path_input = './Data/InputData/'
path_output = './Data/OutputData/'

# Check if CUDA is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')
if device=='cpu':
    num_threads = torch.get_num_threads()
    print(f'Number of available threads: {num_threads}')
    torch.set_num_threads(round(num_threads/2))


# Load trained model
beta = 5
num_inducing_points = 100
model_idx = 52
pipeline = train_val_test(device, num_inducing_points, path_input, path_output)
pipeline.model.load_state_dict(torch.load(path_output+f'trained_models/highD_LC/beta={beta}/model_{model_idx}epoch.pth', map_location=torch.device(device)))
pipeline.likelihood.load_state_dict(torch.load(path_output+f'trained_models/highD_LC/beta={beta}/likelihood_{model_idx}epoch.pth', map_location=torch.device(device)))
pipeline.model.eval()
pipeline.likelihood.eval()
model = pipeline.model.to(device)
likelihood = pipeline.likelihood.to(device)


# Read data
events = pd.read_hdf(path_output + 'conflict_probability/optimal_warning/NearCrashes.h5', key='data')
meta = pd.read_csv(path_output + 'conflict_probability/optimal_warning/Unified_NearCrashes.csv').set_index('trip_id')
proximity_phi = pd.read_csv(path_output+'conflict_probability/proximity_phi.csv')

# Mirror event coordinates as the model is trained on highD where the y-axis points downwards
events = events.rename(columns={'x_i':'y_i', 'y_i':'x_i', 'x_j':'y_j', 'y_j':'x_j',
                                'vx_i':'vy_i', 'vy_i':'vx_i', 'vx_j':'vy_j', 'vy_j':'vx_j',
                                'hx_i':'hy_i', 'hy_i':'hx_i', 'hx_j':'hy_j', 'hy_j':'hx_j'})
events['psi_i'] = coortrans.angle(1, 0, events['hx_i'], events['hy_i'])
events['psi_j'] = coortrans.angle(1, 0, events['hx_j'], events['hy_j'])

folder_list = ['unified1_ttc1_drac0', 
               'unified1_ttc0_drac1',
               'unified1_ttc0_drac0',
               'unified0_ttc1_drac1',
               'unified0_ttc0_drac1',
               'unified_false_warning']
trip_list = [[8622,9101], 
             [8702], 
             [8463,8810,8854],
             [8793],
             [8332],
             [8395,8460,8463,8565,9044]]
n = 13 # optimal threshold for the unified metric


for idx, folder in enumerate(folder_list):
    trip_id_list = trip_list[idx]
    for trip_id in trip_id_list:
        save_dir = path_output + f'video_images/100Car/{folder}/{trip_id}/'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        df = events[events['trip_id']==trip_id].sort_values('time').reset_index(drop=True)
        df = df.merge(proximity_phi, on=['trip_id','time'])
        df['s_centroid'] = np.sqrt((df['x_i'] - df['x_j']) ** 2 + (df['y_i'] - df['y_j']) ** 2)
        df['probability'] = extreme_cdf(df['s_centroid'].values, df['mu'].values, df['sigma'].values, n)
        moment = meta.loc[trip_id]['moment']
        df['time'] = np.round(df['time'] - moment, 2)
        conflict_start = df[df['event']]['time'].min()
        conflict_end = df[df['event']]['time'].max()

        df['delta_v'] = np.sqrt((df['vx_i']-df['vx_j'])**2 + (df['vy_i']-df['vy_j'])**2)
        df['delta_v2'] = df['delta_v']**2
        df['speed_i2'] = df['speed_i']**2
        df['speed_j2'] = df['speed_j']**2
        features = ['length_i','length_j','hx_j','hy_j','delta_v','delta_v2','speed_i2','speed_j2','acc_i','rho']
        df_view_i = coortrans.transform_coor(df, view='i')
        heading_j = df_view_i[['time','hx_j','hy_j']]
        df_view_i = df_view_i.set_index('time')
        df_relative = coortrans.transform_coor(df, view='relative')
        rho = coortrans.angle(1, 0, df_relative['x_j'], df_relative['y_j']).reset_index().rename(columns={0:'rho'})
        rho['time'] = df_relative['time']
        df_relative = df_relative.set_index('time')
        interaction_situation = df.drop(columns=['hx_j','hy_j']).merge(heading_j, on='time').merge(rho, on='time')
        interaction_situation = interaction_situation[features+['time']].set_index('time')

        for t in tqdm(df['time'], desc=f'Trip {trip_id}'):
            fig = visual_100Car(t, df, df_view_i, df_relative, interaction_situation, 
                               model, likelihood, device, n, conflict_start, conflict_end)
            fig.savefig(save_dir+f'frame_{int(t*100)}.png', bbox_inches='tight', dpi=600)
            plt.close(fig)
