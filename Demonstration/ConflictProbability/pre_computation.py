'''
This script computes the mean and standard deviation of the conditional proximity distribution
for matched near-crashes in the 100-Car dataset. This is used to compute the conflict probability in the next step.
'''

import sys
import numpy as np
import pandas as pd
sys.path.append('./')
from DataProcessing.utils.coortrans import coortrans
coortrans = coortrans()
from GaussianProcessRegression.training_utils import *

# Check if CUDA is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')

if device=='cpu':
    num_threads = torch.get_num_threads()
    print(f'Number of available threads: {num_threads}')
    torch.set_num_threads(round(num_threads/2))

# Set input and output paths
path_input = './Data/InputData/'
path_output = './Data/OutputData/'

# Compute mu_list, sigma_list
print('Computing mu_list, sigma_list ...')

## Load data
events = pd.read_hdf(path_input+'100Car/HundredCar_NearCrashes.h5', key='data').reset_index(drop=True)

## Transform coordinates and formulate input data
events['delta_v'] = np.sqrt((events['vx_i']-events['vx_j'])**2 + (events['vy_i']-events['vy_j'])**2)
events['delta_v2'] = events['delta_v']**2
events['speed_i2'] = events['speed_i']**2
events['speed_j2'] = events['speed_j']**2
features = ['length_i','length_j','hx_j','hy_j','delta_v','delta_v2','speed_i2','speed_j2','acc_i','rho']
events_view_i = coortrans.transform_coor(events, view='i')
heading_j = events_view_i[['trip_id','time','hx_j','hy_j']]
events_relative = coortrans.transform_coor(events, view='relative')
rho = coortrans.angle(1, 0, events_relative['x_j'], events_relative['y_j']).reset_index().rename(columns={0:'rho'})
rho[['trip_id','time']] = events_relative[['trip_id','time']]
interaction_situation = events.drop(columns=['hx_j','hy_j']).merge(heading_j, on=['trip_id','time']).merge(rho, on=['trip_id','time'])
interaction_situation = interaction_situation[features+['trip_id','time']]

## Load trained model
beta = 5
model_idx = 52
num_inducing_points = 100
pipeline = train_val_test(device, num_inducing_points)
pipeline.model.load_state_dict(torch.load(path_output+f'trained_models/highD_LC/beta={beta}/model_{model_idx}epoch.pth', map_location=torch.device(device)))
pipeline.likelihood.load_state_dict(torch.load(path_output+f'trained_models/highD_LC/beta={beta}/likelihood_{model_idx}epoch.pth', map_location=torch.device(device)))
pipeline.model.eval()
pipeline.likelihood.eval()
model = pipeline.model.to(device)
likelihood = pipeline.likelihood.to(device)

## Compute mu_list, sigma_list
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    f_dist = model(torch.Tensor(interaction_situation[features].values).to(device))
    y_dist = likelihood(f_dist)
    mu_list, sigma_list = y_dist.mean.cpu().numpy(), y_dist.variance.sqrt().cpu().numpy()

proximity_phi = pd.DataFrame({'trip_id': interaction_situation['trip_id'].values,
                              'time': interaction_situation['time'].values,
                              'mu': mu_list,
                              'sigma': sigma_list})

proximity_phi.to_csv(path_output+'conflict_probability/proximity_phi.csv', index=False)
