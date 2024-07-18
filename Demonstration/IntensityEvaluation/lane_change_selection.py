'''
This script is used to evaluate conflict intensity for lane-changing interactions in the highD dataset.
Two metrics, including the unified metric we proposed in the study and TTC, are used.
Lane-changing interactions are selected based on laneId, and the period of lane-change is identified by the Ricker wavelet.
'''

import sys
import glob
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
sys.path.append('./')
import DataProcessing.utils.TwoDimTTC as TwoDimTTC
from DataProcessing.utils.coortrans import coortrans
coortrans = coortrans()
from GaussianProcessRegression.training_utils import *
from Demonstration.demonstration_utils import *

# Check if CUDA is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')

if device=='cpu':
    num_threads = torch.get_num_threads()
    print(f'Number of available threads: {num_threads}')
    torch.set_num_threads(round(num_threads/2))

# Set input and output paths
path_raw = './Data/RawData/'
path_processed = './Data/ProcessedData/'
path_input = './Data/InputData/'
path_output = './Data/OutputData/'


## Load the trained model
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


## Set thresholds based on the previous experiments
ttc_threshold = 4.2
nhat_threshold = 17

## Compute on the highD dataset
metadatafiles =  sorted(glob.glob(path_raw + 'highD/RecordingMetadata/*.csv'))
metadata = []
for metadatafile in metadatafiles:
    df = pd.read_csv(metadatafile)
    metadata.append(df)
metadata = pd.concat(metadata).set_index('id')

data2save = []
interaction_cases = []
conflict_cases = []
lc_id = 0
conflict_count = 0

for loc_id in range(6, 0, -1):
    data_lc = pd.read_hdf(path_processed+'highD/highD_'+str(loc_id).zfill(2)+'.h5', key='data')
    data_lc = data_lc.sort_values(['track_id','frame_id']).set_index('track_id')
    lane_change = data_lc.groupby('track_id')['laneId'].nunique() > 1
    lc_veh_ids = lane_change.index[lane_change].values
    meta_tracks = []
    for fileid in metadata[metadata['locationId']==loc_id].index:
        meta_track = pd.read_csv(path_raw+'highD/01_tracksMeta.csv')
        meta_track['id'] = (fileid*10000 + meta_track['id']).astype(int)
        meta_tracks.append(meta_track)
    meta_tracks = pd.concat(meta_tracks)

    progress_bar = tqdm(range(len(lc_veh_ids)), desc=f'Location {loc_id}')
    for i in progress_bar:
        lc_veh_id = lc_veh_ids[i]
        veh = data_lc.loc[lc_veh_id]

        # Get the lane markings
        lane_markings = [float(y) for lane in ['lowerLaneMarkings','upperLaneMarkings'] for y in metadata.loc[lc_veh_id//10000][lane].split(';')]
        lane_markings = np.sort(lane_markings)

        # There may be multiple lane changes in one trajectory
        lane_change_points = veh[abs(veh['laneId'].diff())>0]['frame_id'].values
        frame_segments = np.concatenate(([veh['frame_id'].min()], 
                                         (lane_change_points[1:] + lane_change_points[:-1])/2, 
                                         [veh['frame_id'].max()]))
        for frame_start, frame_end in zip(frame_segments[:-1], frame_segments[1:]):
            veh_i = data_lc.loc[lc_veh_id]
            veh_i = veh_i[veh_i['frame_id'].between(frame_start, frame_end)]

            preceding_vehs = veh_i[veh_i['precedingId']>0]['precedingId'].unique()
            following_vehs = veh_i[veh_i['followingId']>0]['followingId'].unique()
            if len(preceding_vehs)==0 or len(following_vehs)==0:
                # Skip as the vehicle changed lane without interacting with other vehicles
                continue
            int_veh_list = np.concatenate([preceding_vehs, following_vehs])
            interaction_cases.append([loc_id, lc_id, lc_veh_id, frame_start, frame_end, ', '.join(int_veh_list.astype(str).tolist())])
            lc_id += 1

            # Note: this combination may involve repeated pairs of vehicles, which we will filter in the next step
            for interact_veh_id in int_veh_list:
                veh_j = data_lc.loc[interact_veh_id].drop(columns=['precedingId', 'followingId'])
                df = veh_i.drop(columns=['precedingId', 'followingId']).merge(veh_j, on='frame_id', suffixes=('_i', '_j'), how='inner')
                df['acc_i'] = df['ax_i']*df['hx_i'] + df['ay_i']*df['hy_i']
                df['time'] = np.round((df['frame_id'] - df['frame_id'].min())/10, 1)
                ttc = TwoDimTTC.TTC(df, 'values')
                df['TTC'] = ttc
                df['s_centroid'] = np.sqrt((df['x_i']-df['x_j'])**2 + (df['y_i']-df['y_j'])**2)
                df['delta_v2'] = (df['vx_i']-df['vx_j'])**2 + (df['vy_i']-df['vy_j'])**2
                df['delta_v'] = np.sqrt(df['delta_v2'])
                df['speed_i2'] = df['speed_i']**2
                df['speed_j2'] = df['speed_j']**2

                df_view_i = coortrans.transform_coor(df, view='i')
                heading_j = df_view_i[['frame_id','hx_j','hy_j']]
                df_relative = coortrans.transform_coor(df, view='relative')
                rho = coortrans.angle(1, 0, df_relative['x_j'], df_relative['y_j']).reset_index().rename(columns={0:'rho'})
                rho['frame_id'] = df_relative['frame_id']
                interaction_situation = df.drop(columns=['hx_j','hy_j']).merge(heading_j, on='frame_id').merge(rho, on='frame_id')
                features = ['length_i','length_j','hx_j','hy_j','delta_v','delta_v2','speed_i2','speed_j2','acc_i','rho']
                interaction_situation = interaction_situation[features+['frame_id']].set_index('frame_id')

                with torch.no_grad(), gpytorch.settings.fast_pred_var():
                    f_dist = model(torch.Tensor(interaction_situation.values).to(device))
                    y_dist = likelihood(f_dist)
                    mu_list, sigma_list = y_dist.mean.cpu().numpy(), y_dist.variance.sqrt().cpu().numpy()
                df['mu'] = mu_list
                df['sigma'] = sigma_list
                df['n_hat'] = np.log(0.5)/np.log(1-lognormal_cdf(df['s_centroid'], df['mu'], df['sigma']))
                # here there may be a warning for invalid logrithm of zero; 
                # this happens when the lognormal_cdf is close to 1, which means the intensity is very low
                # the computated n_hat is 0 and still valid
                df.loc[df['n_hat']<1, 'n_hat'] = 1

                # locate the start and end of the lane change
                frame_start, frame_end = locate_lane_change(veh_i['frame_id'].values, veh_i['y'].values, lane_markings, veh_i['width'].mean())
                
                ttc = df[(df['frame_id']>=frame_start)&(df['frame_id']<=frame_end)]['TTC']
                avg_ttc = ttc[ttc<ttc_threshold].mean()
                # check if there is a continuous period where TTC is below the threshold
                continuous_true = ''.join((ttc<ttc_threshold).values.astype(int).astype(str)).split('0')
                max_length_ttc = len(max(continuous_true, key=len))
                ttc = max_length_ttc >= 10
                if not ttc:
                    avg_ttc = np.nan
                
                unified = df[(df['frame_id']>=frame_start)&(df['frame_id']<=frame_end)]['n_hat']
                avg_unified = np.log10(unified[unified>nhat_threshold]).mean()
                # check if there is a continuous period where the unified metric is above the threshold
                continuous_true = ''.join((unified>nhat_threshold).values.astype(int).astype(str)).split('0')
                max_length_unified = len(max(continuous_true, key=len))
                unified = max_length_unified >= 10
                if not unified:
                    avg_unified = np.nan

                if ttc or unified:
                    df['location'] = loc_id
                    df['lc_id'] = lc_id-1
                    df['conflict_id'] = conflict_count
                    df['veh_id_i'] = lc_veh_id
                    df['veh_id_j'] = interact_veh_id
                    data2save.append(df)
                    conflict_cases.append([loc_id, lc_id-1, conflict_count, lc_veh_id, interact_veh_id, ttc, unified, avg_ttc, avg_unified])
                    conflict_count += 1
                    progress_bar.set_postfix({'Cases in total': conflict_count})

interaction_cases = pd.DataFrame(interaction_cases, columns=['location', 'lc_id', 'veh_id_i', 'frame_start', 'frame_end', 'int_veh_ids'])
interaction_cases.to_csv(path_output + 'intensity_evaluation/highD_interactive_LC.csv', index=False)

conflict_cases = pd.DataFrame(conflict_cases, columns=['location', 'lc_id', 'conflict_id', 'veh_id_i', 'veh_id_j', 'TTC', 'Unified', 'avg_TTC', 'avg_intensity'])
conflict_cases.to_csv(path_output + 'intensity_evaluation/highD_conflict_LC.csv', index=False)

data2save = pd.concat(data2save).reset_index(drop=True)
data2save.to_hdf(path_output + 'intensity_evaluation/highD_conflict_LC_data.h5', key='data', mode='w')
