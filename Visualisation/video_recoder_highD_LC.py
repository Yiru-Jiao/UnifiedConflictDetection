'''
This script produce images for the dynamic visualisation of some lane-changing interactions in highD data.
'''

import os
import sys
import glob
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
sys.path.append('./')
import DataProcessing.utils.TwoDimTTC as TwoDimTTC
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


summary2vis = pd.read_csv(path_output + 'intensity_evaluation/highD_conflict_LC_to_visualse.csv')
metadatafiles =  sorted(glob.glob(path_raw + 'highD/RecordingMetadata/*.csv'))
metadata = []
for metadatafile in metadatafiles:
    df = pd.read_csv(metadatafile)
    metadata.append(df)
metadata = pd.concat(metadata).set_index('id')

for idx in summary2vis.index:
    if summary2vis.loc[idx, 'location']<1:
        continue

    loc_id = summary2vis.loc[idx, 'location']
    veh_id_i = summary2vis.loc[idx, 'veh_id_i']
    veh_id_j = summary2vis.loc[idx, 'veh_id_j']
    frame_start = summary2vis.loc[idx, 'frame_start']
    frame_end = summary2vis.loc[idx, 'frame_end']
    if veh_id_i==260496: # This is an accident and the ego vehicle made another lane-change before
        frame_start = frame_start-120
    
    data = pd.read_hdf(path_processed+'highD/highD_'+str(loc_id).zfill(2)+'.h5', key='data')
    data = data.sort_values(['track_id','frame_id']).set_index('track_id')

    meta_tracks = []
    for fileid in metadata[metadata['locationId']==loc_id].index:
        meta_track = pd.read_csv(path_raw+'highD/01_tracksMeta.csv')
        meta_track['id'] = (fileid*10000 + meta_track['id']).astype(int)
        meta_tracks.append(meta_track)
    meta_tracks = pd.concat(meta_tracks)

    lane_markings = [float(y) for lane in ['lowerLaneMarkings','upperLaneMarkings'] for y in metadata.loc[veh_id_i//10000][lane].split(';')]
    lane_markings = np.sort(lane_markings)

    veh_i = data.loc[veh_id_i].drop(columns=['laneId','precedingId', 'followingId'])
    veh_i = veh_i[veh_i['frame_id'].between(frame_start, frame_end)]
    veh_j = data.loc[veh_id_j].drop(columns=['laneId','precedingId', 'followingId'])
    df = veh_i.merge(veh_j, on='frame_id', suffixes=('_i', '_j'), how='inner')
    df['time'] = np.round((df['frame_id'] - df['frame_id'].min())/10, 1)
    df['delta_v'] = np.sqrt((df['vx_i']-df['vx_j'])**2 + (df['vy_i']-df['vy_j'])**2)
    df['delta_v2'] = df['delta_v']**2
    df['speed_i2'] = df['speed_i']**2
    df['speed_j2'] = df['speed_j']**2
    df['acc_i'] = df['ax_i']*df['hx_i'] + df['ay_i']*df['hy_i']
    df['s_centroid'] = np.sqrt((df['x_i']-df['x_j'])**2 + (df['y_i']-df['y_j'])**2)
    df['TTC'] = TwoDimTTC.TTC(df, 'values')

    save_dir = path_output+'video_images/highD_LC/'
    save_dir = save_dir + summary2vis.loc[idx,'specification']+'/'
    save_dir = save_dir + f"intensity_{summary2vis.loc[idx,'intensity_lower']:.1f}" + '-' + f"{summary2vis.loc[idx,'intensity_upper']:.1f}/"
    save_dir = save_dir + f'{loc_id}_{veh_id_i}_{veh_id_j}/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    other_vehs = data[(data['frame_id']>=df['frame_id'].min())&
                      (data['frame_id']<=df['frame_id'].max())&
                      (data.index!=veh_id_i)].reset_index()
    df_view_i, other_vehs_view_i = coortrans.TransCoorVis(df.set_index('frame_id').copy(), other_vehs.set_index('frame_id').copy(), relative=False)
    df_relative = coortrans.transform_coor(df, view='relative')
    heading_j = df_view_i.reset_index()[['frame_id','hx_j','hy_j']]
    rho = coortrans.angle(1, 0, df_relative['x_j'], df_relative['y_j']).reset_index().rename(columns={0:'rho'})
    rho['frame_id'] = df_relative['frame_id']
    features = ['length_i','length_j','hx_j','hy_j','delta_v','delta_v2','speed_i2','speed_j2','acc_i','rho']
    interaction_situation = df.drop(columns=['hx_j','hy_j']).merge(heading_j, on='frame_id').merge(rho, on='frame_id')
    interaction_situation = interaction_situation[features+['frame_id']].set_index('frame_id')

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        f_dist = model(torch.Tensor(interaction_situation.values).to(device))
        y_dist = likelihood(f_dist)
        mu_list, sigma_list = y_dist.mean.cpu().numpy(), y_dist.variance.sqrt().cpu().numpy()
    df['mu'] = mu_list
    df['sigma'] = sigma_list
    df['n_hat'] = np.log(0.5)/np.log(1-lognormal_cdf(df['s_centroid'], df['mu'], df['sigma']))
    df.loc[df['n_hat']<1, 'n_hat'] = 1

    lc_start, lc_end = locate_lane_change(veh_i['frame_id'].values, veh_i['y'].values, lane_markings, veh_i['width'].mean())
    df = df[(df['frame_id']>=lc_start-30)&(df['frame_id']<=lc_end+30)]

    for frameid in tqdm(df['frame_id'].values, desc=f'{veh_id_i}_{veh_id_j}'):
        fig = visual_highD(lane_markings, frameid, veh_i, veh_j, df, df_view_i, other_vehs, other_vehs_view_i, 
                           interaction_situation, model, likelihood, device, lc_start, lc_end)
        fig.savefig(save_dir+f'frame_{frameid}.png', bbox_inches='tight', dpi=600)
        plt.close(fig)
