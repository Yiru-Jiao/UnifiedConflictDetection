'''
This script extracts lane-changing data from the highD dataset.
The data have been downsampled to 10 fps.
The global view is transformed to the local view of both vehicle i and j,
where vehicle i is the subject vehicle, vehicle j is the target vehicle.
For model training, we use the following features:
features = [length_i, length_j, hx_j-hx_i, hy_j-hy_i, delta_v, delta_v2, speed_i2, speed_j2, acc_i, rho, s, scene_id]
here we do not include acc_j as we will use the 100Car data for validation, 
and the 100Car data does not have target vehicle's acceleration
'''

import numpy as np
import pandas as pd
from tqdm import tqdm
from utils.coortrans import coortrans

path_input = './Data/InputData/highD_LC/'
path_processed = './Data/ProcessedData/highD/'

manualSeed = 131
np.random.seed(manualSeed)


class LaneChangeExtractor():
    def __init__(self, data, initial_lc_id=0):
        super().__init__()
        data = data.sort_values(['track_id','frame_id']).set_index('track_id')
        data['acc'] = data['ax']*data['hx'] + data['ay']*data['hy'] # compute the acceleration in the heading direction
        self.data = data
        lane_change = data.groupby('track_id')['laneId'].nunique() > 1
        self.lc_track_ids = lane_change.index[lane_change].values
        self.initial_lc_id = initial_lc_id

    def extract_lanechange(self,):
        lc_id = self.initial_lc_id
        lane_changes = []
        for track_id in tqdm(self.lc_track_ids):
            veh = self.data.loc[track_id]
            preceding_vehs = veh['precedingId'][veh['precedingId']>0].unique()
            following_vehs = veh['followingId'][veh['followingId']>0].unique()
            if len(preceding_vehs)==0 or len(following_vehs)==0:
                continue # skip if the vehicle changed lane without interacting with other vehicles
            for interact_veh_id in np.concatenate([preceding_vehs, following_vehs]):
                veh_i = self.data.loc[track_id].drop(columns=['laneId','precedingId', 'followingId'])
                veh_j = self.data.loc[interact_veh_id].drop(columns=['laneId','precedingId', 'followingId'])
                df = veh_i.merge(veh_j, on='frame_id', suffixes=('_i', '_j'), how='inner')
                if len(df)<31:
                    continue # skip if the interaction is shorter than 3 seconds
                df['lc_id'] = lc_id
                df['track_id_i'] = track_id
                df['track_id_j'] = interact_veh_id
                lane_changes.append(df)
                lc_id += 1
        return pd.concat(lane_changes, ignore_index=True)


class SceneRepresentator(coortrans):
    def __init__(self, data, initial_scene_id):
        super().__init__()
        self.lc_ids = data['lc_id'].unique()
        data = data.sort_values(['lc_id','frame_id']).set_index('lc_id')
        self.data = data
        self.current_feature_size = 11
        self.initial_scene_id = initial_scene_id
        self.current_features_set = self.segment_data()

    # Segment data and organize features
    def segment_data(self,):
        current_features_set = []
        scene_id = self.initial_scene_id
        for lc_id in tqdm(self.lc_ids):
            df = self.data.loc[lc_id]
            df_view_i = self.transform_coor(df, 'i')
            df_view_j = self.transform_coor(df, 'j')
            df_view_relative = self.transform_coor(df, 'relative')
            indices_end = np.arange(20,len(df),10) # sample every 1 second
            for idx_end in indices_end:
                for df in [df_view_i, df_view_j]:
                    speed_i = df.iloc[idx_end-20:idx_end]['speed_i']
                    speed_j = df.iloc[idx_end-20:idx_end]['speed_j']
                    if ((speed_i<0.1)&(speed_j<0.1)).sum()<10: # if the vehicles do not move for more than 1 second, skip the segment
                        current_features = np.zeros(self.current_feature_size+1)
                        current_features[:2] = df.iloc[idx_end][['length_i','length_j']]
                        current_features[2] = df_view_i.iloc[idx_end]['hx_j']
                        current_features[3] = df_view_i.iloc[idx_end]['hy_j']
                        current_features[4] = np.sqrt((df.iloc[idx_end]['vx_i']-df.iloc[idx_end]['vx_j'])**2 + (df.iloc[idx_end]['vy_i']-df.iloc[idx_end]['vy_j'])**2)
                        current_features[5] = current_features[4]**2
                        current_features[6] = df.iloc[idx_end]['speed_i']**2
                        current_features[7] = df.iloc[idx_end]['speed_j']**2
                        current_features[8] = df.iloc[idx_end]['acc_i']
                        current_features[9] = self.angle(1, 0, df_view_relative.iloc[idx_end]['x_j'], df_view_relative.iloc[idx_end]['y_j'])
                        current_features[10] = np.sqrt(df_view_relative.iloc[idx_end]['x_j']**2 + df_view_relative.iloc[idx_end]['y_j']**2)
                        current_features[-1] = scene_id
                        current_features_set.append(current_features)
                        scene_id += 1
        current_features_set = pd.DataFrame(current_features_set,columns=['length_i','length_j','hx_j','hy_j',
                                                                          'delta_v','delta_v2','speed_i2','speed_j2','acc_i',
                                                                          'rho','s','scene_id'])
        current_features_set['scene_id'] = current_features_set['scene_id'].astype(int)
        return current_features_set


# Extract lane changes
initial_lc_id = 0
for loc_id in range(1,7):
    print('Extracting lane changes at location ' + str(loc_id) + '...')
    data = pd.read_hdf(path_processed+'highD_0'+str(loc_id)+'.h5', key='data')
    lce = LaneChangeExtractor(data, initial_lc_id)
    lane_change = lce.extract_lanechange()
    lane_change = lane_change.drop_duplicates(subset=['track_id_i','track_id_j','frame_id'])
    initial_lc_id = lane_change['lc_id'].max() + 1
    lane_change.to_hdf(path_processed+'lane_changing/lc_0'+str(loc_id)+'.h5', key='data')


# Separate all the lane changes into train, val, and test sets
data_all = pd.concat([pd.read_hdf(path_processed+'lane_changing/lc_0'+str(loc_id)+'.h5', key='data') for loc_id in range(1,7)], ignore_index=True)
data_all = data_all.dropna(subset=['length_i','width_i','length_j','width_j','speed_i','speed_j'])
lc_ids = data_all['lc_id'].unique()
train_lc_ids = np.random.RandomState(manualSeed).choice(lc_ids, int(0.6*len(lc_ids)), replace=False)
data_train = data_all[data_all['lc_id'].isin(train_lc_ids)]
val_lc_ids = np.random.RandomState(manualSeed).choice(np.setdiff1d(lc_ids, train_lc_ids), int(0.2*len(lc_ids)), replace=False)
data_val = data_all[data_all['lc_id'].isin(val_lc_ids)]
test_lc_ids = np.setdiff1d(np.setdiff1d(lc_ids, train_lc_ids), val_lc_ids)
data_test = data_all[data_all['lc_id'].isin(test_lc_ids)]


# Segment lane changes and save scenes
initial_scene_id = 0
for data, suffix in zip([data_train, data_val, data_test], ['train', 'val', 'test']):
    print('Segmenting ' + suffix + ' set...')
    sr = SceneRepresentator(data, initial_scene_id)
    sr.current_features_set.to_hdf(path_input + 'current_features_'+suffix+'.h5', key='features')
    initial_scene_id = sr.current_features_set['scene_id'].max() + 1
    print('Number of scenes in ' + suffix + ' set: ' + str(initial_scene_id - sr.initial_scene_id))
    print('Minimum net distance: ' + str(sr.current_features_set['s'].min()))
    print(sr.current_features_set.head())
