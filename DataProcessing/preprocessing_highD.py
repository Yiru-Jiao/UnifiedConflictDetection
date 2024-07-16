'''
This script preprocesses the highD dataset by performing the following steps:
1. Move the coordinates of vehicles to their centers.
2. Estimate the heading direction using an extended Kalman filter.
3. Extract meta information from the dataset.
4. Process the data for each location in the dataset.
5. Save the processed data in HDF5 format.

Agent type (car/truck/cyclist/pedestrian) is implicitly defined by width and length. 
The width of a car/truck is at least 1.5m, and the length is at least 3.5m.
'''

path_raw = './Data/RawData/'
path_processed = './Data/ProcessedData/'

import glob
from tqdm import tqdm
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import utils.get_heading_ekf as ekf


# Extract meta information
metadatafiles =  sorted(glob.glob(path_raw + 'highD/RecordingMetadata/*.csv'))
metadata = []
for metadatafile in metadatafiles:
    df = pd.read_csv(metadatafile)
    lane_markings = [float(y) for lane in ['lowerLaneMarkings','upperLaneMarkings'] for y in df[lane].iloc[0].split(';')]
    lane_markings = np.sort(lane_markings)
    df['max_y'] = lane_markings[-1] + lane_markings[0]
    lane_markings = lane_markings[-1] + lane_markings[0] - lane_markings
    df['corrected_LaneMarkings'] = ';'.join([str(round(x,2)) for x in lane_markings])
    df.to_csv(metadatafile, index=False)
    metadata.append(df)
metadata = pd.concat(metadata)
metadata['lane_num'] = metadata.lowerLaneMarkings.str.len()//5
metadata['numFrames'] = (metadata['frameRate']*metadata['duration']).astype(int)

print(metadata.groupby('locationId').agg({'numCars':'sum','numTrucks':'sum'}))
trackid_base = 10**len(str(int(metadata['numCars'].max())))
frameid_base = 10**len(str(int(metadata['numFrames'].max())))


ekf_params = np.array([100.2683, 0.01, 0.01, 11.1333, 2.5, 52.4380])
print('Processing order:', metadata.locationId.unique())
for locid in tqdm(metadata.locationId.unique(), desc='location'):
    loc = 'highD_' + str(locid).zfill(2)
    data_files = [str(id).zfill(2) + '_tracks' for id in metadata[(metadata.locationId==locid)]['id'].values]
    metadata_files = [str(id).zfill(2) + '_tracksMeta' for id in metadata[(metadata.locationId==locid)]['id'].values]
    max_y_list = metadata[(metadata.locationId==locid)]['max_y'].values
    data = []

    for data_file, metadata_file, max_y in tqdm(zip(data_files, metadata_files, max_y_list), total=len(data_files), desc='file'):
        file_id = int(data_file[:2])
        df = pd.read_csv(path_raw + 'highD/' + data_file +'.csv')
        meta = pd.read_csv(path_raw + 'highD/' + metadata_file +'.csv')
        df = df.rename(columns={'frame':'frame_id',
                                'id':'track_id',
                                'xVelocity':'vx',
                                'yVelocity':'vy',
                                'xAcceleration':'ax',
                                'yAcceleration':'ay',
                                'width':'length',
                                'height':'width'})
        df['direction'] = meta.set_index('id').reindex(df['track_id'].values)['drivingDirection'].values
        df = df[['track_id','frame_id','x','y','vx','vy','ax','ay','width','length',
                 'laneId','precedingId','followingId','direction']]
                
        # move the coordinates to the center of the vehicles, we don't consider the angle because
        # 1) the bounding box is not sure to have been detected along the heading direction
        # 2) on highway event lane-changes do not have large angles between the lanes
        # 3) further processing will be conducted later to restimate the heading and position
        df['x'] = df['x'] + df['length']/2
        df['y'] = df['y'] + df['width']/2
        # mirror the y-axis because the y-axis in highD points from top to bottom, which is opposite to the common convention
        df['y'] = max_y - df['y']
        df['vy'] = -df['vy']
        df['ay'] = -df['ay']
        # downsample from 25 fps to 10 fps and obtain heading direction using extended kalman filter
        track_ids = df['track_id'].unique()
        df = df.set_index('track_id')
        df = pd.concat(Parallel(n_jobs=15)(delayed(ekf.ekf)(ekf_params, df, track_id, False) for track_id in track_ids)).reset_index(drop=True)
        df['vx'] = df['speed_kf']*np.cos(df['psi_kf'])
        df['vy'] = df['speed_kf']*np.sin(df['psi_kf'])
        df['hx'] = np.cos(df['psi_kf'])
        df['hy'] = np.sin(df['psi_kf'])
        df = df[['track_id','frame_id','x_kf','y_kf', 'psi_kf', 'speed_kf',
                 'vx','vy','ax','ay','hx','hy','width','length',
                 'laneId','precedingId','followingId','direction']].rename(columns={'x_kf':'x','y_kf':'y','psi_kf':'psi_rad','speed_kf':'speed'})
        # redefine indcies to be unique for later data combination
        for var in ['track_id','precedingId','followingId']:
            df.loc[df[var]>0.5,var] = file_id*trackid_base+df.loc[df[var]>0.5,var].values
            df[var] = df[var].astype(int)
        df['frame_id'] = (file_id*frameid_base+df['frame_id']).astype(int)
        data.append(df)

    pd.concat(data).reset_index(drop=True).to_hdf(path_processed + 'highD/' + loc + '.h5', key='data')
    data = []