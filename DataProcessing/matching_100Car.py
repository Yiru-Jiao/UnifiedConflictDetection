'''
This script matches the event target with one of the surrounding vehicles detected by the radar.
Note that this matching is very basic and may be incorrect due to factors such as:
1) detection failure of the target vehicle
2) noise introduced in the process of trajectory reconstruction
'''

import pandas as pd
import numpy as np
import utils.TwoDimTTC as TwoDimTTC

# Define file paths
path_cleaned = '../Process_100Car/CleanedData/'  # Path to cleaned data outside the repository
path_processed = '../Process_100Car/ProcessedData/'  # Path to processed data outside the repository
path_matched = './Data/InputData/100Car/'  # Path to save matched data


# Some targets are not counted in matching because
# 1) the target is not a vehicle
# or 2) the target is a vehicle but not interacting with the ego vehicle
# or 3) the target information is not available
uncounted_target = ['Single vehicle conflict', 'obstacle/object in roadway', 'parked vehicle', 'Other']

# Move the target vehicle points to the center
def move_to_center(events):
    forward = events.loc[events['forward']]
    center = forward[['x_j', 'y_j']].values + forward[['hx_j', 'hy_j']].values * np.tile(forward['length_j'].values, (2, 1)).T / 2
    events.loc[events['forward'], ['x_j', 'y_j']] = center

    rearward = events.loc[~events['forward']]
    center = rearward[['x_j', 'y_j']].values - rearward[['hx_j', 'hy_j']].values * np.tile(rearward['length_j'].values, (2, 1)).T / 2
    events.loc[~events['forward'], ['x_j', 'y_j']] = center
    return events


# Process crash and near-crash data
for crash_type in ['Crash', 'NearCrash']:
    print('Processing', crash_type, 'data...')

    # Read ego vehicle data
    data_ego = pd.read_hdf(path_processed + 'HundredCar_' + crash_type + '_Ego.h5', key='data').reset_index(drop=True)
    data_ego = data_ego.rename(columns={'x_ekf': 'x', 'y_ekf': 'y', 'psi_ekf': 'psi', 'v_ekf': 'speed', 'acc_ekf': 'acc_i'})
    data_ego['event'] = data_ego['event'].astype(bool)
    data_ego = data_ego.sort_values(['trip_id', 'time']).set_index('trip_id')

    # Read surrounding vehicle data
    data_sur = pd.read_hdf(path_processed + 'HundredCar_' + crash_type + '_Surrounding.h5', key='data').reset_index(drop=True)
    data_sur = data_sur.drop(columns=['x', 'y'])
    data_sur = data_sur.rename(columns={'x_ekf': 'x', 'y_ekf': 'y', 'psi_ekf': 'psi', 'v_ekf': 'speed'})
    data_sur['forward'] = data_sur['forward'].astype(bool)
    trip_ids = data_sur['trip_id'].unique()
    data_sur = data_sur.sort_values(['trip_id', 'time']).set_index('trip_id')

    # Read metadata
    meta = pd.read_csv(path_cleaned + 'HundredCar_metadata_' + crash_type + 'Event.csv').set_index('webfileid')
    meta = meta.loc[trip_ids]
    meta = meta[~meta['target'].isin(uncounted_target)]

    data_ego = data_ego[data_ego.index.isin(meta.index)]
    data_sur = data_sur[data_sur.index.isin(meta.index)]

    print(f'There are {data_ego.index.nunique()} trips processed')

    events = []
    for trip_id in meta.index:

        meta_trip = meta.loc[trip_id]
        df_ego = data_ego.loc[trip_id].set_index('sync').iloc[1:]
        df_sur = data_sur.loc[trip_id].iloc[1:]  # Remove the first row because it is not reliable
        meta.loc[trip_id, 'event start time'] = df_ego.loc[meta_trip['event start'], 'time']
        meta.loc[trip_id, 'event end time'] = df_ego.loc[meta_trip['event end'], 'time']
        df_ego['event'] = False
        df_ego.loc[meta_trip['event start']:meta_trip['event end'], 'event'] = True

        merged = df_ego[df_ego['event']].merge(df_sur, on='time', suffixes=('_ego', '_sur'))
        forward = merged[merged['forward']].groupby('time')['range'].idxmin()
        forward = merged.loc[forward][['time', 'target_id']]
        rearward = merged[~merged['forward']].groupby('time')['range'].idxmin()
        rearward = merged.loc[rearward][['time', 'target_id']]
        merged = merged.loc[merged.groupby('time')['range'].idxmin()][['time', 'target_id']]

        unique_forward = forward['target_id'].unique()
        unique_rearward = rearward['target_id'].unique()
        unique_merged = merged['target_id'].unique()
        if ('lead' in meta.loc[trip_id]['target']) and (len(unique_forward) == 1):
            target_id = unique_forward[0]
        elif ('follow' in meta.loc[trip_id]['target']) and (len(unique_rearward) == 1):
            target_id = unique_rearward[0]
        elif len(unique_merged) == 1:
            target_id = unique_merged[0]
        else:
            continue

        veh_i = df_ego[['time', 'x', 'y', 'psi', 'speed', 'acc_i', 'event']].reset_index()
        veh_i['trip_id'] = trip_id
        veh_j = df_sur[df_sur['target_id'] == target_id][['time', 'x', 'y', 'psi', 'speed', 'target_id', 'range', 'forward']].copy()
        df = veh_i.merge(veh_j, on='time', suffixes=('_i', '_j'), how='inner')
        if df[df['event']]['range'].min() < 4.5:  # Ensure no other vehicles can be between the ego and the target during the event
            df['width_i'] = meta.loc[trip_id]['ego_width']
            df['length_i'] = meta.loc[trip_id]['ego_length']
            df['width_j'] = meta.loc[trip_id]['target_width']
            df['length_j'] = meta.loc[trip_id]['target_length']
            df['hx_i'] = np.cos(df['psi_i'])
            df['hy_i'] = np.sin(df['psi_i'])
            df['hx_j'] = np.cos(df['psi_j'])
            df['hy_j'] = np.sin(df['psi_j'])
            df['vx_i'] = df['speed_i'] * df['hx_i']
            df['vy_i'] = df['speed_i'] * df['hy_i']
            df['vx_j'] = df['speed_j'] * df['hx_j']
            df['vy_j'] = df['speed_j'] * df['hy_j']
            df = move_to_center(df)
            df['s_box'] = TwoDimTTC.CurrentD(df, 'values')
            df.loc[df['s_box'] < 1e-6, 's_box'] = 1e-6
            meta.loc[trip_id, 'moment'] = df.loc[df[df['event']]['s_box'].idxmin(), 'time']

            duration_before_event = df[df['event']]['time'].min() - df['time'].min()
            # make sure there are at least 6 seconds movement before the event (3s safe and 3s dangerous)
            if (duration_before_event >= 6):
                # make sure the ego vehicle is not hard-braking in the first 3 seconds
                if (np.all(df[df['time'] <= (df['time'].min() + 3.)]['acc_i'] > -1.)):
                    # make sure the vehicles move faster than 3 m/s (i.e., not in congestion) in the first 3 seconds
                    if np.all(df[['speed_i', 'speed_j']].iloc[0] > 3.):
                        events.append(df)

    events = pd.concat(events).sort_values(['trip_id', 'time']).reset_index(drop=True)
    events.to_hdf(path_matched + 'HundredCar_' + crash_type + 'es.h5', key='data')
    print('Minimum net distance:', events['s_box'].min())

    meta = meta.loc[events['trip_id'].unique()]
    meta.to_csv(path_matched + 'HundredCar_metadata_' + crash_type + 'es.csv')
    print(f'There are {len(meta)} {crash_type}es matched and saved.')
