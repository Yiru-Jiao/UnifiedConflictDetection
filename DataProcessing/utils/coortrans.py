'''
This file contains the methods for coordinate transformation for interaction extraction.
'''

import numpy as np
import pandas as pd


class coortrans():
    def __init__(self):
        pass


    def rotate_coor(self, xyaxis, yyaxis, x2t, y2t):
        '''
        Rotate the coordinates (x2t, y2t) to the coordinate system with the y-axis along (xyaxis, yyaxis).

        Parameters:
        - xyaxis: x-coordinate of the y-axis in the new coordinate system
        - yyaxis: y-coordinate of the y-axis in the new coordinate system
        - x2t: x-coordinate to be rotated
        - y2t: y-coordinate to be rotated

        Returns:
        - x: rotated x-coordinate
        - y: rotated y-coordinate
        '''
        x = yyaxis/np.sqrt(xyaxis**2+yyaxis**2)*x2t-xyaxis/np.sqrt(xyaxis**2+yyaxis**2)*y2t
        y = xyaxis/np.sqrt(xyaxis**2+yyaxis**2)*x2t+yyaxis/np.sqrt(xyaxis**2+yyaxis**2)*y2t
        return x, y


    def transform_coor(self, pairs, view):
        '''
        Transform the coordinates of the pairs to the relative view or the view of a selected ego vehicle.

        Parameters:
        - pairs: DataFrame containing the pairs of coordinates
        - view: 'relative' for relative view or 'i' for view of vehicle i or 'j' for view of vehicle j

        Returns:
        - transformed pairs DataFrame
        '''
        if 'frame_id' in pairs.columns:
            time_ref = 'frame_id'
        elif 'time' in pairs.columns:
            time_ref = 'time'
        
        if view=='relative':
            # Calculate the reference coordinate system based on the relative view
            coor_ref = pd.DataFrame({'x_axis':pairs['vx_i']-pairs['vx_j'], 
                                     'y_axis':pairs['vy_i']-pairs['vy_j'], 
                                     'x_origin':pairs['x_i'], 
                                     'y_origin':pairs['y_i']}, index=pairs.index)
            condition = ((pairs['vx_i']-pairs['vx_j'])==0)&((pairs['vy_i']-pairs['vy_j'])==0)
            coor_ref.loc[condition, ['x_axis','y_axis']] = pairs.loc[condition, ['hx_i','hy_i']].values
        elif view=='i':
            # Calculate the reference coordinate system based on the view of vehicle i
            coor_ref = pd.DataFrame({'x_axis':pairs['hx_i'], 
                                     'y_axis':pairs['hy_i'], 
                                     'x_origin':pairs['x_i'], 
                                     'y_origin':pairs['y_i']}, index=pairs.index)
        elif view=='j':
            # Calculate the reference coordinate system based on the view of vehicle j
            dict_rename = {key: key.replace('_i','_m').replace('_j','_i').replace('_m','_j') for key in pairs.columns}
            dict_rename.pop(time_ref, None)
            pairs = pairs.rename(columns=dict_rename)
            coor_ref = pd.DataFrame({'x_axis':pairs['hx_i'], 
                                     'y_axis':pairs['hy_i'], 
                                     'x_origin':pairs['x_i'], 
                                     'y_origin':pairs['y_i']}, index=pairs.index)
        
        # Rotate the coordinates of pairs and update the DataFrame
        x_i, y_i = np.zeros(len(pairs)), np.zeros(len(pairs))
        x_j, y_j = self.rotate_coor(coor_ref['x_axis'], coor_ref['y_axis'], pairs['x_j']-coor_ref['x_origin'], pairs['y_j']-coor_ref['y_origin'])
        pairs = pairs.assign(x_i=x_i, y_i=y_i, x_j=x_j, y_j=y_j)
        for var in ['h', 'v']:
            for obj in ['i', 'j']:
                x, y = self.rotate_coor(coor_ref['x_axis'], coor_ref['y_axis'], pairs[f'{var}x_{obj}'], pairs[f'{var}y_{obj}'])
                pairs[f'{var}x_{obj}'] = x
                pairs[f'{var}y_{obj}'] = y

        return pairs


    def angle(self, vec1x, vec1y, vec2x, vec2y):
        '''
        Calculate the angle between two vectors.

        Parameters:
        - vec1x: x-component of the first vector
        - vec1y: y-component of the first vector
        - vec2x: x-component of the second vector
        - vec2y: y-component of the second vector

        Returns:
        - angle: angle between the two vectors
        '''
        sin = vec1x * vec2y - vec2x * vec1y
        cos = vec1x * vec2x + vec1y * vec2y
        return np.arctan2(sin, cos)


    def TransCoorVis(self, pairs, surrounding, relative=True):
        '''
        Transform the coordinates of the pairs and their surroundings
        to the relative view or the view of an selected ego vehicle.
        Note: for visualisation

        Parameters:
        - pairs: DataFrame containing the pairs of coordinates
        - surrounding: DataFrame containing the coordinates of the surroundings
        - relative: boolean indicating whether to use relative view or not

        Returns:
        - transformed pairs DataFrame
        - transformed surrounding DataFrame
        '''
        if relative:
            # Calculate the reference coordinate system based on the relative view
            coor_ref = pd.DataFrame({'x_axis':pairs['vx_i']-pairs['vx_j'], 
                                     'y_axis':pairs['vy_i']-pairs['vy_j'], 
                                     'x_origin':pairs['x_i'], 
                                     'y_origin':pairs['y_i']}, index=pairs.index)
            condition = ((pairs['vx_i']-pairs['vx_j'])==0)&((pairs['vy_i']-pairs['vy_j'])==0)
            coor_ref.loc[condition, ['x_axis','y_axis']] = pairs.loc[condition, ['hx_i','hy_i']].values
        else:
            # Calculate the reference coordinate system based on the view of vehicle i
            coor_ref = pd.DataFrame({'x_axis':pairs['hx_i'], 
                                     'y_axis':pairs['hy_i'], 
                                     'x_origin':pairs['x_i'], 
                                     'y_origin':pairs['y_i']}, index=pairs.index)

        # Rotate the coordinates of pairs and update the DataFrame
        x_i, y_i = np.zeros(len(pairs)), np.zeros(len(pairs))
        x_j, y_j = self.rotate_coor(coor_ref['x_axis'], coor_ref['y_axis'], pairs['x_j']-coor_ref['x_origin'], pairs['y_j']-coor_ref['y_origin'])
        pairs = pairs.assign(x_i=x_i, y_i=y_i, x_j=x_j, y_j=y_j)
        for var in ['h', 'v']:
            for obj in ['i', 'j']:
                x, y = self.rotate_coor(coor_ref['x_axis'], coor_ref['y_axis'], pairs[f'{var}x_{obj}'], pairs[f'{var}y_{obj}'])
                pairs[f'{var}x_{obj}'] = x
                pairs[f'{var}y_{obj}'] = y
        
        # Rotate the coordinates of the surroundings and update the DataFrame
        coor_ref['y_j'] = pairs['y_j']
        coor_ref = coor_ref.loc[surrounding.index.values].reset_index()
        surrounding = surrounding.reset_index()
        x, y = self.rotate_coor(coor_ref['x_axis'], coor_ref['y_axis'], surrounding['x']-coor_ref['x_origin'], surrounding['y']-coor_ref['y_origin'])
        hx, hy = self.rotate_coor(coor_ref['x_axis'], coor_ref['y_axis'], surrounding['hx'], surrounding['hy'])
        surrounding = surrounding.assign(x=x, y=y, hx=hx, hy=hy)

        if 'frame_id' in surrounding.columns:
            time_ref = 'frame_id'
        elif 'time' in surrounding.columns:
            time_ref = 'time'
        return pairs, surrounding.set_index(time_ref)
    