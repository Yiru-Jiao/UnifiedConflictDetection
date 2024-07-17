'''
This file contains some functions for visualising interactions.
'''

import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import colors
from matplotlib.collections import PatchCollection
from mpl_toolkits.mplot3d import Axes3D
font = {'family' : 'Arial',
        'size'   : 9}
plt.rc('font', **font)
plt.rcParams['mathtext.fontset'] = 'stix' #dejavuserif

sys.path.append('./')
from DataProcessing.utils.coortrans import coortrans
coortrans = coortrans()
from Demonstration.demonstration_utils import *
from GaussianProcessRegression.training_utils import *


class RotateRectangle(Rectangle): # adapted from a Stack Overflow answer https://stackoverflow.com/a/60413175
    def __init__(self, xy, width, length, **kwargs):
        super().__init__(xy, width, length, **kwargs)
        self.rel_point_of_rot = np.array([width/2,length/2])
        self.xy_center = self.get_xy()
        self.set_angle(self.angle)

    def _apply_rotation(self):
        angle_rad = self.angle * np.pi / 180
        m_trans = np.array([[np.cos(angle_rad), -np.sin(angle_rad)],
                            [np.sin(angle_rad), np.cos(angle_rad)]])
        shift = -m_trans @ self.rel_point_of_rot
        self.set_xy(self.xy_center + shift)

    def set_angle(self, angle):
        self.angle = angle
        self._apply_rotation()

    def set_xy_center(self, xy):
        self.xy_center = xy
        self._apply_rotation()


def rgb_color(colorname, alpha=1.):
    color = colors.to_rgb(colorname)
    return (color[0], color[1], color[2], alpha)


def draw_vehs(ax, vehs, vehs_ij, annotate=False):
    patches = []
    for vehid in vehs['track_id'].values:
        veh = vehs[vehs['track_id']==vehid].iloc[0]
        angle = np.arctan2(veh['hy'], veh['hx']) * 180 / np.pi - 90
        if annotate:
            ax.text(veh['x'], veh['y'], str(veh['track_id']), fontsize=10)
        patches.append(RotateRectangle((veh['x'], veh['y']), veh['width'], veh['length'], angle=angle))
    ax.add_collection(PatchCollection(patches, color='tab:blue', alpha=0.2))
    
    for vehid, color in zip(['_i','_j'],['r','b']):
        x, y, hx, hy, width, length = vehs_ij[['x'+vehid, 'y'+vehid, 'hx'+vehid, 'hy'+vehid, 'width'+vehid, 'length'+vehid]].values[0]
        angle = np.arctan2(hy, hx) * 180 / np.pi - 90
        patches = [(RotateRectangle((x, y), width=width, length=length, angle=angle))]
        ax.add_collection(PatchCollection(patches, color=color, alpha=0.6))

    return ax


def draw_vehs_ij(ax, vehs_ij, draw_j=True, arrow=True):
    if draw_j:
        for vehid, color in zip(['_i','_j'],['r','b']):
            x, y, hx, hy, width, length = vehs_ij[['x'+vehid, 'y'+vehid, 'hx'+vehid, 'hy'+vehid, 'width'+vehid, 'length'+vehid]].values[0]
            angle = np.arctan2(hy, hx) * 180 / np.pi - 90
            patches = [(RotateRectangle((x, y), width=width, length=length, angle=angle))]
            ax.add_collection(PatchCollection(patches, color=color, alpha=0.6))
            if arrow:
                ax.arrow(x, y, hx*5, hy*5, head_width=2, head_length=2, fc=color, ec=color, clip_on=False)
    else:
        x, y, hx, hy, width, length = vehs_ij[['x_i', 'y_i', 'hx_i', 'hy_i', 'width_i', 'length_i']].values[0]
        angle = np.arctan2(hy, hx) * 180 / np.pi - 90
        patches = [(RotateRectangle((x, y), width=width, length=length, angle=angle))]
        ax.add_collection(PatchCollection(patches, color='r', alpha=0.6))
        if arrow:
            ax.arrow(x, y, hx*5, hy*5, head_width=2, head_length=2, fc='r', ec='r', clip_on=False)
    return ax


def visual_100Car(t, df, df_view_i, df_relative, interaction_situation, model, likelihood, device, n, conflict_start, conflict_end):
    fig = plt.figure(figsize=(7.5,2.7))
    gs = fig.add_gridspec(24, 29, wspace=1)
    cmap = mpl.cm.plasma

    ax_v = fig.add_subplot(gs[:10,:6])
    ax_v.plot(df['time'], df['speed_i'], '--', label='Ego vehicle', c='r', lw=0.5)
    ax_v.plot(df['time'], df['speed_j'], '-.', label='Target vehicle', c='b', lw=0.5)
    ax_v.plot(df[df['time']<=t]['time'], df[df['time']<=t]['speed_i'], c='r', lw=1)
    ax_v.plot(df[df['time']<=t]['time'], df[df['time']<=t]['speed_j'], c='b', lw=1)
    ax_v.set_ylim(ax_v.get_ylim())
    ax_v.fill_between([conflict_start, conflict_end], ax_v.get_ylim()[0], ax_v.get_ylim()[1], color=cmap(0.8), alpha=0.15, edgecolor='none')
    min_v = min(df['speed_i'].min(), df['speed_j'].min())
    max_v = max(df['speed_i'].max(), df['speed_j'].max())
    ax_v.set_yticks([np.ceil(min_v), np.floor(max_v)])
    ax_v.vlines(0, ax_v.get_ylim()[0], ax_v.get_ylim()[1], colors=cmap(0.8), lw=0.3)
    ax_v.set_title('Speed (m/s)', fontsize=9)
    ax_v.set_xticks([np.ceil(df['time'].min()), 0])
    ax_v.set_xticklabels([])
    ax_v.legend(loc='lower left', fontsize=8, frameon=False, handlelength=1., handletextpad=0.1, borderaxespad=0.1, borderpad=0.1)

    ax_acc = fig.add_subplot(gs[:10,7:13])
    ax_acc.plot(df['time'], df['acc_i'], '--', label='Ego vehicle', c='r', lw=0.5)
    ax_acc.plot(df[df['time']<=t]['time'], df[df['time']<=t]['acc_i'], c='r', lw=1)
    ax_acc.set_ylim(ax_acc.get_ylim())
    ax_acc.fill_between([conflict_start, conflict_end], ax_acc.get_ylim()[0], ax_acc.get_ylim()[1], color=cmap(0.8), alpha=0.15, edgecolor='none')
    if df['acc_i'].max()>0 and df['acc_i'].min()<0:
        ax_acc.set_yticks([np.ceil(df['acc_i'].min()), 0, np.floor(df['acc_i'].max())])
    else:
        ax_acc.set_yticks([np.ceil(df['acc_i'].min()), np.floor(df['acc_i'].max())])
    ax_acc.vlines(0, ax_acc.get_ylim()[0], ax_acc.get_ylim()[1], colors=cmap(0.8), lw=0.3)
    ax_acc.set_title('Acceleration (m/s$^2$)', fontsize=9)
    ax_acc.set_xticks(ax_v.get_xticks())
    ax_acc.set_xticklabels([])
    
    ax_ttc = fig.add_subplot(gs[14:,7:13])
    ax_drac = ax_ttc.twinx()
    ax_ttc.plot(df['time'], df['TTC'], '--', c=cmap(0.3), lw=0.5, label='TTC')
    ax_drac.plot(df['time'], df['DRAC'], '-.', c=cmap(0.5), lw=0.5, label='DRAC')
    ax_ttc.plot(df[df['time']<=t]['time'], df[df['time']<=t]['TTC'], c=cmap(0.3), lw=1)
    ax_drac.plot(df[df['time']<=t]['time'], df[df['time']<=t]['DRAC'], c=cmap(0.5), lw=1)
    ax_ttc.vlines(0, ax_ttc.get_ylim()[0], ax_ttc.get_ylim()[1], colors=cmap(0.8), lw=0.3)
    ax_ttc.set_title('TTC (s) and DRAC (m/s$^2$)', fontsize=9)
    ax_ttc.set_ylim(-0.4, 8.4)
    ax_drac.set_ylim(-0.045, 0.945)
    ax_ttc.fill_between([conflict_start, conflict_end], ax_ttc.get_ylim()[0], ax_ttc.get_ylim()[1], color=cmap(0.8), alpha=0.15, edgecolor='none')
    ax_ttc.set_yticks([0, 4.0, 8.0])
    ax_drac.set_yticks([0., 0.45, 0.9])
    ax_ttc.yaxis.label.set_color(cmap(0.3))
    ax_drac.yaxis.label.set_color(cmap(0.5))
    ax_ttc.tick_params(axis='y', colors=cmap(0.3), labelsize=8, pad=1)
    ax_drac.tick_params(axis='y', colors=cmap(0.5), labelsize=8, pad=1)
    ax_ttc.set_xlim(ax_v.get_xlim())
    ax_ttc.set_xticks(ax_v.get_xticks())
    ax_ttc.set_xlabel('Time (s)', fontsize=9, labelpad=1)
    handle1, label1 = ax_ttc.get_legend_handles_labels()
    handle2, label2 = ax_drac.get_legend_handles_labels()
    ax_ttc.legend(handle1+handle2, label1+label2, loc='center left', fontsize=8, frameon=False, handlelength=1., handletextpad=0.1, borderaxespad=0.1, borderpad=0.1)

    ax_unified = fig.add_subplot(gs[14:,:6])
    ax_unified.plot(df['time'], df['probability'], '--', c=cmap(0.), lw=0.5)
    ax_unified.plot(df[df['time']<=t]['time'], df[df['time']<=t]['probability'], c=cmap(0.), lw=1)
    ax_unified.set_ylim(-0.05, 1.05)
    ax_unified.fill_between([conflict_start, conflict_end], ax_unified.get_ylim()[0], ax_unified.get_ylim()[1], color=cmap(0.8), alpha=0.15, edgecolor='none')
    ax_unified.vlines(0, ax_unified.get_ylim()[0], ax_unified.get_ylim()[1], colors=cmap(0.8), lw=0.3)
    ax_unified.set_title('Conflict probability', fontsize=9)
    ax_unified.set_xticks(ax_v.get_xticks())
    ax_unified.set_xlabel('Time (s)', fontsize=9, labelpad=1)
    ax_unified.yaxis.label.set_color(cmap(0.))
    ax_unified.tick_params(axis='y', colors=cmap(0.))

    x = np.linspace(-37+1e-6,37,74+1)
    y = np.linspace(-50+1e-6,50,100+1)
    xx, yy = np.meshgrid(x, y)
    num_points = int(len(x)*len(y))
    df2compute = pd.DataFrame(np.zeros((num_points,2)), columns=['x_j','y_j'])
    features = ['length_i','width_i','length_j','width_j','hx_i','hy_i','hx_j','hy_j','vx_i','vy_i','vx_j','vy_j']
    df2compute[features] = np.array([df_view_i.loc[t][features].values]*num_points).astype(float)
    df2compute['x_i'] = 0.
    df2compute['y_i'] = 0.
    df2compute['x_j'] = xx.reshape(-1)
    df2compute['y_j'] = yy.reshape(-1)
    df2compute = coortrans.transform_coor(df2compute, view='relative')
    rho_list = coortrans.angle(1, 0, df2compute['x_j'], df2compute['y_j'])
    s_list = np.sqrt(df2compute['x_j']**2 + df2compute['y_j']**2).values
    virtual_scene = np.array([interaction_situation.loc[t].values]*len(rho_list))
    virtual_scene[:,-1] = rho_list
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        f_dist = model(torch.Tensor(virtual_scene).to(device))
        y_dist = likelihood(f_dist)
        mu_list, sigma_list = y_dist.mean.cpu().numpy(), y_dist.variance.sqrt().cpu().numpy()
        f_dist = model(torch.Tensor(np.array([interaction_situation.loc[t].values])).to(device))
        y_dist = likelihood(f_dist)
        mu, sigma = y_dist.mean.cpu().numpy()[0], y_dist.variance.sqrt().cpu().numpy()[0]

    ax_interaction = fig.add_subplot(gs[:-6,13:21])
    ax_interaction.set_title('Proximity density', fontsize=9)
    ax_interaction.set_xlabel('Lateral position (m)', fontsize=9, labelpad=1)
    ax_interaction.yaxis.tick_right()
    ax_interaction.set_yticklabels([])
    safe_spacing = lognormal_pdf(s_list, mu_list, sigma_list).reshape(xx.shape)
    levels = np.arange(0-0.05, 1+0.1, 0.1)
    im = ax_interaction.contourf(xx, yy, safe_spacing, levels=levels, cmap='YlGn', alpha=0.8, zorder=-10)
    cax_interaction = fig.add_subplot(gs[-2:,14:20])
    cbr_interaction = fig.colorbar(im, cax=cax_interaction, orientation='horizontal')
    cbr_interaction.set_ticks(np.arange(0, 1+0.005, 0.1)[::2])
    cbr_interaction.set_label('Normalised $f_S(s;\\phi^{(t)})$', fontsize=9, labelpad=1)
    cax_interaction.set_ylim(0,2)
    cax_interaction.set_frame_on(False)
    safe_spacing = lognormal_pdf(df_relative.loc[t]['s_centroid'], mu, sigma)
    cax_interaction.arrow(safe_spacing, 2, 0, -0.8, fc='b', ec='b', head_width=0.025, head_length=0.5)

    ax_conflict = fig.add_subplot(gs[:-6,21:])
    ax_conflict.set_title('Conflict probability', fontsize=9)
    ax_conflict.set_xlabel('Lateral position (m)', fontsize=9, labelpad=1)
    ax_conflict.set_ylabel('Longitudinal position (m)', fontsize=9, labelpad=12, rotation=270)
    ax_conflict.yaxis.set_label_position('right')
    extreme_spacing = extreme_cdf(s_list, mu_list, sigma_list, n).reshape(xx.shape)
    im = ax_conflict.contourf(xx, yy, extreme_spacing, levels=levels, cmap='YlOrRd', alpha=0.8, zorder=-10)
    cax_conflict = fig.add_subplot(gs[-2:,22:28])
    cbr_conflict = fig.colorbar(im, cax=cax_conflict, orientation='horizontal')
    cbr_conflict.set_ticks(np.arange(0, 1+0.005, 0.1)[::2])
    cbr_conflict.set_label('$C(s;\\phi^{(t)},'+f'n={n})$', fontsize=9, labelpad=1)
    cax_conflict.set_ylim(0,2)
    cax_conflict.set_frame_on(False)
    extreme_spacing = extreme_cdf(df_relative.loc[t]['s_centroid'], mu, sigma, n)
    cax_conflict.arrow(extreme_spacing, 2, 0, -0.8, fc='b', ec='b', head_width=0.025, head_length=0.5)

    ax_interaction = draw_vehs_ij(ax_interaction, df_view_i.loc[t].to_frame().T, arrow=False)
    ax_conflict = draw_vehs_ij(ax_conflict, df_view_i.loc[t].to_frame().T, arrow=False)

    df_traj = df[(df['time']<=t)&(df['time']>=(t-1))].reset_index(drop=True)
    for suffix, color in zip(['i','j'], ['r','b']):
        x_axis, y_axis = df_traj.iloc[-1]['hx_i'], df_traj.iloc[-1]['hy_i']
        traj_x, traj_y = coortrans.rotate_coor(x_axis, y_axis, 
                                                df_traj[f'x_{suffix}']-df_traj.iloc[-1]['x_i'], df_traj[f'y_{suffix}']-df_traj.iloc[-1]['y_i'])
        ax_interaction.plot(traj_x, traj_y, c=color, lw=0.7, alpha=0.5)

    for ax in [ax_interaction, ax_conflict]:
        ax.set_xlim(-37, 37)
        ax.set_ylim(-50, 50)
        ax.set_aspect('equal')
        ax.invert_xaxis() # x-axis is inverted to align with the coordinate system in highD

    ax_conflict.tick_params(axis='y', which='major', pad=12)
    for tick in ax_conflict.yaxis.get_majorticklabels():
        tick.set_horizontalalignment('center')

    return fig


def visual_highD(lane_markings, frameid, veh_i, veh_j, df, df_view_i, other_vehs, other_vehs_view_i, interaction_situation, model, likelihood, device, frame_1, frame_2):
    fig = plt.figure(figsize=(7.5,4))
    gs = fig.add_gridspec(28, 30, wspace=1)

    ax = fig.add_subplot(gs[:5,:])
    ax.set_xlim(min(veh_i['x'].min(), veh_j['x'].min()), max(veh_i['x'].max(), veh_j['x'].max()))
    ax.set_ylim(min(lane_markings)-3, max(lane_markings)+3)
    ax.hlines(lane_markings, other_vehs['x'].min(), other_vehs['x'].max(), color='gray', alpha=0.5, lw=0.5)
    ax.plot(veh_i[veh_i['frame_id']<=frameid]['x'], veh_i[veh_i['frame_id']<=frameid]['y'], 'r', lw=1, alpha=0.2)
    ax.plot(veh_j[veh_j['frame_id']<=frameid]['x'], veh_j[veh_j['frame_id']<=frameid]['y'], 'b', lw=1, alpha=0.35)
    ax = draw_vehs(ax, other_vehs[other_vehs['frame_id']==frameid].reset_index(), df[df['frame_id']==frameid], annotate=False)
    ax.xaxis.tick_top()
    ax.tick_params(axis='x', which='major', labelsize=8, pad=0)
    ax.tick_params(axis='y', which='major', labelsize=8, pad=0)
    ax.set_xlabel('Position (m)', fontsize=9, labelpad=5)
    ax.xaxis.set_label_position('top')
    ax.invert_yaxis() # y-axis is inverted because the y-axis in highD directs from top to bottom

    ax_v = fig.add_subplot(gs[8:16,1:7])
    ax_v.plot(df['time'], df['speed_i'], '--', label='Ego vehicle', c='r', lw=0.5)
    ax_v.plot(df['time'], df['speed_j'], '-.', label='Target vehicle', c='b', lw=0.5)
    ax_v.plot(df[df['frame_id']<=frameid]['time'], df[df['frame_id']<=frameid]['speed_i'], c='r', lw=1)
    ax_v.plot(df[df['frame_id']<=frameid]['time'], df[df['frame_id']<=frameid]['speed_j'], c='b', lw=1)
    ax_v.set_ylim(ax_v.get_ylim())
    ax_v.fill_between(df[(df['frame_id']>frame_1)&(df['frame_id']<frame_2)]['time'], ax_v.get_ylim()[0], ax_v.get_ylim()[1],
                        color=mpl.cm.plasma(0.8), alpha=0.15, edgecolor='none')
    min_v = min(df['speed_i'].min(), df['speed_j'].min())
    max_v = max(df['speed_i'].max(), df['speed_j'].max())
    ax_v.set_yticks([np.ceil(min_v), np.floor(max_v)])
    ax_v.set_title('Speed (m/s)', fontsize=9)
    ax_v.set_xticks([np.ceil(df['time'].min()), np.floor(df['time'].max())])
    ax_v.set_xticklabels([])
    ax_v.legend(loc='best', fontsize=8, frameon=False, handlelength=1., handletextpad=0.1, borderaxespad=0.1, borderpad=0.1)

    ax_acc = fig.add_subplot(gs[8:16,8:14])
    ax_acc.plot(df['time'], df['acc_i'], '--', label='Ego vehicle', c='r', lw=0.5)
    ax_acc.plot(df[df['frame_id']<=frameid]['time'], df[df['frame_id']<=frameid]['acc_i'], c='r', lw=1)
    if df['acc_i'].max()>0 and df['acc_i'].min()<0:
        if np.ceil(df['acc_i'].min())==0 or np.floor(df['acc_i'].max())==0:
            ax_acc.set_yticks([np.ceil(df['acc_i'].min()*10)/10, 0., np.floor(df['acc_i'].max()*10)/10])
            ax_acc.tick_params(axis='y', which='major', pad=1)
        elif np.ceil(df['acc_i'].min()) == np.floor(df['acc_i'].max()):
            ax_acc.set_yticks([np.ceil(df['acc_i'].min()*10)/10, np.floor(df['acc_i'].max()*10)/10])
            ax_acc.tick_params(axis='y', which='major', pad=1)
        else:
            ax_acc.set_yticks([np.ceil(df['acc_i'].min()), 0, np.floor(df['acc_i'].max())])
    else:
        ax_acc.set_yticks([np.ceil(df['acc_i'].min()), np.floor(df['acc_i'].max())])
    ax_acc.set_ylim(ax_acc.get_ylim())
    ax_acc.fill_between(df[(df['frame_id']>frame_1)&(df['frame_id']<frame_2)]['time'], ax_acc.get_ylim()[0], ax_acc.get_ylim()[1],
                        color=mpl.cm.plasma(0.8), alpha=0.15, edgecolor='none')
    ax_acc.set_title('Acceleration (m/s$^2$)', fontsize=9)
    ax_acc.set_xticks(ax_v.get_xticks())
    ax_acc.set_xticklabels([])
    
    ax_ttc = fig.add_subplot(gs[20:,8:14])
    ax_ttc.plot(df['time'], df['TTC'], '--', c=mpl.cm.plasma(0.3), lw=0.5, label='TTC')
    ax_ttc.plot(df[df['frame_id']<=frameid]['time'], df[df['frame_id']<=frameid]['TTC'], c=mpl.cm.plasma(0.3), lw=1)
    ax_ttc.fill_between(df[(df['frame_id']>frame_1)&(df['frame_id']<frame_2)]['time'], -0.5, 10.5, 
                        color=mpl.cm.plasma(0.8), alpha=0.15, edgecolor='none')
    ax_ttc.set_title('TTC (s)', fontsize=9)
    ax_ttc.set_ylim(-0.5, 10.5)
    ax_ttc.set_yticks([0, 5, 10])
    ax_ttc.yaxis.label.set_color(mpl.cm.plasma(0.3))
    ax_ttc.tick_params(axis='y', colors=mpl.cm.plasma(0.3), pad=1)
    ax_ttc.set_xlim(ax_v.get_xlim())
    ax_ttc.set_xticks(ax_v.get_xticks())
    ax_ttc.set_xlabel('Time (s)', fontsize=9, labelpad=1)

    ax_unified = fig.add_subplot(gs[20:,1:7])
    ax_unified.plot(df['time'], df['n_hat'], '--', c=mpl.cm.plasma(0.), lw=0.5)
    ax_unified.plot(df[df['frame_id']<=frameid]['time'], df[df['frame_id']<=frameid]['n_hat'], c=mpl.cm.plasma(0.), lw=1)
    ax_unified.set_yscale('log')
    y_lim = ax_unified.get_ylim()
    ax_unified.set_ylim(y_lim[0], y_lim[1])
    ax_unified.fill_between(df[(df['frame_id']>frame_1)&(df['frame_id']<frame_2)]['time'], y_lim[0], y_lim[1], 
                            color=mpl.cm.plasma(0.8), alpha=0.15, edgecolor='none')
    ax_unified.set_title('Conflict intensity', fontsize=9)
    ax_unified.set_xticks(ax_v.get_xticks())
    ax_unified.set_xlabel('Time (s)', fontsize=9, labelpad=1)
    ax_unified.yaxis.label.set_color(mpl.cm.plasma(0.))
    ax_unified.tick_params(axis='y', colors=mpl.cm.plasma(0.))

    x = np.linspace(-35+1e-6,35,70+1)
    y = np.linspace(-60+1e-6,60,120+1)
    xx, yy = np.meshgrid(x, y)
    num_points = int(len(x)*len(y))
    df2compute = pd.DataFrame(np.zeros((num_points,2)), columns=['x_j','y_j'])
    features = ['length_i','width_i','length_j','width_j','hx_i','hy_i','hx_j','hy_j','vx_i','vy_i','vx_j','vy_j']
    df2compute[features] = np.array([df_view_i.loc[frameid][features].values]*num_points).astype(float)
    df2compute['x_i'] = 0.
    df2compute['y_i'] = 0.
    df2compute['x_j'] = xx.reshape(-1)
    df2compute['y_j'] = yy.reshape(-1)
    df2compute = coortrans.transform_coor(df2compute, view='relative')
    rho_list = coortrans.angle(1, 0, df2compute['x_j'], df2compute['y_j'])
    s_list = np.sqrt(df2compute['x_j']**2 + df2compute['y_j']**2).values
    virtual_scene = np.array([interaction_situation.loc[frameid].values]*len(rho_list))
    virtual_scene[:,-1] = rho_list
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        f_dist = model(torch.Tensor(virtual_scene).to(device))
        y_dist = likelihood(f_dist)
        mu_list, sigma_list = y_dist.mean.cpu().numpy(), y_dist.variance.sqrt().cpu().numpy()
    s, mu, sigma, n_hat = df[df['frame_id']==frameid][['s_centroid','mu','sigma','n_hat']].values[0]
    s_max = df['s_centroid'].max()
    n_max = df['n_hat'].max()

    ax_conflict = fig.add_subplot(gs[8:-5,21:], projection='3d', computed_zorder=False)
    ax_conflict.set_title('Conflict intensity', fontsize=9)
    ax_interaction = fig.add_subplot(gs[8:-5,16:22])
    ax_interaction.set_title('Proximity density', fontsize=9)
    ax_interaction.set_xlabel('Lateral position (m)', fontsize=9, labelpad=1)
    ax_interaction.set_ylabel('Longitudinal position (m)', fontsize=9, labelpad=0, y=0.65)
    safe_spacing = lognormal_pdf(s_list, mu_list, sigma_list).reshape(xx.shape)
    levels = np.arange(0-0.05, 1+0.1, 0.1)
    im = ax_interaction.contourf(xx, yy, safe_spacing, levels=levels, cmap='YlGn', alpha=0.8, zorder=-10)
    ax_interaction = draw_vehs(ax_interaction, other_vehs_view_i.loc[frameid], df_view_i.loc[frameid].to_frame().T, annotate=False)
    
    df_traj = df[(df['frame_id']<=frameid)&(df['frame_id']>=(frameid-10))].reset_index(drop=True)        
    for suffix, color in zip(['i','j'], ['r','b']):
        x_axis, y_axis = df_traj.iloc[-1]['hx_i'], df_traj.iloc[-1]['hy_i']
        traj_x, traj_y = coortrans.rotate_coor(x_axis, y_axis, 
                                                df_traj[f'x_{suffix}']-df_traj.iloc[-1]['x_i'], df_traj[f'y_{suffix}']-df_traj.iloc[-1]['y_i'])
        ax_interaction.plot(traj_x, traj_y, c=color, lw=0.7, alpha=0.5)
    
    ax_interaction.set_xlim(-35, 35)
    ax_interaction.set_ylim(-60, 60)
    ax_interaction.invert_xaxis() # x-axis is inverted to align with the coordinate system in highD
    ax_interaction.set_aspect('equal')
    cax_interaction = fig.add_subplot(gs[-2:,16:22])
    cbr_interaction = fig.colorbar(im, cax=cax_interaction, orientation='horizontal')
    cbr_interaction.set_ticks(np.arange(0, 1+0.005, 0.1)[::2])
    cbr_interaction.set_label('Normalised $f_S(s;\\phi^{(t)})$', fontsize=9, labelpad=1)
    cax_interaction.set_ylim(0,2)
    cax_interaction.set_frame_on(False)
    safe_spacing = lognormal_pdf(s, mu, sigma)
    cax_interaction.arrow(safe_spacing, 2, 0, -0.8, fc='b', ec='b', head_width=0.025, head_length=0.5)

    n_list = np.round(np.linspace(1, n_max+1, 100))
    s_list = np.linspace(1e-6, s_max, 100)
    nn, ss = np.meshgrid(n_list, s_list)
    pp = extreme_cdf(ss, mu, sigma, nn)
    im = ax_conflict.plot_surface(ss, nn, pp, cmap='YlOrRd', vmin=0, vmax=1, zorder=-1, alpha=0.9)
    ax_conflict.contour(ss, nn, pp, zdir='x', offset=s, levels=[s], colors='b', linewidths=0.8, zorder=1)
    ax_conflict.scatter(s, n_hat, 0.5, c='b', marker='x', s=15, lw=1, zorder=1, label='$\\hat{n}=C^{-1}(s^{(t)},\\phi^{(t)},p=0.5)$')
    ax_conflict.plot(np.ones(100)*s, np.ones(100)*n_hat, np.linspace(0,0.5,100), 'k--', alpha=0.5, lw=0.8, zorder=1)
    ax_conflict.plot(s_list, np.ones(100)*n_hat, np.zeros(100), 'r-', zorder=0)
    ax_conflict.text(s_max, n_hat, 0.05, '$\\hat{n}=$'+str(round(n_hat)), zdir='x', color='r', fontsize=9, ha='right')
    cax_conflict = fig.add_subplot(gs[-1:,23:29])
    cbr_conflict = fig.colorbar(im, cax=cax_conflict, orientation='horizontal')
    cbr_conflict.set_ticks(np.arange(0, 1+0.005, 0.1)[::2])
    cbr_conflict.set_label('$C(n,s;\\phi^{(t)})$', fontsize=9, labelpad=1)
    cax_conflict.set_ylim(0,1)
    handles, labels = ax_conflict.get_legend_handles_labels()
    handles.extend(plt.plot([], [], 'b-', lw=0.8))
    labels.append('$p=C(n;s^{(t)},\\phi^{(t)})$')
    ax_conflict.legend(handles, labels, bbox_to_anchor=(0.55, -0.25), ncol=1, loc='lower center',
                        fontsize=8, frameon=False, handlelength=1.5, handletextpad=0.2, borderaxespad=0.1, borderpad=0.1)
    ax_conflict.view_init(15, 315)
    ax_conflict.set_xlim(0, s_list[-1])
    ax_conflict.set_ylim(0, n_list[-1])
    ax_conflict.set_zlim(0, 1)
    ax_conflict.set_zticks([0, 0.5, 1])
    ax_conflict.set_xlabel('$s$ (m)', labelpad=-4)
    ax_conflict.set_ylabel('$n$', labelpad=-3)
    ax_conflict.text(s_max, -10, 1.15, '$p$')
    ax_conflict.tick_params(axis='x', which='major', pad=-3)
    ax_conflict.tick_params(axis='y', which='major', pad=-3)
    ax_conflict.tick_params(axis='z', which='major', pad=-3)
    ax_conflict.invert_yaxis()
    ax_conflict.get_proj = lambda: np.dot(Axes3D.get_proj(ax_conflict), np.diag([0.9, 0.9, 1.1, 1]))

    return fig