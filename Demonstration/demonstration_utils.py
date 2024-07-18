'''
This script contains the functions for demonstrating the proposed method.
'''

import sys
import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib import pyplot as plt
from scipy.special import erf
sys.path.append('./')
import DataProcessing.utils.TwoDimTTC as TwoDimTTC
from DataProcessing.utils.coortrans import coortrans
coortrans = coortrans()
from GaussianProcessRegression.training_utils import *


def lognormal_cdf(x, mu, sigma):
    '''
    Calculate the cumulative distribution function (CDF) of a lognormal distribution.

    Parameters:
    - x: Input values.
    - mu: Mean of the lognormal distribution.
    - sigma: Standard deviation of the lognormal distribution.

    Returns:
    - CDF values.
    '''
    return 1/2+1/2*erf((np.log(x)-mu)/sigma/np.sqrt(2))


def lognormal_pdf(x, mu, sigma, rescale=True):
    '''
    Calculate the probability density function (PDF) of a lognormal distribution.

    Parameters:
    - x: Input values.
    - mu: Mean of the lognormal distribution.
    - sigma: Standard deviation of the lognormal distribution.
    - rescale: Whether to rescale the PDF to have a maximum value of 1.

    Returns:
    - PDF values.
    '''
    p = 1/x/np.sqrt(2*np.pi)/sigma*np.exp(-1/2*(np.log(x)-mu)**2/sigma**2)
    if rescale:
        mode = np.exp(mu-sigma**2)
        pmax = 1/mode/np.sqrt(2*np.pi)/sigma*np.exp(-1/2*sigma**2)
        p = p/pmax
    return p


def extreme_cdf(x, mu, sigma, n=10):
    '''
    Calculate the cumulative distribution function (CDF) of the extreme value distribution. 
    The distribution describes the probability of the number of extreme (minima) events where
    the variable X is larger than x within n trials.
    The events are assumed to follow a lognormal distribution.

    Parameters:
    - x: Input values.
    - mu: Mean of the base lognormal distribution.
    - sigma: Standard deviation of the base lognormal distribution.
    - n: Number of trials, i.e., level of the extreme events. 

    Returns:
    - CDF values.
    '''
    return (1-lognormal_cdf(x,mu,sigma))**n


def extreme_pdf(x, mu, sigma, n=10, rescale=True):
    '''
    Calculate the probability density function (PDF) of the extreme value distribution.

    Parameters:
    - x: Input values.
    - mu: Mean of the base lognormal distribution.
    - sigma: Standard deviation of the base lognormal distribution.
    - n: Number of trials, i.e., level of the extreme events.

    Returns:
    - PDF values.
    '''
    p = n*(1-lognormal_cdf(x,mu,sigma))**(n-1)*lognormal_pdf(x,mu,sigma)
    if rescale:
        pmax = p.max() # it's too complicated to calculate the exact maximum value
        p = p/pmax
    return p


def locate_lane_change(reference, lateral_position, lane_markings, veh_width, return_position=False):
    '''
    Identifies the start and end points of a lane change based on the given reference and lateral position.

    Parameters:
        reference (array-like): Monotonically increasing reference values, such as longitudinal position or time.
        lateral_position (array-like): Lateral position values.
        lane_markings (array-like): Lateral position values of the lane markings.
        veh_width (float): Width of the vehicle.
        return_position (bool, optional): If True, also returns the start and end positions of the lane change. Default is False.

    Returns:
        tuple: A tuple containing the start and end points of the lane change. If return_position is True, the tuple also includes the start and end positions.

    Notes:
        - The reference array should be monotonically increasing.
    '''

    # make a copy of the lateral position array
    input_lateral_position = lateral_position.copy()

    # make sure the first lateral position is smaller than the last
    if lateral_position[-1] < lateral_position[0]:
        lane_markings = lateral_position.max() - lane_markings
        lateral_position = lateral_position.max() - lateral_position

    # calculate the derivative of the lateral position, derivative >= 0 is necessary for lane change
    lateral_derivative = np.gradient(lateral_position)

    # start to count a lane change when the vehicle deviates more than 1/3 of its width from the lane center
    smaller_position = lane_markings[np.argsort(abs(lane_markings-lateral_position[0]))][:2].mean()
    smaller_position = smaller_position + veh_width / 3
    larger_position = lane_markings[np.argsort(abs(lane_markings-lateral_position[-1]))][:2].mean()
    larger_position = larger_position - veh_width / 3

    lateral_position = (lateral_position - (smaller_position + larger_position) / 2) / (larger_position - smaller_position) * 2
    transformed = (-lateral_position - lateral_position ** 3) * np.exp(-lateral_position ** 2)

    ref_start = reference[lateral_derivative >= 0][transformed[lateral_derivative >= 0].argmax()]
    ref_end = reference[lateral_derivative >= 0][transformed[lateral_derivative >= 0].argmin()]

    if return_position:
        pos_start = input_lateral_position[reference <= ref_start].max()
        pos_end = input_lateral_position[reference >= ref_end].min()
        return ref_start, ref_end, pos_start, pos_end
    else:
        return ref_start, ref_end


def compute_phi(events, path_output):
    """
    Compute mu and sigma using a trained model.

    Args:
        events (pandas.DataFrame): Input data containing event information.
        path_output (str): Path to the output directory.

    Returns:
        pandas.DataFrame: proximity_phi containing trip_id, time, mu, and sigma.

    """
    # Check if CUDA is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using {device} device')

    if device=='cpu':
        num_threads = torch.get_num_threads()
        print(f'Number of available threads: {num_threads}')
        torch.set_num_threads(round(num_threads/2))

    # Mirror the coordinates as the model is trained on highD where the y-axis points downwards
    events = events.rename(columns={'x_i':'y_i', 'y_i':'x_i', 'x_j':'y_j', 'y_j':'x_j',
                                    'vx_i':'vy_i', 'vy_i':'vx_i', 'vx_j':'vy_j', 'vy_j':'vx_j',
                                    'hx_i':'hy_i', 'hy_i':'hx_i', 'hx_j':'hy_j', 'hy_j':'hx_j'})
    events['psi_i'] = coortrans.angle(1, 0, events['hx_i'], events['hy_i'])
    events['psi_j'] = coortrans.angle(1, 0, events['hx_j'], events['hy_j'])

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

    return proximity_phi


def determine_conflicts(data, conflict_indicator, parameters):
    '''
    Determine conflicts based on the given indicator and parameters.

    Parameters:
    - data: Input data.
    - conflict_indicator: Indicator for conflict detection ('TTC', 'PSD', or 'Unified').
    - parameters: Parameters for conflict detection.

    Returns:
    - Data with conflict information.
    '''
    data = data.reset_index()

    if conflict_indicator=='TTC':
        ttc_threshold = parameters[0]
        data['conflict'] = False
        data['indicator_value'] = TwoDimTTC.TTC(data, 'values')
        data.loc[(data['indicator_value']<ttc_threshold), 'conflict'] = True
        return data
    
    elif conflict_indicator=='DRAC':
        drac_threshold = parameters[0]
        data['s_box'] = TwoDimTTC.CurrentD(data, 'values')
        data.loc[data['s_box']<1e-6, 's_box'] = 1e-6
        data['conflict'] = False
        data['delta_v'] = np.sqrt((data['vx_i']-data['vx_j'])**2 + (data['vy_i']-data['vy_j'])**2)
        follower_speed = data['forward'].astype(int)*data['speed_i'] + (1-data['forward'])*data['speed_j']
        leader_speed = data['forward'].astype(int)*data['speed_j'] + (1-data['forward'])*data['speed_i']
        data['indicator_value'] = data['delta_v']**2 / 2 / data['s_box']
        data.loc[follower_speed<=leader_speed, 'indicator_value'] = 0.
        data.loc[(data['indicator_value']>drac_threshold), 'conflict'] = True
        return data

    elif conflict_indicator=='PSD':
        psd_threshold = parameters[0]
        data['s_box'] = TwoDimTTC.CurrentD(data, 'values')
        data.loc[data['s_box']<1e-6, 's_box'] = 1e-6
        data['conflict'] = False
        brake_dec = 5.5 # braking deceleration rate
        follower_speed = data['forward'].astype(int)*data['speed_i'] + (1-data['forward'])*data['speed_j']
        data['braking_distance'] = follower_speed**2 / 2 / brake_dec
        data['indicator_value'] = data['s_box'] / data['braking_distance']
        data.loc[data['indicator_value']<psd_threshold, 'conflict'] = True
        return data
    
    elif conflict_indicator=='Unified':
        n, proximity_phi = parameters
        data['s_centroid'] = np.sqrt((data['x_i']-data['x_j'])**2 + (data['y_i']-data['y_j'])**2)
        data = data.merge(proximity_phi, on=['trip_id','time'])
        data['probability'] = extreme_cdf(data['s_centroid'].values, data['mu'].values, data['sigma'].values, n)
        data['conflict'] = False
        # 0.5 means that the probability of conflict is larger than the probability of non-conflict
        data.loc[data['probability']>0.5, 'conflict'] = True
        return data


def warning(events, meta, parameters, indicator, record_data=False):
    '''
    Perform warning analysis on the given events and metadata.

    Parameters:
    - events: Event data.
    - meta: Metadata.
    - parameters: Parameters for conflict detection.
    - indicator: Indicator for conflict detection ('TTC', 'PSD', or 'Unified').
    - record_data: Whether to record additional data.

    Returns:
    - Warning analysis results.
    '''
    events = events.sort_values(['trip_id','time'])
    events = events.set_index('trip_id')
    meta = meta.rename(columns={'webfileid':'trip_id'}).set_index('trip_id')
    meta = meta[['event start time','event end time','moment']].copy()
    trip_ids = meta.index

    ## Apply to each trip
    if record_data:
        indicated_events = []
    for trip_id in trip_ids:
        data = events.loc[trip_id].copy()
        moment = meta.loc[trip_id]['moment'] # moment of the minimum distance

        data = determine_conflicts(data, indicator, parameters)
        true_warning = data[(data['conflict'])&(data['event'])]
        event_period = data[data['event']]
        meta.loc[trip_id,'warning period'] = len(true_warning)/len(event_period)

        # the 3 seconds in the event before the moment are supposed to be dangerous
        within_3s = data[(data['time']>=moment-3)&(data['time']<=moment)&(data['event'])]
        true_warning = within_3s[within_3s['conflict']]
        if len(true_warning)>0:
            meta.loc[trip_id,'true warning'] = True
        else:
            meta.loc[trip_id,'true warning'] = False

        # the first 3 seconds are assumed to be safe
        beginning = data['time'].min()
        first_3s = data[(data['time']>=beginning)&(data['time']<=beginning+3)&(~data['event'])]
        false_warning = first_3s[first_3s['conflict']]
        if len(false_warning)>0:
            meta.loc[trip_id,'false warning'] = True
        else:
            meta.loc[trip_id,'false warning'] = False

        # record the first warning before the moment
        if record_data:
            indicated_events.append(data)
            warning = data[data['time']<=moment]['conflict'].astype(int).values
            warning_change = warning[1:] - warning[:-1]
            first_warning = np.where(warning_change==1)[0]
            if len(first_warning)>0:
                meta.loc[trip_id,'first warning'] = data.loc[first_warning[-1]+1,'time']
            else:
                meta.loc[trip_id,'first warning'] = np.nan
        
    if record_data:
        indicated_events = pd.concat(indicated_events).reset_index(drop=True)
        return meta.reset_index(), indicated_events
    else:
        return meta.reset_index()


def read_data(indicator, path_output):
    warning = pd.read_csv(path_output + 'conflict_probability/' + indicator + '_warning.csv')
    statistics, optimal = extract_warning_info(warning, indicator)
    return warning, statistics, optimal


def read_selected(conflict_indicator, path_output):
    selected = pd.read_csv(path_output + 'conflict_probability/optimal_warning/' + conflict_indicator + '_NearCrashes.csv', index_col=0)
    selected = selected[(selected['true warning'])]
    return selected


def extract_warning_info(warning, conflict_indicator):
    statistics = warning.groupby(['threshold']).agg({'true warning':'sum','false warning':'sum','warning period':'median'})
    statistics['true positive rate'] = statistics['true warning'] / warning['trip_id'].nunique()
    statistics['false positive rate'] = statistics['false warning'] / warning['trip_id'].nunique()
    if conflict_indicator=='DRAC' or conflict_indicator=='Unified':
        statistics = statistics.sort_values(by=['false positive rate','true positive rate','threshold'], ascending=[True, True, False]).reset_index()
    else:
        statistics = statistics.sort_values(by=['false positive rate','true positive rate','threshold']).reset_index()

    statistics['combined rate'] = (1-statistics['true positive rate'])**2+(statistics['false positive rate'])**2
    optimal_rate = statistics['combined rate'].min()
    optimal_warning = statistics[statistics['combined rate']==optimal_rate]
    optimal_threshold = optimal_warning.iloc[0]
    print(conflict_indicator, 
            'optimal threshold:', optimal_threshold['threshold'], 
            'tpr:', round(optimal_threshold['true positive rate']*100, 2),
            'fpr:', round(optimal_threshold['false positive rate']*100, 2))
    optimal_warning = warning[warning['threshold']==optimal_threshold['threshold']]
    
    return statistics, optimal_warning.set_index('trip_id')


def plot_warning(warning_ttc, warning_drac, warning_psd, warning_unified):
    statistics_ttc, optimal_ttc, ttc_selected = warning_ttc
    statistics_drac, optimal_drac, drac_selected = warning_drac
    statistics_psd, optimal_psd, psd_selected = warning_psd
    statistics_unified, optimal_unified, unified_selected = warning_unified

    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.2), constrained_layout=True)
    cmap = mpl.cm.plasma
    
    xticks = []
    statistics_list = [statistics_psd, statistics_drac, statistics_ttc, statistics_unified]
    optimal_list = [optimal_psd, optimal_drac, optimal_ttc, optimal_unified]
    color_list = [cmap(0.8), cmap(0.5), cmap(0.3), cmap(0.)]
    ls_list = ['s:','p-.','H--','o-']
    for statistics_, optimal_, color, ls in zip(statistics_list, optimal_list, color_list, ls_list):
        statistics_ = statistics_.reset_index()
        optimal_point = statistics_[statistics_['threshold']==optimal_['threshold'].iloc[0]]
        statistics_ = statistics_[['false positive rate','true positive rate']].drop_duplicates()
        axes[0].plot(statistics_['false positive rate'], statistics_['true positive rate'], 
                    ls, label=' ', color=color, ms=2, mfc='none', mew=0.5, lw=0.75, zorder=10)
        axes[0].plot(optimal_point['false positive rate'], optimal_point['true positive rate'],
                    ls[0], label=' ', color=color, ms=3, mfc='none', mew=0.75, zorder=10)
        axes[0].plot(optimal_point['false positive rate'], optimal_point['true positive rate'],
                    ls[0], label=' ', color=color, ms=6, mfc='none', mew=0.6, zorder=10)
        xticks.append([optimal_point['false positive rate'].iloc[0], str(optimal_point['threshold'].iloc[0])])
        radius = np.sqrt((1-optimal_point['true positive rate'])**2+optimal_point['false positive rate']**2).iloc[0]
        axes[0].add_patch(plt.Circle((0, 1), radius, color=color, fill=False, clip_on=True, lw=0.25, zorder=-5, ls=ls[1:]))

    radius = np.sqrt((1-optimal_point['true positive rate'])**2+optimal_point['false positive rate']**2).iloc[0]
    axes[0].add_patch(plt.Circle((0, 1), radius, color=cmap(0.), fill=False, clip_on=True, lw=0.25, zorder=-5))
    ax_focus = axes[0].inset_axes([0.43, 0.1, 0.53, 0.53], xlim=(-0.01, 0.16), ylim=(0.84, 1.01))
    ax_focus.tick_params(axis='both', labelsize=8, pad=1)
    ax_focus.set_yticks([0.85, 0.90, 0.95, 1.0])
    ax_focus.set_xticks([0, 0.05, 0.10, 0.15])
    for statistics_, optimal_, color, ls in zip(statistics_list, optimal_list, color_list, ls_list):
        statistics_ = statistics_.reset_index()
        optimal_point = statistics_[statistics_['threshold']==optimal_['threshold'].iloc[0]]
        statistics_ = statistics_[['false positive rate','true positive rate']].drop_duplicates()
        ax_focus.plot(statistics_['false positive rate'], statistics_['true positive rate'], 
                    ls, label=' ', color=color, ms=2, mfc='none', mew=0.5, lw=0.75, zorder=10)
        ax_focus.plot(optimal_point['false positive rate'], optimal_point['true positive rate'],
                    ls[0], label=' ', color=color, ms=3, mfc='none', mew=0.75, zorder=10)
        ax_focus.plot(optimal_point['false positive rate'], optimal_point['true positive rate'],
                    ls[0], label=' ', color=color, ms=6, mfc='none', mew=0.6, zorder=10)
        xticks.append([optimal_point['false positive rate'].iloc[0], str(optimal_point['threshold'].iloc[0])])
        radius = np.sqrt((1-optimal_point['true positive rate'])**2+optimal_point['false positive rate']**2).iloc[0]
        ax_focus.add_patch(plt.Circle((0, 1), radius, color=color, fill=False, clip_on=True, lw=0.35, zorder=-5, ls=ls[1:]))
    rect, lines = axes[0].indicate_inset_zoom(ax_focus, edgecolor='k')
    for line in lines:
        line.set_color('k')
        line.set_linestyle('--')
        line.set_linewidth(0.5)
    rect.set_edgecolor('k')
    rect.set_linestyle('--')
    rect.set_linewidth(0.5)

    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend([(handles[0]), (handles[1],handles[2]), 
                    (handles[3]), (handles[4],handles[5]),
                    (handles[6]), (handles[7],handles[8]),
                    (handles[9]), (handles[10],handles[11])],
                    ['PSD', 'Optimal PSD', 'DRAC', 'Optimal DRAC', 'TTC', 'Optimal TTC', 'Unified', 'Optimal Unified'],
                    bbox_to_anchor=(1., 0.5), loc='center left', fontsize=8, frameon=False)
    axes[0].set_xlabel('False positive rate')
    axes[0].set_title('ROC curves of metrics', fontsize=9)
    axes[0].set_xlim(-0.05, 1.05)
    axes[0].set_ylabel('True positive rate')
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])

    selected_list = [psd_selected, drac_selected, ttc_selected, unified_selected]
    for _selected, color, marker, label, pos in zip(selected_list, color_list, ['s','p','H','o'], ['PSD', 'DRAC', 'TTC', 'Unified'], [0, 1, 2, 3]):
        axes[1].boxplot(_selected['warning period'].dropna()*100, labels=[label+'\n'+str(round(_selected['warning period'].median()*100, 1))+'%'],
                        positions=[pos], widths=0.5,
                        boxprops=dict(color=color, lw=0.75),
                        whiskerprops=dict(color=color, lw=0.75),
                        capprops=dict(color=color, lw=0.75),
                        medianprops=dict(color=color, lw=0.75),
                        flierprops=dict(marker=marker, mfc='none', mec=color, ms=4, mew=0.75))
        
        axes[2].boxplot(_selected['timeliness'].dropna(), labels=[label+f'\n{_selected['timeliness'].median():.2f}s'],
                        positions=[pos], widths=0.5,
                        boxprops=dict(color=color, lw=0.75),
                        whiskerprops=dict(color=color, lw=0.75),
                        capprops=dict(color=color, lw=0.75),
                        medianprops=dict(color=color, lw=0.75),
                        flierprops=dict(marker=marker, mfc='none', mec=color, ms=4, mew=0.5))
        
        print(label, 'true warning events:', len(_selected), 
              'median period:', round(_selected['warning period'].dropna().median()*100, 2),
              'median timeliness:', round(_selected['timeliness'].dropna().median(), 2))

    axes[1].set_title('Warning period (%)', fontsize=9)
    axes[1].set_ylim([-5, 105])
    axes[1].set_yticks([0, 20, 40, 60, 80, 100])
    axes[2].set_title('Warning timeliness (s)', fontsize=9)

    for ax in axes:
        asp = np.diff(ax.get_xlim())[0] / np.diff(ax.get_ylim())[0]
        ax.set_aspect(asp)

    return fig
