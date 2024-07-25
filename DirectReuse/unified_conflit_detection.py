import torch
import gpytorch
import numpy as np
import math

# Check if CUDA is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')

if device=='cpu':
    num_threads = torch.get_num_threads()
    print(f'Number of available threads: {num_threads}')
    torch.set_num_threads(round(num_threads/2))

# Set random seed for reproducibility
manualSeed = 131
torch.manual_seed(manualSeed)
np.random.seed(manualSeed)


# SVGP model: Sparse Variational Gaussian Process
class SVGP(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points):

        # Determine variational distribution and strategy
        # Number of inducing points is better to be smaller than 1000 to speed up the training
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True
            )
        super(SVGP, self).__init__(variational_strategy)

        # Mean module
        self.mean_module = gpytorch.means.ConstantMean()

        # Kernel module
        mixture_kernel = gpytorch.kernels.SpectralMixtureKernel(num_mixtures=10, ard_num_dims=10)
        rbf_kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RQKernel(ard_num_dims=10))
        self.covar_module = mixture_kernel + rbf_kernel

        # To make mean positive
        self.softplus = torch.nn.Softplus()

    def forward(self, x):
        mean_x = self.mean_module(x)
        mean_x = self.softplus(mean_x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def define_model(num_inducing_points):
    # Create representative points for the input space
    # This is defined when training. Don't change it when applying the model.
    inducing_points = np.concatenate([np.random.uniform(4.,12.,(num_inducing_points,1)), # length_i
                                      np.random.uniform(4.,12.,(num_inducing_points,1)), # length_j
                                      np.random.uniform(-1,1,(num_inducing_points,1)), # hx_j
                                      np.random.uniform(-1,1,(num_inducing_points,1)), # hy_j
                                      np.random.uniform(0.,20.,(num_inducing_points,1)), # delta_v
                                      np.random.uniform(0.,400.,(num_inducing_points,1)), # delta_v2
                                      np.random.uniform(0.,3000.,(num_inducing_points,1)), # speed_i2
                                      np.random.uniform(0.,3000.,(num_inducing_points,1)), # speed_j2
                                      np.random.uniform(-5.5,5.5,(num_inducing_points,1)), # acc_i
                                      np.random.uniform(-np.pi,np.pi,(num_inducing_points,1))], # rho
                                      axis=1)
    inducing_points = torch.from_numpy(inducing_points).float()

    # Define the model
    model = SVGP(inducing_points)
    # Define the likelihood of the model
    likelihood = gpytorch.likelihoods.GaussianLikelihood()

    return model, likelihood


## Load trained model
model, likelihood = define_model(num_inducing_points=100) # number of inducing points is determined when training and cannot be changed here
model.load_state_dict(torch.load(f'./model_52epoch.pth', map_location=torch.device(device)))
likelihood.load_state_dict(torch.load(f'./likelihood_52epoch.pth', map_location=torch.device(device)))
model.eval()
likelihood.eval()
model = model.to(device)
likelihood = likelihood.to(device)
print('Model loaded successfully.')


def rotate_coor(xyaxis, yyaxis, x2t, y2t):
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
    x = yyaxis/math.sqrt(xyaxis**2+yyaxis**2)*x2t-xyaxis/math.sqrt(xyaxis**2+yyaxis**2)*y2t
    y = xyaxis/math.sqrt(xyaxis**2+yyaxis**2)*x2t+yyaxis/math.sqrt(xyaxis**2+yyaxis**2)*y2t
    return x, y


def angle(vec1x, vec1y, vec2x, vec2y):
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
    return math.atan2(sin, cos)


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
    return 1/2+1/2*math.erf((math.log(x)-mu)/sigma/math.sqrt(2))


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


def assess_conflict(states, coordinate_orentation='upwards', output='probability', n=25):
    """
    Assess the probability or intensity of conflict between two vehicles.

    Parameters:
    states (tuple or list): A tuple or list containing the states of the two vehicles. 
                            The states should be in the following order:
                            (x_j, y_j, vy_i, vx_j, vy_j, hx_j, hy_j, length_i, length_j, acc_i)
    coordinate_orentation (str, optional): The orientation of the coordinate system. Default is 'upwards',
                                           which means the y-axis points upwards. It can also be 'downwards'.
    output (str, optional): The type of assessment result to return. 
                            It can be 'probability', 'intensity', or 'both'. Default is 'probability'.
    n (int, optional): The level of conflict intensity to use for computing the probability. 
                       Only applicable when output is 'probability'. Default is 25.

    Returns:
    float or tuple: The probability or intensity of conflict between the two vehicles. 
                    If output is 'probability', a float representing the probability of conflict is returned.
                    If output is 'intensity', a float representing the intensity of conflict is returned.
                    If output is 'both', a tuple containing both the probability and intensity of conflict is returned.

    """

    x_j, y_j, vy_i, vx_j, vy_j, hx_j, hy_j, length_i, length_j, acc_i = states 
    # In a ego view it is supposed that x_i=0, y_i=0, vx_i=0, hx_i=0, hy_i=1
    x_i, y_i, vx_i, hx_i, hy_i = 0., 0., 0., 0., 1.
    if coordinate_orentation=='upwards':
        # Mirror the coordinates as the model is trained on highD where the y-axis points downwards
        x_i, y_i, x_j, y_j = y_i, x_i, y_j, x_j
        vx_i, vy_i, vx_j, vy_j = vy_i, vx_i, vy_j, vx_j
        hx_i, hy_i, hx_j, hy_j = hy_i, hx_i, hy_j, hx_j

    # Transform coordinates into relative coordinate system
    if (vx_i-vx_j==0)&(vy_i-vy_j==0):
        ref_xyaxis, ref_yyaxis = hx_i, hy_i
    else:
        ref_xyaxis = vx_i - vx_j
        ref_yyaxis = vy_i - vy_j
    rx_j, ry_j = rotate_coor(ref_xyaxis, ref_yyaxis, x_j, y_j)
    rho = angle(1., 0., rx_j, ry_j)
    
    # features = ['length_i','length_j','hx_j','hy_j','delta_v','delta_v2','speed_i2','speed_j2','acc_i','rho']
    delta_v2 = (vx_i-vx_j)**2+(vy_i-vy_j)**2
    delta_v = math.sqrt(delta_v2)
    speed_i2 = vx_i**2+vy_i**2
    speed_j2 = vx_j**2+vy_j**2
    features = [[length_i, length_j, hx_j, hy_j, delta_v, delta_v2, speed_i2, speed_j2, acc_i, rho]]

    # Compute mu and sigma
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        f_dist = model(torch.Tensor(features).to(device))
        y_dist = likelihood(f_dist)
        mu, sigma = y_dist.mean.cpu().numpy()[0], y_dist.variance.sqrt().cpu().numpy()[0]

    proximity = math.sqrt(x_j**2+y_j**2)
    if output=='probability':
        return extreme_cdf(proximity, mu, sigma, n)
    elif output=='intensity':
        # 0.5 means that the probability of conflict is larger than the probability of non-conflict
        max_intensity = math.log(0.5)/math.log(1-lognormal_cdf(proximity, mu, sigma))
        max_intensity = max(1., max_intensity)
        return max_intensity
    elif output=='both':
        probability = extreme_cdf(proximity, mu, sigma, n)
        max_intensity = math.log(0.5)/math.log(1-lognormal_cdf(proximity, mu, sigma))
        max_intensity = max(1., max_intensity)
        return probability, max_intensity
