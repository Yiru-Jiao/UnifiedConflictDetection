'''
This script contains necessary functions for training and testing the Sparse Variational Gaussian Process (SVGP) model.
Condition variables: [length_i,length_j,hx_j-hx_i,hy_j-hy_i,'delta_v','speed_i','speed_j','acc_i','rho']
Reference variable: ['s']
he following classes are included:
1. DataOrganiser: a class that organizes the data for training and testing the SVGP model. 
                  It creates a dataloader and provides methods to retrieve data samples.
2. SVGP: a class that defines the Sparse Variational Gaussian Process (SVGP) model. 
         It inherits from the `gpytorch.models.ApproximateGP` class and implements the forward method.
3. train_val_test: a class that handles the training and validation process of the SVGP model. 
                   It creates the dataloaders, defines the model, likelihood, and loss function, 
                   and performs the training loop.
The file also includes helper functions for creating inducing points and a validation loop.
'''

import glob
import torch
import gpytorch
from tqdm import tqdm
import numpy as np
import pandas as pd
import warnings
from torch.utils.data import DataLoader

manualSeed = 131
torch.manual_seed(manualSeed)
np.random.seed(manualSeed)


# Create dataloader
class DataOrganiser:
    def __init__(self, dataset, path_input):
        self.dataset = dataset
        self.path_input = path_input
        self.read_data()

    def __len__(self,):
        return len(self.idx_list)

    def __getitem__(self, idx):
        # idx is the index of items in the dataset
        int_suit = self.interaction_situation.loc[self.idx_list[idx]].values
        int_suit = torch.from_numpy(int_suit).float()
        cur_spac = self.current_spacing.loc[self.idx_list[idx]].values
        cur_spac = torch.from_numpy(cur_spac).float()
        return int_suit, cur_spac

    def read_data(self,):
        features = pd.read_hdf(self.path_input + 'current_features_' + self.dataset + '.h5', key='features')
        self.idx_list = features['scene_id'].values
        features = features.set_index('scene_id')
        self.interaction_situation = features.drop(columns=['s']).copy()
        # log-transform spacing, and the spacing must be larger than 0
        if np.any(features['s']<=1e-6):
            warnings.warn('There are spacings smaller than or equal to 0.')
            features.loc[features['s']<=1e-6, 's'] = 1e-6
        self.current_spacing = np.log(features[['s']]).copy()
        features = []


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


class train_val_test():
    def __init__(self, device, num_inducing_points, path_input='./', path_output='./'):
        self.device = device
        self.path_input = path_input
        self.path_output = path_output

        # Define model and likelihood
        self.define_model(num_inducing_points)


    def create_dataloader(self, batch_size, beta=5):
        self.batch_size = batch_size
        self.beta = beta
        self.path_save = self.path_output+f'beta={self.beta}/'

        # Create dataloader
        self.train_dataloader = DataLoader(DataOrganiser('train', self.path_input), batch_size=self.batch_size, shuffle=True)
        self.val_dataloader = DataLoader(DataOrganiser('val', self.path_input), batch_size=self.batch_size, shuffle=True)
        self.test_dataloader = DataLoader(DataOrganiser('test', self.path_input), batch_size=self.batch_size, shuffle=True)
        print(f'Dataloader created, number of training samples: {len(self.train_dataloader.dataset)}\n')
       # Determine loss function
        self.loss_func = gpytorch.mlls.PredictiveLogLikelihood(self.likelihood, self.model, num_data=len(self.train_dataloader.dataset), beta=self.beta)


    def define_model(self, num_inducing_points):
        self.inducing_points = self.create_inducing_points(num_inducing_points)
        self.inducing_points = torch.from_numpy(self.inducing_points).float()

        # Define the model
        self.model = SVGP(self.inducing_points)

        # Define the likelihood of the model
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()


    def create_inducing_points(self, num_inducing_points):
        # Create representative points for the input space
        inducing_points = pd.DataFrame({'length_i': np.random.uniform(4.,12.,num_inducing_points),
                                        'length_j': np.random.uniform(4.,12.,num_inducing_points),
                                        'hx_j': np.random.uniform(-1,1,num_inducing_points),
                                        'hy_j': np.random.uniform(-1,1,num_inducing_points),
                                        'delta_v': np.random.uniform(0.,20.,num_inducing_points),
                                        'delta_v2': np.random.uniform(0.,400.,num_inducing_points),
                                        'speed_i2': np.random.uniform(0.,3000.,num_inducing_points),
                                        'speed_j2': np.random.uniform(0.,3000.,num_inducing_points),
                                        'acc_i': np.random.uniform(-5.5,5.5,num_inducing_points),
                                        'rho': np.random.uniform(-np.pi,np.pi,num_inducing_points)})
        return inducing_points.values


    # Validation loop
    def val_loop(self,):
        self.model.eval()
        self.likelihood.eval()

        val_loss = 0
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            progress_bar = tqdm(enumerate(self.val_dataloader), unit='batch', total=len(self.val_dataloader))
            progress_bar.set_description('Validation')
            for count_batch, (interaction_situation, current_spacing) in progress_bar:
                output = self.model(interaction_situation.to(self.device))
                loss = -self.loss_func(output, current_spacing.squeeze().to(self.device)).item()
                val_loss += loss
                progress_bar.set_postfix({'val_loss=': val_loss/(count_batch+1)})

        self.model.train()
        self.likelihood.train()

        return val_loss/(count_batch+1)


    def train_model(self, num_epochs=100, initial_lr=0.1):
        self.initial_lr = initial_lr

        # Move model and likelihood to device
        self.model = self.model.to(self.device)
        self.likelihood = self.likelihood.to(self.device)
        self.loss_func = self.loss_func.to(self.device)

        # Training
        num_batches = len(self.train_dataloader)
        loss_records = np.zeros((num_epochs, num_batches))
        val_loss_records = [100., 99., 98., 97., 96.]

        self.model.train()
        self.likelihood.train()
        self.optimizer = torch.optim.Adam(
            list(self.model.parameters()) + list(self.likelihood.parameters()),
            lr=self.initial_lr, amsgrad=True)

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.6, patience=3, verbose='deprecated',
            threshold=1e-3, threshold_mode='rel', cooldown=1, min_lr=1e-4
        )

        for count_epoch in range(num_epochs):
            progress_bar = tqdm(enumerate(self.train_dataloader), unit='batch', total=num_batches)
            for batch, (interaction_situation, current_spacing) in progress_bar:
                progress_bar.set_description(f'Epoch {count_epoch}')

                output = self.model(interaction_situation.to(self.device))
                loss = -self.loss_func(output, current_spacing.squeeze().to(self.device))
                loss_records[count_epoch, batch] = loss.item()
                progress_bar.set_postfix({'lr=': self.optimizer.param_groups[0]['lr'], 'loss=': loss.item()})

                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

            val_loss = self.val_loop()
            self.scheduler.step(val_loss)
            val_loss_records.append(val_loss)

            # Save model every epoch
            torch.save(self.model.state_dict(), self.path_save+f'model_{count_epoch+1}epoch.pth')
            torch.save(self.likelihood.state_dict(), self.path_save+f'likelihood_{count_epoch+1}epoch.pth')

            if np.all(abs(np.diff(np.array(val_loss_records)[-5:]))<5e-4):
                # early stopping if validation loss converges
                warnings.warn('Validation loss converges and training stops at Epoch '+str(count_epoch))
                break

        # Save loss records
        loss_records = pd.DataFrame(loss_records, columns=[f'batch_{i}' for i in range(num_batches)])
        loss_records.to_csv(self.path_save+'loss_records.csv', index=False)


    def test_model(self,):
        # Move model and likelihood to device
        self.model = self.model.to(self.device)
        self.likelihood = self.likelihood.to(self.device)
        self.loss_func = self.loss_func.to(self.device)

        # Read trained model list
        self.model_list = sorted(glob.glob(self.path_save+'likelihood*.pth'), key=lambda x: int(x.split('_')[-1].split('epoch')[0]))
        self.evaluation = pd.DataFrame(np.zeros((len(self.model_list),6)),columns=['train_loss','val_loss','test_loss','train_nll','val_nll','test_nll'])
        self.evaluation['epoch'] = [int(x.split('_')[-1].split('epoch')[0]) for x in self.model_list]
        self.evaluation = self.evaluation.set_index('epoch')

        progress_bar = tqdm(zip(self.evaluation.index.values, self.model_list), total=len(self.model_list))
        for count_epoch, file_epoch in progress_bar:

            # Load trained parameters
            self.model.load_state_dict(torch.load(file_epoch.replace('likelihood','model'), map_location=torch.device(self.device)))
            self.likelihood.load_state_dict(torch.load(file_epoch, map_location=torch.device(self.device)))
            progress_bar.set_description(f'Epoch {count_epoch}')

            self.model.eval()
            self.likelihood.eval()
            # Evaluate the model
            for data_loader, setname in zip([self.train_dataloader, self.val_dataloader, self.test_dataloader], ['train','val','test']):
                num_batches = len(data_loader)
                with torch.no_grad(), gpytorch.settings.fast_pred_var():
                    for interaction_situation, current_spacing in tqdm(data_loader, desc=setname, total=num_batches):
                        f_dist = self.model(interaction_situation.to(self.device))
                        loss = -self.loss_func(f_dist, current_spacing.squeeze().to(self.device))
                        self.evaluation.loc[count_epoch,setname+'_loss'] += loss.item()
                        y_dist = self.likelihood(f_dist)
                        univariate_normal = torch.distributions.Normal(y_dist.mean, y_dist.variance.sqrt()) # args=[loc,scale]
                        nll = -univariate_normal.log_prob(current_spacing.squeeze().to(self.device)).sum()
                        self.evaluation.loc[count_epoch,setname+'_nll'] += nll.item()
                self.evaluation.loc[count_epoch,setname+'_loss'] /= num_batches
                self.evaluation.loc[count_epoch,setname+'_nll'] /= num_batches
                progress_bar.set_postfix({f'loss=': self.evaluation.loc[count_epoch,setname+'_loss']})

            self.evaluation.to_csv(self.path_save+'evaluation.csv', index=True)

