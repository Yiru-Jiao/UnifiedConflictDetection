'''
This file is used to train, validate, and test the model for various settings.
Results are saved in the OutputData folder.

The code performs the following steps:
1. Sets the input and output paths for the data.
2. Imports necessary libraries and modules.
3. Checks if a CUDA-enabled GPU is available and sets the device accordingly.
4. Defines the training parameters such as initial learning rate, batch size, and number of quasi-epochs.
5. Loops over different values of beta and trains the model using the specified parameters.
6. Evaluates the trained model by testing it at regular intervals for different values of beta.
'''

import torch
from training_utils import *

path_input = './Data/InputData/highD_LC/'
path_output = './Data/OutputData/trained_models/highD_LC/'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')
if device=='cpu':
    num_threads = torch.get_num_threads()
    print(f'Number of available threads: {num_threads}')
    torch.set_num_threads(round(num_threads/2))


initial_lr = 0.025
batch_size = 2048
num_qepochs = 60
num_inducing_points = 100

# Training
for beta in [5, 10]:
    pipeline = train_val_test(device, num_inducing_points, path_input, path_output)
    pipeline.create_dataloader(batch_size, beta)
    print('Training...')
    pipeline.train_model(num_qepochs, initial_lr)


# Evaluation
for beta in [5, 10]:
    pipeline = train_val_test(device, num_inducing_points, path_input, path_output)
    pipeline.create_dataloader(batch_size, beta)
    print('Evaluating...')
    pipeline.test_model()
