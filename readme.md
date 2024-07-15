# Code for "A Unified Theory and Statistical Learning Approach for Traffic Conflict Detection"
This study is being submitted to a journal. A preprint is available at [arXiv](https://arxiv.org/abs/).

## Access to dynamic figures
Dynamic visualisations in this paper are saved in the folder [`./Data/DynamicFigures/`](Data/DynamicFigures/). Below we present the example in Figure 9 of a conflict where the ego (red) vehicle changes lane twice continuously and has a potential collision with the target (blue) vehicle in the intermediate lane.

<img src="Data/DynamicFigures/Figure9/Figure9.gif" width="50%" height="50%"/>

## Abstract
This study proposes a unified theory and statistical learning approach for traffic conflict detection, addressing the long-existing call for a consistent and comprehensive methodology to evaluate the collision risk emerged in road user interactions. The proposed theory assumes a context-dependent probabilistic collision risk and frames conflict detection as estimating the risk by statistical learning from observed proximities and contextual variables. Three primary tasks are integrated: representing interaction context from selected observables, inferring proximity distributions in different contexts, and applying extreme value theory to relate conflict intensity with conflict probability. As a result, this methodology is adaptable to various road users and interaction scenarios, enhancing its applicability without the need for pre-labelled conflict data. Demonstration experiments are executed using real-world trajectory data, with the unified metric trained on lane-changing interactions on German highways and applied to near-crash events from the 100-Car Naturalistic Driving Study in the U.S. The experiments demonstrate the methodology's ability to provide effective collision warnings,  generalise across different datasets and traffic environments, cover a broad range of conflicts, and deliver a long-tailed distribution of conflict intensity. This study contributes to traffic safety by offering a consistent and explainable methodology for conflict detection applicable across various scenarios. Its societal implications include enhanced safety evaluations of traffic infrastructures, more effective collision warning systems for autonomous and driving assistance systems, and a deeper understanding of road user behaviour in different traffic conditions, contributing to a potential reduction in accident rates and improving overall traffic safety.

## Package requirements
`numpy`, `pandas`, `tqdm`, `scipy`, `joblib`, `pytorch`, `gpytorch=1.11`, `opencv`, `imageio`, `matplotlib`, `jupyter notebook`

## In order to repeat the experiments:
- __Step 0 Data preparation__
    - Step 0.1 Apply for and download the highD dataset from [the website](https://levelxdata.com/highd-dataset/) and put the data in the folder `./Data/RawData/highD/`.
    - Step 0.2 We have processed and saved the 100Car NDS data in the folder `./Data/ProcessedData/100Car/`. The readers are still encouraged to explore the raw data with the code in the [repository](https://github.com/Yiru-Jiao/Reconstruct100CarNDSData) if interested.

- __Step 1 Data processing__
    - Step 1.1 Run `./DataProcessing/preprocessing_highD.py` to add heading directions of vehicles.
    - Step 1.2 Run `./DataProcessing/extraction_highD_LC.py` to extract trajectories involving lane-changing interactions in the highD dataset.
    - Step 1.3 Run `./DataProcessing/matching_100Car.py` to select applicable crashes and near-crashes in the 100Car NDS data.

- __Step 2 Gaussian Process Regression model training__
    - Step 1.1 Run `./GaussianProcessingRegression/model_training.py` to train and test the Gaussian Process Regression models.
    - Step 1.2 Use `./GaussianProcessingRegression/model_evaluation.ipynb` to visually evaluate the trained models.

- __Step 3 Demonstration part I: conflict probability estimation__
    - Step 3.1 Run `./Demonstration/ConflictProbability/pre_computation.py` to pre-compute the parameters of proximity distribution for each time moment in the near-crashes.
    - Step 3.2 Run `./Demonstration/ConflictProbability/collision_warning.py` to perform collision warnings with PSD, DRAC, TTC, and the proposed unified metric, under a variety of thresholds.
    - Step 3.3 Use `./Demonstration/ConflictProbability/result_analysis.ipynb` to find the optimal threshold for each metric.
    - Step 3.4 Run `./Demonstration/ConflictProbability/optimal_warning.py` to perform collision warnings with the optimal thresholds.
    - Step 3.5 Use `./Demonstration/ConflictProbability/result_analysis.ipynb` again to analyse the results together.

- __Step 4 Demonstration part II: conflict intensity evaluation__
    - Step 4.1 Run `./Demonstration/IntensityEvaluation/lane_change_selection.py` to evaluate the conflict intensity of lane-changing interactions in highD data.
    - Step 4.2 Use `./Demonstration/IntensityEvaluation/result_analysis.ipynb` to analyse the results.

- __Step 5 Dynamic visualisation of examples__
    - Step 5.1 Run `./Visualisation/video_recoder_100Car.py` to produce images for the dynamic visualisation of near-crashes that have at least one metric failed to indicate a conflict.
    - Step 5.2 Run `./Visualisation/video_recoder_highD_LC.py` to produce images for the dynamic visualisation of some lane-changing interactions in highD data.
    - Step 5.3 Run `./Visualisation/video_maker.py` to generate videos from the images.
    - Step 5.4 Run `./Visualisation/gif_maker.py` to generate gifs from the images (1/4 of the original resolution).

## Citation
````latex
@misc{jiao2024,
  title = {A unified theory and statistical learning approach for traffic conflict detection},
  author = {Jiao,  Yiru and Calvert,  Simeon C. and van Cranenburgh,  Sander and van Lint,  Hans},
  year = {2024},
  eprint={},
  doi = {},
  archivePrefix = {arXiv},  
}
````
