# Code for "A Unified Theory and Statistical Learning Approach for Traffic Conflict Detection"

This study is in submission. A preprint is available at [arXiv](https://arxiv.org/abs/2407.10959). __Questions, suggestions, and collaboration are welcome. Please feel free to reach out via email or GitHub Issues__.

## Access to dynamic figures
Dynamic visualisations in this paper are saved in the folder [`./Data/DynamicFigures/`](Data/DynamicFigures/). Below we present the example in Figure 9 of a conflict where the ego (red) vehicle changes lane twice continuously and has a potential collision with the target (blue) vehicle in the intermediate lane.

<p align="center">
  <img src="Data/DynamicFigures/Figure9/Figure9.gif" alt="animated" width="75%" height="75%"/>
</p>

## Abstract
This study proposes a unified theory and statistical learning approach for traffic conflict detection, addressing the long-existing call for a consistent and comprehensive methodology to evaluate the collision risk emerging in road user interactions. The proposed theory assumes context-dependent probabilistic collision risk and frames conflict detection as assessing this risk by statistical learning of extreme events in daily interactions. Experiments using real-world trajectory data are conducted for demonstration. Firstly, a unified metric for indicating conflicts is trained with lane-changing interactions on German highways. This metric and other existing metrics are then applied to near-crash events from the 100-Car Naturalistic Driving Study in the U.S. for performance comparison. Results of the experiments show that the trained metric provides effective collision warnings,  generalises across distinct datasets and traffic environments, covers a broad range of conflict types, and delivers a long-tailed distribution of conflict intensity. Reflecting on these results, the proposed theory ensures consistent evaluation by a generic formulation that encompasses varying assumptions of traffic conflicts; the statistical learning approach then enables a comprehensive consideration of influencing factors such as motion states of road users, environment conditions, and participant characteristics. Therefore, the theory and learning approach jointly provide an explainable and adaptable methodology for conflict detection among different road users and across various interaction scenarios. This promises to reduce accidents and improve overall traffic safety, by enhanced safety assessment of traffic infrastructures, more effective collision warning systems for autonomous driving, and a deeper understanding of road user behaviour in different traffic conditions.

## Dependencies 
**Thanks to @Jeffrey-Lim for providing the solution with Rye.**

To reuse the methods in general, with any version of python that satisfies the requirements of `gpytorch` version 1.11, the existing code should work without any modification. Minor modifications may be needed for other dependencies.

To reproduce the results in the same python environment, please ensure [Rye](https://rye-up.com/guide/installation/) is installed on your machine and run:

```bash
rye sync
```

Alternatively, you may manually install `gpytorch=1.11` and `pytables` with Python 3.10+. For an encapsulated environment, you may create a virtual environment with Python 3.12.4 and install the dependencies from `requirements-dev.lock` using `pip`:

```bash
pip install -r requirements-dev.lock
```

## In order to reuse the trained model:
- __Step 0 Getting familiar with use__
    - Go the directory `./DirectReuse/` and use the jupyter notebook `test.ipynb` to familiarise with the functions. The notebook provides test examples for both conflict probability estimation and conflict intensity evaluation.
- __Step 1 Model preparation__
    - Copy the files in the directory `./DirectReuse/`, except for `test.ipynb`, to where you want to reuse the model.
- __Step 2 Use it in oneline__
    - Now the model is ready to use after `from unified_conflit_detection import *`, with one single function `assess_conflict(states,*args)`! The function takes the states of two potentially interacting vehicles as input and returns either 1) conflict probability at a certain intensity level or 2) maximum possible conflict intensity.

**Note:* the trained model currently considers limited information for methodology demonstration purposes. You may consider training a more sophisticated model, or following up with the authors for future updates.

## In order to repeat the experiments:
- __Step 0 Data preparation__
    - Step 0.1 Apply for and download the highD dataset from [the website](https://levelxdata.com/highd-dataset/) and put the data in the folder `./Data/RawData/highD/`.
    - Step 0.2 We have processed and saved the 100Car NDS data in the folder `./Data/RawData/100Car/`. If interested, the readers are welcome to explore original 100Car NDS data with the code in the [repository](https://github.com/Yiru-Jiao/Reconstruct100CarNDSData).

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
      title={A unified theory and statistical learning approach for traffic conflict detection}, 
      author={Jiao, Yiru and Calvert, Simeon C. and van Cranenburgh, Sander and van Lint, Hans},
      year={2024},
      eprint={2407.10959},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
}
````
