# Code for "A Unified Probabilistic Approach to Traffic Conflict Detection"

This study is being published in Analytic Methods in Accident Research and accessible via https://doi.org/10.1016/j.amar.2024.100369. Note that I'm still negotiating with the journal to have paper proofing with LaTeX to ensure correct equations. For your convenience, an unofficially formatted version (with vector graphics) is available at [arXiv](https://arxiv.org/abs/2407.10959). __Questions, suggestions, and collaboration are welcome. Please feel free to reach out via email or GitHub Issues__.

## Access to dynamic figures
Dynamic visualisations in this paper are saved in the folder [`./Data/DynamicFigures/`](Data/DynamicFigures/). Below we present the example in Figure 9 of a conflict where the ego (red) vehicle changes lane twice continuously and has a potential collision with the target (blue) vehicle in the intermediate lane.

<p align="center">
  <img src="Data/DynamicFigures/Figure9/Figure9.gif" alt="animated" width="75%" height="75%"/>
</p>

## Abstract
Traffic conflict detection is essential for proactive road safety by identifying potential collisions before they occur. Existing methods rely on surrogate safety measures tailored to specific interactions (e.g., car-following, side-swiping, or path-crossing) and require varying thresholds in different traffic conditions. This variation leads to inconsistencies and limited adaptability of conflict detection in evolving traffic environments. Consequently, a need persists for consistent detection of traffic conflicts across interaction contexts. To address this need, this study proposes a unified probabilistic approach. The proposed approach establishes a unified framework of traffic conflict detection, where traffic conflicts are formulated as context-dependent extreme events of road user interactions. The detection of conflicts is then decomposed into a series of statistical learning tasks: representing interaction contexts, inferring proximity distributions, and assessing extreme collision risk. The unified formulation accommodates diverse hypotheses of traffic conflicts and the learning tasks enable data-driven analysis of factors such as motion states of road users, environment conditions, and participant characteristics. Jointly, this approach supports consistent and comprehensive evaluation of the collision risk emerging in road user interactions. Our experiments using real-world trajectory data show that the approach provides effective collision warnings, generalises across distinct datasets and traffic environments, covers a broad range of conflict types, and captures a long-tailed distribution of conflict intensity. The findings highlight its potential to enhance the safety assessment of traffic infrastructures and policies, improve collision warning systems for autonomous driving, and deepen the understanding of road user behaviour in safety-critical interactions.

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
@article{jiao2024unified,
      title={A Unified Probabilistic Approach to Traffic Conflict Detection}, 
      author={Yiru Jiao and Simeon C. Calvert and Sander {van Cranenburgh} and Hans {van Lint}},
      year={2025},
      journal={Analytic Methods in Accident Research},
      volume={45},
      pages={100369}
}
````
