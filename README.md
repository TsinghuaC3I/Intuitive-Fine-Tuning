# IFT: Intuitve Fine-Tuning

## Overview

This repository contains the code for the paper "Intuitive Fine-Tuning: Towards Simplifying Alignment into a Single Process". 

The code is based on the [eric-mitchell/direct-preference-optimization](https://github.com/eric-mitchell/direct-preference-optimization) repository.

## Setup

    pip install -r requirements.txt

## Running IFT

    bash commands/run_mistral_ift.sh

## Hyperparameters

* `Temporal Residual Connection`:
    * `lambda_schedule`: The schedule mode of `lambda`. The default value is set to `null`, which means the static mode. `linear` mode is also provided for the dynamic mode.
    * `min_lambda` & `max_lambda`: The minimum value of `lambda`. The default value of both is set to 0.2, which means the static mode. If the `lambda_schedule` is set to `linear`, the `min_lambda` and `max_lambda` will be used to control the start and end value of `lambda` during training.
    * `lambda_disturb`: The disturbance distribution of `lambda`. The default value is set to `null`, which means no disturbance. `normal` mode is also provided for the disturbance distribution.
    * `disturb_std`: The standard deviation of the `lambda_disturb`. This hyperparameter is only worked when the `lambda_disturb` is not `null`.

* `Relation Propagation`: 
    * `gamma`: The decay factor of the Relation Propagation. The default value is set to 0.95.
    * `propagation_type`: The variable attribute to Relation Propagation. The default value is set to `loss`. `mask` and `logps` are also provided for the variable attribute.
    * `propagation_side`: The side of the Relation Propagation. The default value is set to `left`. `right` is also provided for the side of the Relation Propagation.
    * `propagation_norm`: The normalization mode of the Relation Propagation. The default value is set to `L1`. `L2`, `softmax` and `log` are also provided for the normalization mode.

# Citing IFT

If you find IFT useful in your research, please consider citing the following paper:

    @article{
        hua2024intuitive,
        title={Intuitive Fine-Tuning: Towards Simplifying Alignment into a Single Process},
        author={Hua, Ermo and Qi, Biqing and Zhang, Kaiyan and Yu, Yue and Ding, Ning and Lv, Xingtai and Tian, Kai and Zhou, Bowen},
        journal={arXiv preprint arXiv:2405.11870},
        year={2024}
    }