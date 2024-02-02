# AI-Lite-cooling-system  

## Overview
Since AI-Lite has transitioned into a commercialized product, only the prototype code has been made open source. Additionally, we also provide a portion of real experimental data from physical clusters.

## Simulator
```python
virtual_env_generator_v1.py
virtual_env_generator_v1_2.py
virtual_env_simulation.py
virtual_env_simulation_2.py
```
The aforementioned files are used to simulate the data center environment.

## AI-Lite and Baseline Schemes
```python
gaussian_bandit.py
gaussian_ucb_agent.py
gp_ucb.py

Advanced_PG.py
Advanced_PG_test.py
Advanced_PPO.py
PG_small_action.py
PG_small_action_test.py
PPO_small.py
ppo_agent.py
double_XGBoost_regression.py
gaussian_bandit_RandomForestRegressor.py
gaussian_bandit_SVR.py
gaussian_bandit_gaussian_bandit.py
gaussian_bandit_neural_network.py
linear_contextual_bandit_agent.py
xgboost_test.py
```
The file 'gaussian_bandit.py' above serves as our core implementation for AI-Lite. Other files in this section represent baseline schemes.

## Data
The "data" folder contains experimental data from physical clusters, spanning from August 2023 to December 2023.





