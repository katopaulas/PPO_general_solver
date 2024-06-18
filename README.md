# PPO_general_solver (Tested on Cartpole/Lunarlander)


## Introduction
Welcome to a simple straighforward Reinforcement learning solution for various problems. The script provided is a basis for fast prototyping on RL tasks without any big data optimization.

## Overview

- `ppo_generic.py` : The backbone that runs the whole process. Can be fine tuned.

## Prerequisites
1. Python (version 3.6 or higher)
2. Install the required Python packages:

```bash
pip install -r requirements.txt
```
Note: After the generic Gymnasium Installation there is a possibility that some extra dependencies will be needed for each specific environment. Run the script and the gymnasium API will propose the missing installations.

## Parameters/Hyperparameters

1. Parameters

	- `ENVIRONMENT_NAME`: The custom or AIGYM environment. (For custom environments register is recommended)

2. Hyperparameters
	- `BS`: The batch size for the training step. (Recommend > 512)
	- `MAX_REPLAY_BUFFER`: The size of the agent memory. 
	- `HIDDEN_STATE_SZ`: Neuron size for the hidden layers.
	- `EPISODES` : Episodes or epochs to run the environment.
	- `LEARNING_RATE`: Pytorch optimizer learning rate.
	- `VALUE_SCALE`: Critic value scale.
	- `ENTROPY_SCALE`: Entropy scale. (higher values encourage exploration)
	- `DISCOUNT_FACTOR`: Discount factor for future rewards.
	- `TRAIN_STEPS`: Train iterations per batch of data.
	- `PPO_EPSILON`: PPO clipping espsilon.
	- `APPROX_KL_THRESHOLD_BREAK`: KL threshold to interrupt train. (above the threshold our past policy deviates a lot from the current)

## Usage

To run the script, execute the following command:

```bash
python ppo_generic.py
```