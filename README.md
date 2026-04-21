# OSTREA - One Script To Rule 'Em All

---

## Overview

This is a simple script made to quickly run and test many reinforcement learning algorithms in (almost all) the environments provided by the [gymnasium library](https://gymnasium.farama.org/index.html) with just one command.

Right now only the most popular model-free algorithms are included but other ones will probably be added in the future.

Some implementations have few little differences if compared to the standard ones, check the comments in the code for more details.

The main goal of this script is to have a easy way to find out what solution a particular algorithm would find in each environment and how the result would be affected if something changes in the internal logic, for this reason all the implementations are made to be as readable and hackable as possible 

Even if formally in reinforcement learning V and Q are two different functions here all value functions are called just "value" for readability

---

## Project structure

The main file is `ostrea.py`, it creates the environment and the agent according to the user's choices and it contains the training loop

Every algo is implemented in its dedicated "algo\_agent.py" python file, which can also be run independently with `$ python *_agent.py ` to run the algorithm on random data for testing and debugging purposes

The same training loop is used for all the algorithms, they collect data from the environment for a fixed amount of timesteps to fill a buffer, then they call their own update function to consume that data and empty the buffer.

Algorithms with replay memory like dql will add all the content of the buffer to their internal memory before cleaning it.

Changing the buffer size will also change the frequency of calls to the update function

The agent is periodically tested during training and when it achieves a new best score its inference network (value net for value based methods or the policy net for policy based methods) is saved in the `model.pt` file

---

## Supported algorithms

| algorithm | continuous actions | discrete actions |
| :------------ | :------------- | :------------- |
| dql | &cross; |  &check;|
| ddpg| &check; |  &cross;|
| vpg | &check; |  &check;|
| ppo | &check; |  &check;|
| sac | &check; |  &check;|

---

## Supported environments 

* cartpole
* lander
* lander\_continuous
* cheetah
* humanoid
* ant
* walker
* bipedal 
* bipedal\_hardcore
* acrobot
* reacher
* mountaincar\_continuous
* mountaincar
* pendulum
* pusher
* hopper
* humanoid\_standup
* inverted\_d\_pendulum
* inverted\_pendulum
* swimmer

---

## Features

* support to both continuous and discrete environments when possible
* vectorized environments
* gpu computing (used by default during training if available)
* simple automatic logging
* save/load checkpoints

---

## Usage 

#### Setup

This script uses torch and gymnasium as external dependencies
* install torch following the instructions on the official website 
* install gymnasium with `$ pip install "gymnasium[all]"`

#### Main flags

| flags and params   | description                 | value                           |
|:------------------:|-----------------------------|---------------------------------|
|`-d`                | run without writing any file|                                 |
|`-h`                | print help message          |                                 |
|`-l`                | list algorithms and envs    |                                 |
|`-r`                | record episodes during test |                                 |
|`-a ALGO`           | choose the algorithm        | one of the supported algorithms |
|`-e ENV`            | choose the environment      | one of the supported envs       |
|`-c CHECKPOINTFILE` | load checkpoint/model       | path to checkpoint/model file   |
|`--test N`          | run the script in test mode | numbers of episodes to run      |
|`--notes NOTE`      | add a note to the summary   | string to be used as a note     |

#### Train a model

To start a new training session run the script specifying the algorithm and the environment 

For example to train ppo with the discrete lunar lander environment run:

`$ python ostrea.py -a ppo -e lander`

The script will periodically save a full training checkpoint containing everything needed to resume the session (all the nets, optimizers state etc) and also a model file with only the final network needed for inference, the inference model will be saved only if better than the previous one 

_NOTE: when loading existing models the script will use the network architectures defined in the chosen algorithm's source file, make sure they are the same_

#### Resume training

To resume a stopped training session run the script specifying the algorithm, the environment and the checkpoint to load

For example to resume a session started with the previous command run:

`$ python ostrea.py -a ppo -e lander -c path/to/ckpt.pt`

_NOTE: when loading existing models the script will use the network architectures defined in the chosen algorithm's source file, make sure they are the same_

#### Test a model

To test a trained model with N espisodes run the script in test mode with `--test N` and specify the model, the algorithm used to train it and the environment

For example to test the model trained using ppo in the discrete lunar lander environment for 10 episodes run:

`$ python ostrea.py --test 10 -a ppo -e lander -c path/to/model.pt`

Add also `-r` to export videos of the episodes

_NOTE: when loading existing models the script will use the network architectures defined in the chosen algorithm's source file, make sure they are the same_

#### Hyperparameters tuning

Each algorithm's source file defines its own set of hyperameters to be used like learning rates etc. 

Everything in there can be changed and new parameters can be added directly in the `Params()` call, they will immediately become accessible as attributes of the params object

For example to add a parameter called NEWPARAM with a value of 69:

```
params = Params(...
               NEWPARAM = 69,
               ...)

print(params.NEWPARAM) 
```
For more details check `parameters.py`

Environment specific parameters (like gravity for lunar lander etc.) can be edited in the environments table in `environments.py`

---

## Results 

#### here are some of the results i got using this script (use a browser)

bipedal walker (hardcore) solved using sac

https://github.com/user-attachments/assets/c0bc7218-f38f-4bed-94fd-ff4ef20c44cf

cartpole solved using vpg

https://github.com/user-attachments/assets/2492cb2f-cde4-4ca3-8d8a-756f30b626c7

cheetah solved using sac

https://github.com/user-attachments/assets/84db1ad6-40b7-40f0-903f-1b34b0613303

lander (discrete) solved using dql with epsilon decay

https://github.com/user-attachments/assets/9240fd15-ce5f-447f-a9d0-6a893673b9b8

lander (continuous) solved using ddpg

https://github.com/user-attachments/assets/dd4ecafd-5659-41a3-b3e4-15eacba30029

lander (continuous) solved using ppo (with state dependent covariance matrix)

https://github.com/user-attachments/assets/12387291-f384-4102-8980-9eebee6e8b04

humanoid solved using ppo 

https://github.com/user-attachments/assets/ee6561d8-0a50-4235-b0ef-9b8b313188c2

#### honorable mentions

proximal policy optimization turned into pirate policy optimization

https://github.com/user-attachments/assets/6333ed50-6205-4d4c-af57-6dd6caef50aa

ppo agent found the optimal motion of a cheetah (having a stroke)

https://github.com/user-attachments/assets/7fa3e207-b9eb-4e0b-adb1-cfe01ce85b5f

sac agent accidentally discovered parkour

https://github.com/user-attachments/assets/9226ea7c-aeaf-4343-a3eb-8d65e8c48750

---

## Resources

1. https://incompleteideas.net/book/RLbook2020.pdf - Sutton & Barto *(the bible of rl)*
2. https://arxiv.org/pdf/1801.01290 - Soft Actor Critic 1
3. https://arxiv.org/pdf/1812.05905 - Soft Actor Critic 2
4. https://arxiv.org/pdf/1707.06347 - Proximal Policy Optimization
5. https://arxiv.org/pdf/1506.02438 - Generalized Advantage Estimation
6. https://proceedings.neurips.cc/paper_files/paper/1999/file/464d828b85b0bed98e80ade0a5c43b0f-Paper.pdf - Policy Gradient Theorem (VPG)
7. https://arxiv.org/pdf/1312.5602 - Deep Q Network 1
8. https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf - Deep Q Network 2
9. https://arxiv.org/pdf/1509.02971 - DDPG
