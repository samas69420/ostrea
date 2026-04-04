# OSTREA - One Script To Rule 'Em All

---

## Overview

This is a simple script made to quickly run and test many reinforcement learning algorithms in (almost all) the environments provided by the [gymnasium library](https://gymnasium.farama.org/index.html) with just one command.

Right now only the most popular model-free algorithms are included but other ones will probably be added in the future.

Some implementations have some little differences if compared to the standard ones, see the comments in the code for more details.

The main goal of this script is to have a easy way to find out what solution a particular algorithm would find in each environment and how the result would be affected if something changes in the internal logic, for this reason all the implementations are made to be as readable and hackable as possible 

Even if formally in reinforcement learning V and Q are two different functions here all value functions are called just "value" for readability

---

## Project structure

The main file is `ostrea.py`, it creates the environment and the agent according to the user's choices and it contains the training loop

Every algo is implemented in its dedicated "algo\_agent.py" python file, which can also be run independently with `$ python *_agent.py ` to run the algorithm on random data for testing and debugging reason

The same training loop is used for all the algorithms, they collect data from the environment for a fixed amount of timesteps to fill a buffer, then they call their own update function to consume that data and empty the buffer.

Algorithms with replay memory like dql will add all the content of the buffer to their internal memory before cleaning it.

Changing the buffer size will also change the frequency of calls to the update function

The agent is periodically tested during training and when it achieves a new best score since training started its networks get saved

---

## Supported algorithms

* dql
* vpg 
* ddpg
* ppo
* sac

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

---

## Features

* support to both continuous and discrete environments when possible
* vectorized environments
* gpu computing (used by default in training if available)
* simple automatic logging
* save/load networks checkpoints 

_NOTE: the networks can be saved and loaded, however at the moment any other learned parameter (like the temperature for sac) will be reinitialized_ 

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
|`-mp POLICYFILE`    | load policy network         | path to policy network          |
|`-mv VALUEFILE`     | load value network          | path to value network           |
|`--test N`          | run the script in test mode | numbers of episodes to run      |
|`--notes NOTE`      | add a note to the summary   | string to be used as a note     |

#### Train a model

To start a new training session run the script specifying the algorithm and the environment 

For example to train ppo with the discrete lunar lander environment run:

`$ python ostrea.py -a ppo -e lander`

You can also add the arguments to load existing models for policy and value networks, in this case those models will be used as the starting point for the training session

_NOTE: when loading existing models the script will use the network architectures defined in the chosen algorithm's source file, make sure they are the same_

#### Test a model

To test a existing model with N espisodes run the script in test mode with `--test N` and specify the models, the algorithm used to train them and the environment 

For example to test the models trained using ppo in the discrete lunar lander environment for 10 episodes run:

`$ python ostrea.py --test 10 -a ppo -e lander -mp path/to/policy_net.pt -mv path/to/value_net.pt`

For value based methods like dql only the value model is needed and `-mp` can be omitted

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

Changing environment specific parameters (like gravity for lunar lander etc.) can be done in the environment declaration in the main script

---

## Results 

#### here are some of the results i got using this script

bipedal walker (hardcore) solved using sac

![](media/sac_bipedal.mp4)

cartpole solved using vpg

![](media/vpg_cartpole.mp4)

cheetah solved using sac

![](media/sac_cheetah.mp4)

lander (discrete) solved using dql with epsilon decay

![](media/dql_lander.mp4)

lander (continuous) solved using ddpg

![](media/ddpg_lander_c.mp4)

lander (continuous) solved using ppo (with state dependent covariance matrix)

![](media/ppo_lander_c.mp4)

humanoid solved using ppo 

![](media/ppo_humanoid.mp4)

#### honorable mentions

proximal policy optimization turned into pirate policy optimization

![](media/arrrr.mp4)

ppo agent found the optimal motion of a cheetah (having a stroke)

![](media/cheetah_s.mp4)

sac agent accidentally discovered parkour

![](media/cheetah_p.mp4)


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
