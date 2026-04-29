import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from parameters.ddpg_params import params
from utils.checkpoint import CheckpointHandler
from utils.replaymemory import ReplayMemory


class DDPGAgent:

    """
    implementation of a reinforcement learning agent that uses DDPG algorithm

    this implementation can be used only in environments with continuous actions

    this implementation assumes actions in the range [-1,1]

    for exploration a gaussian noise is added to actions computed by the deterministic policy

    resources: 
    https://arxiv.org/pdf/1509.02971
    """


    def __init__(self, parameters):

        if not parameters.env_is_continuous:
            raise ValueError("DDPG only works for continuous action spaces")

        # extract the hardcoded values from parameters

        self.gamma = parameters.GAMMA
        self.value_lr =  parameters.VALUE_LR
        self.policy_lr = parameters.POLICY_LR
        self.memory_maxlen = parameters.MEMORY_MAXLEN
        self.memory_batch_size = parameters.MEMORY_BATCH_SIZE
        self.warmup = parameters.WARMUP
        self.tau = parameters.TAU
        self.noise_mag = parameters.NOISE_MAG
        self.device = parameters.DEVICE
        self.gradient_steps = parameters.GRADIENT_STEPS 
        self.policy_method = parameters.POLICY_METHOD

        # extract the other values added before calling the constructor

        self.obs_size = parameters.obs_size
        self.action_space_dim = parameters.action_space_dim
        self.checkpoint = parameters.checkpoint

        value_net_input_dim = self.obs_size + self.action_space_dim

        self.buffer = []
        self.tot_steps = 0

        self.memory = ReplayMemory(maxlen=self.memory_maxlen)

        self.policy_net = nn.Sequential(
          nn.Linear(self.obs_size, 100),
          nn.Tanh(),
          nn.Linear(100, 100),
          nn.Tanh(),
          nn.Linear(100, self.action_space_dim),
          nn.Tanh()).to(self.device)

        self.target_policy_net = nn.Sequential(
          nn.Linear(self.obs_size, 100),
          nn.Tanh(),
          nn.Linear(100, 100),
          nn.Tanh(),
          nn.Linear(100, self.action_space_dim),
          nn.Tanh()).to(self.device)

        self.value_net = nn.Sequential(
          nn.Linear(value_net_input_dim , 64),
          nn.LeakyReLU(),
          nn.Linear(64, 64),
          nn.LeakyReLU(),
          nn.Linear(64, 64),
          nn.LeakyReLU(),
          nn.Linear(64, 1)).to(self.device) 

        self.target_value_net = nn.Sequential(
          nn.Linear(value_net_input_dim , 64),
          nn.LeakyReLU(),
          nn.Linear(64, 64),
          nn.LeakyReLU(),
          nn.Linear(64, 64),
          nn.LeakyReLU(),
          nn.Linear(64, 1)).to(self.device) 

        self.target_value_net.load_state_dict(self.value_net.state_dict())
        self.target_policy_net.load_state_dict(self.policy_net.state_dict())

        self.optim_policy = torch.optim.Adam(self.policy_net.parameters(),
                          lr = self.policy_lr)

        self.optim_value = torch.optim.Adam(self.value_net.parameters(),
                          lr = self.value_lr)

        self.checkpoint_handler = CheckpointHandler(self)

        if self.checkpoint:
            self.load_checkpoint(self.checkpoint, self.device)
        else:
            print("no checkpoint, training new networks")

        self.mse = torch.nn.MSELoss()


    def save_checkpoint(self, checkpoint_path):
        # save the full training checkpoint
        self.checkpoint_handler.save(checkpoint_path, full=True)


    def save_model(self, checkpoint_path):
        # save only the model for inference
        self.checkpoint_handler.save(checkpoint_path, full=False)


    def load_checkpoint(self, checkpoint_path, device):
        # used for both training checkpoints and inference models
        self.checkpoint_handler.load(checkpoint_path, device)


    def choose_action_greedy(self, obs):

        # use the policy net to deterministically compute the action

        with torch.no_grad():
            action = policy_net_out = self.policy_net(obs)

        return action


    def choose_action(self, obs):

        # use the policy net to compute the action and add noise for exploration

        self.tot_steps += 1

        noise_distribution = Normal(torch.zeros(self.action_space_dim), self.noise_mag)
        noise = noise_distribution.sample().to(self.device)

        with torch.no_grad():
            action = self.policy_net(obs)

        # exploratory action
        action = action + noise
        
        return (action,None)


    def update_memory(self):

        # move all the buffer content into replay memory for later use
        self.memory.buffer.extend(self.buffer)
        self.buffer = []


    def update(self):

        T = len(self.buffer)
        self.update_memory()

        if len(self.memory) < self.memory_batch_size or \
            self.tot_steps < self.warmup:
            return

        for _ in range(self.gradient_steps):

            batch = self.memory.sample(self.memory_batch_size)

            states = torch.stack([t[0] for t in batch]).flatten(0,1)
            actions = torch.stack([t[1] for t in batch]).flatten(0,1)
            rewards = torch.stack([t[2] for t in batch]).flatten(0,1)
            next_states = torch.stack([t[3] for t in batch]).flatten(0,1)
            dones = torch.stack([t[4] for t in batch]).flatten(0,1)

            # update Q network with classic sarsa update but using the target
            # networks to compute the target (ddpg is off-policy)

            # Q(S_t,A_t) <- R_t+1 + not_done*gamma*Q(S_t+1,A_t+1)

            with torch.no_grad():

                # sample next actions in states found while exploring

                next_actions = self.target_policy_net(next_states)
                next_s_a_pairs = torch.concat((next_states, next_actions),dim=-1)
                targets = rewards + self.gamma*(1-dones.to(int))*(self.target_value_net(next_s_a_pairs).squeeze())
                targets = targets.to(torch.float32) 

            s_a_pairs = torch.concat((states,actions),dim=-1)
            Q_s_a = self.value_net(s_a_pairs).squeeze(-1)

            loss = self.mse(targets,Q_s_a)

            self.optim_value.zero_grad()
            loss.backward()
            self.optim_value.step()

            # update policy
            # the update rule here is basically the chain rule applied to the 
            # total return (estimated using Q) 
    
            actions = self.policy_net(states)
            s_a_pairs = torch.concat((states,actions),dim=-1)
            Q_s_a = self.value_net(s_a_pairs).squeeze(-1)

            ddpg_objective = -Q_s_a.mean()

            self.optim_policy.zero_grad()
            ddpg_objective.backward()
            self.optim_policy.step()

            # (soft) update target networks

            for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
                target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

            for target_param, param in zip(self.target_policy_net.parameters(), self.policy_net.parameters()):
                target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
