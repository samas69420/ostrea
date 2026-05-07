import torch
import torch.nn as nn
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.categorical import Categorical
from utils.replaymemory import ReplayMemory
from utils.checkpoint import CheckpointHandler
from agents.base_agent import BaseAgent


class DQLAgent(BaseAgent):
    
    """
    implementation of a reinforcement learning agent that uses DQL algorithm

    This implementation can be used only in environments with discrete actions,
    it also uses a optional decay for the exploration parameter epsilon 

    resources:
    https://arxiv.org/pdf/1312.5602
    https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf
    """


    def __init__(self, parameters):

        if parameters.SEED:
            torch.manual_seed(parameters.SEED)

        self.device = parameters.DEVICE
        self.eps = parameters.EPSILON
        self.gamma = parameters.GAMMA
        self.value_lr = parameters.VALUE_LR
        self.memory_maxlen = parameters.MEMORY_MAXLEN
        self.memory_batch_size = parameters.MEMORY_BATCH_SIZE
        self.n_env = parameters.N_ENV
        self.min_eps = parameters.MIN_EPS
        self.use_decay = parameters.USE_DECAY
        self.eps_lin_decay = parameters.EPS_LIN_DECAY
        self.update_target_net_freq = parameters.UPDATE_TARGET_NET_FREQ
        self.gradient_steps = parameters.GRADIENT_STEPS
        self.policy_method = parameters.POLICY_METHOD

        self.obs_size = parameters.obs_size
        self.action_space_dim = parameters.action_space_dim
        self.continuous_actions = parameters.env_is_continuous
        self.checkpoint = parameters.checkpoint

        self.tot_steps = 0

        self.buffer = []
        self.memory = ReplayMemory(maxlen=self.memory_maxlen)
        
        self.value_net = nn.Sequential(
          nn.Linear(self.obs_size, 64),
          nn.LeakyReLU(),
          nn.Linear(64, 64),
          nn.LeakyReLU(),
          nn.Linear(64, self.action_space_dim)).to(self.device)

        self.target_value_net = nn.Sequential(
          nn.Linear(self.obs_size, 64),
          nn.LeakyReLU(),
          nn.Linear(64, 64),
          nn.LeakyReLU(),
          nn.Linear(64, self.action_space_dim)).to(self.device)

        self.optim_value = torch.optim.Adam(self.value_net.parameters(), lr = self.value_lr)

        self.checkpoint_handler = CheckpointHandler(self)

        if self.checkpoint:
            self.load_checkpoint(self.checkpoint, self.device)
        else:
            print("no checkpoint, training new networks")

        self.mse = torch.nn.MSELoss()


    def decay_epsilon(self):
        # linear decay
        self.eps = max(self.min_eps, self.eps-self.eps_lin_decay)


    def choose_action(self, obs):

        self.tot_steps += 1
        if self.tot_steps % self.update_target_net_freq == 0:
            self.target_value_net.load_state_dict(self.value_net.state_dict())
        
        with torch.no_grad():
            Qs = self.value_net(obs)
            max_actions = torch.argmax(Qs,dim=-1)

        random_actions = torch.randint(self.action_space_dim,(self.n_env,)).to(self.device)
        mask = (torch.rand(self.n_env) < self.eps).to(self.device)

        # exploratory actions 
        actions = random_actions.where(mask, max_actions)

        # second argument returned as "None" is just to make this function 
        # compatible with the call in main script 
        return (actions, None) 


    def choose_action_greedy(self, obs):

        with torch.no_grad():
            Qs = self.value_net(obs)
            max_actions = torch.argmax(Qs,dim=-1)

        return max_actions


    def update_memory(self):
        self.memory.buffer.extend(self.buffer)
        self.buffer = []


    def update(self):

        self.update_memory()

        for _ in range(self.gradient_steps):

            batch = self.memory.sample(self.memory_batch_size)

            states = torch.cat([e[0] for e in batch])           # (T*n_env,obs_size)
            actions = torch.cat([e[1] for e in batch])                   # (T*n_env)
            rewards = torch.cat([e[2] for e in batch]).to(torch.float32) # (T*n_env)
            next_states = torch.cat([e[3] for e in batch])      # (T*n_env,obs_size)
            dones = torch.cat([e[4] for e in batch])                     # (T*n_env)

            # update Q network using the td(0) error evaluated using greedy policy
            # and computed on a batch of data sampled from replay memory

            # Q(S_t,A_t) <- R_t+1 + not_done*gamma*max_a(Q(S_t+1,a))

            with torch.no_grad():
                targets = rewards + self.gamma*(1-dones.to(int))*torch.max(self.target_value_net(next_states),dim=-1)[0]

            Q_s = self.value_net(states)
            Q_s_a = torch.gather(Q_s,1,actions.unsqueeze(1)).squeeze()

            loss = self.mse(targets,Q_s_a)

            self.optim_value.zero_grad()
            loss.backward()
            self.optim_value.step()

        if self.use_decay:
            self.decay_epsilon()

        return loss
