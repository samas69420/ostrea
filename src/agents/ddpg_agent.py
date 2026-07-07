import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from parameters.ddpg_params import params
from utils.checkpoint import CheckpointHandler
from utils.replaymemory import ReplayMemory
from agents.base_agent import BaseAgent
from networks.mlp import MLP


class Model:
    """
    class to manage neural networks separately, the idea is that even if
    the learning algorithm uses some approximators it should be
    approximator-agnostic and it shouldn't take care also of the internal
    details of how the approximator is structured
    """

    def __init__(self,
                 obs_size,
                 action_space_dim,
                 lr,
                 device,
                 tau):

        self.observation_is_3d_tensor = len(obs_size) == 3
        self.action_space_dim = action_space_dim
        self.device = device
        self.tau = tau

        if self.observation_is_3d_tensor:
            raise NotImplementedError("currently only vector inputs are supported")

        elif len(obs_size) == 1:
            # input is a vector
            policy_net_input_dim = obs_size[0]
            value_net_input_dim = obs_size[0] + self.action_space_dim

        self.policy_net = MLP(input_dim = policy_net_input_dim,
                              output_dim = self.action_space_dim,
                              n_layers = 4,
                              hidden_dim = 256,
                              activation_constructor = nn.LeakyReLU,
                              device = self.device)

        self.target_policy_net = MLP(input_dim = policy_net_input_dim,
                                     output_dim = self.action_space_dim,
                                     n_layers = 4,
                                     hidden_dim = 256,
                                     activation_constructor = nn.LeakyReLU,
                                     device = self.device)

        self.value_net = MLP(input_dim = value_net_input_dim,
                              output_dim = 1,
                              n_layers = 4,
                              hidden_dim = 256,
                              activation_constructor = nn.LeakyReLU,
                              device = self.device)

        self.target_value_net = MLP(input_dim = value_net_input_dim,
                                    output_dim = 1,
                                    n_layers = 4,
                                    hidden_dim = 256,
                                    activation_constructor = nn.LeakyReLU,
                                    device = self.device)

        self.target_value_net.load_state_dict(self.value_net.state_dict())
        self.target_policy_net.load_state_dict(self.policy_net.state_dict())

        self.target_value_net.require_grad = False
        self.target_policy_net.require_grad = False

        all_trainable_params = list(self.value_net.parameters()) \
                             + list(self.policy_net.parameters())

        self.optim = torch.optim.Adam(all_trainable_params,
                          lr = lr)


    def compute_action(self, obs, target_net = False):

        if not target_net:
            action = self.policy_net(obs)
        else:
            action = self.target_policy_net(obs)

        return action


    def update_parameters(self, loss_value, ddpg_objective):

        self.optim.zero_grad()

        # compute gradients for value net weights

        loss_value.backward()

        # compute gradients for (only) policy net weights

        for p in self.value_net.parameters():
            p.requires_grad = False

        ddpg_objective.backward()

        for p in self.value_net.parameters():
            p.requires_grad = True

        self.optim.step()

        # (soft) update target networks

        for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

        for target_param, param in zip(self.target_policy_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)


    def value(self, s_a_pair, target_net = False):
        if target_net:
            s_a_value = self.target_value_net(s_a_pair)
        else:
            s_a_value = self.value_net(s_a_pair)
        return s_a_value


    def __str__(self):

        result = ""

        for name, obj in self.__dict__.items():
            if isinstance(obj, torch.nn.Module):
                result += "network: " + name + '\n'
                result += str(obj) + '\n'
            if isinstance(obj, torch.optim.Optimizer):
                result += "optimizer: " + name + '\n'
                result += str(obj) + '\n'

        return result


class DDPGAgent(BaseAgent):

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
        self.lr =  parameters.LR
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

        self.buffer = []
        self.tot_steps = 0

        self.memory = ReplayMemory(maxlen=self.memory_maxlen)

        self.model = Model(self.obs_size,
                           self.action_space_dim,
                           self.lr,
                           self.device,
                           self.tau)

        self.checkpoint_handler = CheckpointHandler(self)

        if self.checkpoint:
            self.load_checkpoint(self.checkpoint, self.device)
        else:
            print("no checkpoint, training new networks")

        self.loss_fn = torch.nn.MSELoss()


    def choose_action_greedy(self, obs):

        # this should be done in a separate encode function
        obs = obs.to(torch.float32)

        # use the policy net to deterministically compute the action
        with torch.no_grad():
            action = self.model.compute_action(obs)

        return action


    def choose_action(self, obs):

        # use the policy net to compute the action and add noise for exploration

        self.tot_steps += 1

        # this should be done in a separate encode function
        obs = obs.to(torch.float32)

        noise_distribution = Normal(torch.zeros(self.action_space_dim), self.noise_mag)
        noise = noise_distribution.sample().to(self.device)

        with torch.no_grad():
            action = self.model.compute_action(obs)

        # exploratory action
        action = action + noise
        
        return (action,None)


    def update_memory(self):

        # ignore the transition if any of the training environments have
        # been truncated
        # without this check the buffer will contain transitions with:
        # S_t = last state of episode N just before truncation
        # S_t_plus_1 = first state of the episode N+1 after reset
        # this condition should be avoided since the transition is impossible

        some_episode_was_truncated = False

        for s,a,r,ns,te,tr,_ in self.buffer:
            if not some_episode_was_truncated:
                transition = (s,a,r,ns,te,_)
                self.memory.buffer.append(transition)
            if tr.any() == True:
                some_episode_was_truncated = True
            else:
                some_episode_was_truncated = False

        self.buffer = []


    def update(self):

        T = len(self.buffer)
        self.update_memory()

        if len(self.memory) < self.memory_batch_size or \
            self.tot_steps < self.warmup:
            return

        for _ in range(self.gradient_steps):

            batch = self.memory.sample(self.memory_batch_size)

            states = torch.stack([t[0] for t in batch]).flatten(0,1).to(torch.float32)
            actions = torch.stack([t[1] for t in batch]).flatten(0,1)
            rewards = torch.stack([t[2] for t in batch]).flatten(0,1)
            next_states = torch.stack([t[3] for t in batch]).flatten(0,1).to(torch.float32)
            term = torch.stack([t[4] for t in batch]).flatten(0,1)

            # update Q loss for classic sarsa update but using the target
            # networks to compute the target (ddpg is off-policy)

            # Q(S_t,A_t) <- R_t+1 + not_done*gamma*Qtarg(S_t+1,A_t+1)

            with torch.no_grad():

                dones = term.to(torch.float32)

                # sample next actions in states found while exploring

                next_actions = self.model.compute_action(next_states, target_net=True)
                next_s_a_pairs = torch.concat((next_states, next_actions),dim=-1)
                next_s_a_values  = self.model.value(next_s_a_pairs, target_net = True)
                targets = rewards + self.gamma*(1-dones)*(next_s_a_values.squeeze())
                targets = targets.to(torch.float32) 

            s_a_pairs = torch.concat((states,actions),dim=-1)
            Q_s_a = self.model.value(s_a_pairs).squeeze(-1)

            loss_value = self.loss_fn(targets,Q_s_a)

            # compute policy objective
            # the update rule here is basically the chain rule applied to the 
            # total return (estimated with Q)
    
            actions = self.model.compute_action(states)
            s_a_pairs = torch.concat((states,actions),dim=-1)
            Q_s_a = self.model.value(s_a_pairs).squeeze(-1)

            ddpg_objective = -Q_s_a.mean()

            self.model.update_parameters(loss_value, ddpg_objective)
