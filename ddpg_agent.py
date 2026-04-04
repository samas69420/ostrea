import torch
import random
import torch.nn as nn
import gymnasium
from collections import deque
from torch.distributions.normal import Normal
from parameters import Params

params = Params(

                # environment/general training parameters 

                SEED = None,                     # seed used with torch
                MAX_TRAINING_STEPS = 100e6,      # 100M
                BUFFER_SIZE = 100,               # size of episode buffer that triggers the update
                PRINT_FREQ_STEPS = 100,          # after how many steps the logs should be printed during training
                UPDATE_PLOT_SAVE_FREQ = 100,     # after how many updates the avg return plot should be saved 
                N_EVAL_EPISODES = 10,            # how many episodes should be used for evaluation during training
                GAMMA = 0.99,
                N_ENV = 64,
                DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),

                # agent parameters 

                WARMUP = 3000,                   # warmup steps without any update
                MODEL_NAME_POL = "policy.pt",
                MODEL_NAME_VAL = "value_net.pt",
                MEMORY_MAXLEN = 500_000,
                MEMORY_BATCH_SIZE = 256,
                GRADIENT_STEPS = 100,            # how many gradient steps should be done in the update function, same value as buffer size to have a updates/data ratio close to 1
                VALUE_LR =  1e-3,
                POLICY_LR = 1e-3,
                NOISE_MAG = 0.2,                 # noise magnitude for exploration
                TAU = 0.005,                     # soft update parameter for target nets
                POLICY_METHOD = True,
                ALGO_NAME = "ddpg"

               )


class ReplayMemory:
    
    def __init__(self, maxlen):
        
        self.buffer = deque(maxlen=maxlen)

    def __len__(self):

        return len(self.buffer)

    def append(self, element):

        self.buffer.append(element)

    def sample(self, n):

        return random.sample(self.buffer, n)


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

        self.obs_size = parameters.obs_size
        self.action_space_dim = parameters.action_space_dim
        self.value_model_checkpoint = parameters.value_model_checkpoint
        self.policy_model_checkpoint = parameters.policy_model_checkpoint

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

        if self.policy_model_checkpoint:
            try:
                self.policy_net.load_state_dict(torch.load(self.policy_model_checkpoint, map_location = self.device))
                print(f"policy checkpoint loaded")
            except Exception as e:
                print(f"cant load policy weights: \n {e} \n")
        else:
            print("training a new policy net")

        if self.value_model_checkpoint:
            try:
                self.value_net.load_state_dict(torch.load(self.value_model_checkpoint, map_location = self.device))
                print(f"value checkpoint loaded")
            except Exception as e:
                print(f"cant load value weights: \n {e} \n")
        else:
            print("training a new value net")

        self.target_value_net.load_state_dict(self.value_net.state_dict())
        self.target_policy_net.load_state_dict(self.policy_net.state_dict())

        self.optim_policy = torch.optim.Adam(self.policy_net.parameters(),
                          lr = self.policy_lr)

        self.optim_value = torch.optim.Adam(self.value_net.parameters(),
                          lr = self.value_lr)

        self.mse = torch.nn.MSELoss()


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


if __name__ == "__main__":

    print(f"using device {params.DEVICE}")

    params.value_model_checkpoint = None 
    params.policy_model_checkpoint = None
    params.env_is_continuous = True
    params.obs_size = 2
    params.action_space_dim = 3
    params.WARMUP = 0
    params.MEMORY_BATCH_SIZE  = 5

    agent = DDPGAgent(params)

    print("agent created")

    # test loop with random data, suppose a vectorized environment and 10 steps

    n_env = params.N_ENV

    state = torch.rand(n_env,2).to(params.DEVICE)

    for t in range(10):

        with torch.no_grad():
        
            actions, cont_log_prob = agent.choose_action(state)

            reward = torch.rand(n_env).to(params.DEVICE)
            new_state = torch.rand(n_env,2).to(params.DEVICE)
            done = (torch.ones(n_env) if t % 10 == 0 else torch.zeros(n_env)).to(params.DEVICE)

            agent.buffer.append((state,actions,reward,new_state,done))

            state = new_state

    agent.update()

    print("update done")

