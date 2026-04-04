import torch
import random
import torch.nn as nn
import gymnasium
from collections import deque
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.categorical import Categorical
from parameters import Params

params = Params(

                # environment/general training parameters 

                SEED = 69420,                    # seed used with torch
                MAX_TRAINING_STEPS = 100e6,      # 100M
                BUFFER_SIZE = 100,               # size of episode buffer that triggers the update
                PRINT_FREQ_STEPS = 100,          # after how many steps the logs should be printed during training       
                UPDATE_PLOT_SAVE_FREQ = 100,     # after how many updates the avg return plot should be saved 
                N_EVAL_EPISODES = 10,            # how many episodes should be used for evaluation during training 
                GAMMA = 0.99,
                N_ENV = 64,
                DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),

                # agent parameters 

                WARMUP = 10000,                  # warmup steps without any update
                MODEL_NAME_POL = "policy.pt",
                MODEL_NAME_VAL = "value_net.pt",
                MEMORY_MAXLEN = 1_000_000,
                MEMORY_BATCH_SIZE = 256,
                GRADIENT_STEPS = 100,            # how many gradient steps should be done in the update function, same value as buffer size to have a updates/data ratio close to 1
                NUMERICAL_EPSILON = 1e-6,        # small value for numerical stabilty
                TARGET_H = 0.1,                  # target entropy (only for discrete action spaces, computed automatically in continuous case)
                VALUE_LR =  3e-4,
                POLICY_LR = 3e-4,
                ALPHA_LR =  1e-3,
                TAU = 0.01,                      # soft update parameter for target nets
                MAX_LOGVAR = 2.,
                MIN_LOGVAR = -20.,
                POLICY_METHOD = True,
                ALGO_NAME = "sac"

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


class SACAgent:

    """
    implementation of a reinforcement learning agent that uses SAC algorithm

    this implementation can be used in environments with both 
    continuous and discrete action spaces

    with continuous actions the policy network will compute 
    the parameters (means and cov matrix) of the n-dimensional normal 
    distribution the actions will be sampled from

    with discrete actions the policy network will predict the logits
    (unnormalized scores) that will be used with categorical distribution
    to compute the probability of each one of the possible n actions 

    this implementation currently uses only one value network 
    (and its "target version" for a total of 3 networks)

    the temperature parameter will be learned but reinitialized in 
    each new training session

    target entropy is used as hyperparameter for discrete actions only
    and it is automatically set to -|A| for continuous actions

    this implementation assumes actions in the range [-1,1]
    
    resources:
    https://arxiv.org/pdf/1801.01290
    https://arxiv.org/pdf/1812.05905
    """

    def __init__(self, parameters):

        self.gamma = parameters.GAMMA
        self.value_lr =  parameters.VALUE_LR
        self.policy_lr = parameters.POLICY_LR
        self.alpha_lr = parameters.ALPHA_LR
        self.memory_maxlen = parameters.MEMORY_MAXLEN
        self.memory_batch_size = parameters.MEMORY_BATCH_SIZE
        self.gradient_steps = parameters.GRADIENT_STEPS 
        self.warmup = parameters.WARMUP
        self.tau = parameters.TAU
        self.device = parameters.DEVICE
        self.numerical_epsilon = parameters.NUMERICAL_EPSILON
        self.max_logvar = parameters.MAX_LOGVAR
        self.min_logvar = parameters.MIN_LOGVAR

        self.obs_size = parameters.obs_size
        self.action_space_dim = parameters.action_space_dim
        self.continuous_actions = parameters.env_is_continuous
        self.value_model_checkpoint = parameters.value_model_checkpoint
        self.policy_model_checkpoint = parameters.policy_model_checkpoint

        self.log_alpha = torch.nn.Parameter(torch.rand(1).to(self.device))

        if self.continuous_actions:
            policy_net_output_dim = 2*self.action_space_dim 
            q_net_input_dim = self.obs_size + self.action_space_dim
            q_net_output_dim = 1
            self.target_h = -self.action_space_dim
        else:
            policy_net_output_dim = self.action_space_dim
            q_net_input_dim = self.obs_size
            q_net_output_dim = self.action_space_dim
            self.target_h = parameters.TARGET_H

        self.buffer = []
        self.tot_steps = 0

        self.memory = ReplayMemory(maxlen=self.memory_maxlen)

        self.policy_net = nn.Sequential(
          nn.Linear(self.obs_size, 100),
          nn.Tanh(),
          nn.Linear(100, 100),
          nn.Tanh(),
          nn.Linear(100, policy_net_output_dim)).to(self.device)

        self.q_net = nn.Sequential(
          nn.Linear(q_net_input_dim , 100),
          nn.LeakyReLU(),
          nn.Linear(100, 100),
          nn.LeakyReLU(),
          nn.Linear(100, q_net_output_dim)).to(self.device) 

        self.target_q_net = nn.Sequential(
          nn.Linear(q_net_input_dim , 100),
          nn.LeakyReLU(),
          nn.Linear(100, 100),
          nn.LeakyReLU(),
          nn.Linear(100, q_net_output_dim)).to(self.device) 

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
                self.q_net.load_state_dict(torch.load(self.value_model_checkpoint, map_location = self.device))
                print(f"value checkpoint loaded")
            except Exception as e:
                print(f"cant load value weights: \n {e} \n")
        else:
            print("training a new value net")

        self.target_q_net.load_state_dict(self.q_net.state_dict())

        # because main script use value as name TODO change this
        self.value_net = self.q_net 

        self.optim_policy = torch.optim.Adam(self.policy_net.parameters(),
                          lr = self.policy_lr)

        self.optim_value = torch.optim.Adam(self.q_net.parameters(),
                          lr = self.value_lr)

        self.optim_temp = torch.optim.Adam([self.log_alpha],
                          lr = self.alpha_lr)

        self.mse = torch.nn.MSELoss()


    def choose_action_greedy(self, obs):

        with torch.no_grad():

            if self.continuous_actions:

                # get only the means as actions 

                policy_net_out = self.policy_net(obs)
                means = policy_net_out[:self.action_space_dim]
                action = torch.tanh(means)

            else:

                # run the policy to get logits

                logits  = self.policy_net(obs)
                probs_distribution = Categorical(logits=logits)
                action = probs_distribution.probs.argmax()

        return action


    def choose_action(self, obs):

        self.tot_steps += 1

        # sample from the actual distribution generated by the net
        
        with torch.no_grad():

            if self.continuous_actions:

                # run the policy to get means and covariances
                policy_net_out = self.policy_net(obs)

                # create (unbounded) probability distribution  
                means = policy_net_out[:,:self.action_space_dim]
                log_var = policy_net_out[:,self.action_space_dim:]
                var = torch.exp(torch.clamp(log_var, min = self.min_logvar, max = self.max_logvar))
                cov = torch.diag_embed(var)
                probs = MultivariateNormal(means,cov)

                # sample from prob distribution and apply tanh
                action = probs.sample()
                action = torch.tanh(action)

            else:

                # run the policy to get logits
                logits  = self.policy_net(obs)

                # create the discrete distribution
                probs_distribution = Categorical(logits=logits)

                # sample from prob distribution
                action = probs_distribution.sample()

        return (action,None)


    def update_memory(self):
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
            actions = torch.stack([t[1] for t in batch]).flatten(0,1) # bounded
            rewards = torch.stack([t[2] for t in batch]).flatten(0,1)
            next_states = torch.stack([t[3] for t in batch]).flatten(0,1)
            dones = torch.stack([t[4] for t in batch]).flatten(0,1)

            # update soft q function 

            if self.continuous_actions:

                # with continuous actions to compute the target it is necessary
                # to evaluate Q on sampled next actions
                # Q(S_t,A_t) <- R_t+1 + not_done*gamma*Q(S_t+1,A_t+1) - alpha*log(pi(S_t+1,A_t+1))

                with torch.no_grad():

                    # sample next actions using the target policy in states
                    # collected by behavior policy (without reparametrization 
                    # trick cause gradient is not needed here)

                    policy_net_out = self.policy_net(next_states)

                    next_means = policy_net_out[:,:self.action_space_dim]
                    log_var = policy_net_out[:,self.action_space_dim:]

                    var = torch.exp(torch.clamp(log_var, min = self.min_logvar, max = self.max_logvar))
                    cov = torch.diag_embed(var)
                    next_probs_dist = MultivariateNormal(next_means,cov)

                    next_actions_unbounded = next_probs_dist.sample()
                    next_action_log_probs = next_probs_dist.log_prob(next_actions_unbounded)

                    next_actions = torch.tanh(next_actions_unbounded)

                    # probability correction because of tanh
                    next_action_log_probs -= torch.log(1-torch.tanh(next_actions_unbounded)**2+self.numerical_epsilon).sum(-1)

                    next_s_a_pairs = torch.concat((next_states, next_actions),dim=-1)
                    targets = rewards + self.gamma*(1-dones.to(int))*(self.target_q_net(next_s_a_pairs).squeeze() - torch.exp(self.log_alpha.detach()) * next_action_log_probs)
                    targets = targets.to(torch.float32) 

                s_a_pairs = torch.concat((states,actions),dim=-1)
                Q_s_a = self.q_net(s_a_pairs).squeeze(-1)

            else:
                
                # with discrete action space there is no need to apply tanh
                # and also entropy and value of the next state can be computed 
                # exactly using all actions instead of sampling
                # Q(S_t,A_t) <- R_t+1 + not_done*gamma*(sum_a(pi(S_t+1,A_t+1)*Q(S_t+1,A_t+1)) + alpha*entropy(pi(S_t+1,A_t+1)))
                
                with torch.no_grad():

                    logits  = self.policy_net(next_states)

                    next_probs_dist = Categorical(logits=logits)
                    entropy = next_probs_dist.entropy()
                    probs = next_probs_dist.probs
                    next_values = (probs*(self.target_q_net(next_states))).sum(-1)
                    targets = rewards + self.gamma*(1-dones.to(int))*(next_values + torch.exp(self.log_alpha.detach())*entropy)
                    targets = targets.to(torch.float32) 

                Q_s = self.q_net(states)
                Q_s_a = torch.gather(Q_s,-1,actions.unsqueeze(1)).squeeze()

            loss = self.mse(targets,Q_s_a)

            self.optim_value.zero_grad()
            loss.backward()
            self.optim_value.step()

            # update policy 

            if self.continuous_actions:

                # sample actions using reparametrization trick
                
                policy_net_out = self.policy_net(states)
                means = policy_net_out[:,:self.action_space_dim]
                log_var = policy_net_out[:,self.action_space_dim:]
                log_var = torch.clamp(log_var, min = self.min_logvar, max = self.max_logvar)
                var = torch.exp(log_var)
                std = torch.exp(0.5 * log_var)
                eps = torch.randn_like(means)
                actions_unbounded = means + (std * eps)
                actions = torch.tanh(actions_unbounded)

                s_a_pairs = torch.concat((states,actions),dim=-1)
                Q_s_a = self.q_net(s_a_pairs).squeeze(-1)

                cov = torch.diag_embed(var)
                distributions = MultivariateNormal(means,cov)
                logprobs = distributions.log_prob(actions_unbounded)
                logprobs -= torch.log(1-torch.tanh(actions_unbounded)**2+self.numerical_epsilon).sum(-1)

                sac_objective = (torch.exp(self.log_alpha.detach()) * logprobs - Q_s_a).mean()
            
            else:

                # with discrete action space also the objective can be computed 
                # exactly considering all the actions instead of sampling one

                logits  = self.policy_net(states)
                probs_dist = Categorical(logits=logits)
                entropy = probs_dist.entropy()
                probs = probs_dist.probs
                values = (probs*(self.q_net(states))).sum(-1)
                sac_objective = -(values + torch.exp(self.log_alpha.detach())*entropy).mean()

            self.optim_policy.zero_grad()
            sac_objective.backward()
            self.optim_policy.step()

            # update alpha

            if self.continuous_actions:

                # log_alpha is used instead of alpha because the algorithm is 
                # actually learning the log itself, in this way get the 
                # right gradient 

                alpha_loss = (self.log_alpha * (-1*logprobs.detach() - self.target_h)).mean()

                self.optim_temp.zero_grad()
                alpha_loss.backward()
                self.optim_temp.step()

            else:

                # with discrete actions the entropy can be calculated
                # in each state and the errors can be averaged

                alpha_loss = (self.log_alpha * (entropy.detach() - self.target_h)).mean()

                self.optim_temp.zero_grad()
                alpha_loss.backward()
                self.optim_temp.step()

            # update target network

            for target_param, param in zip(self.target_q_net.parameters(), self.q_net.parameters()):
                target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

if __name__ == "__main__":

    print(f"using device {params.DEVICE}")

    params.value_model_checkpoint = None 
    params.policy_model_checkpoint = None
    params.env_is_continuous = False 
    params.obs_size = 2
    params.action_space_dim = 3
    params.WARMUP = 0
    params.MEMORY_BATCH_SIZE  = 5

    discrete_agent = SACAgent(params)

    print("discrete agent created")

    params.env_is_continuous = True 

    continuous_agent = SACAgent(params)

    print("continuous agent created")

    # test loop with random data, suppose a vectorized environment and 10 steps

    n_env = params.N_ENV

    state = torch.rand(n_env,2).to(params.DEVICE)

    for t in range(10):

        with torch.no_grad():
        
            discrete_actions, disc_log_prob = discrete_agent.choose_action(state)
            continuous_actions, cont_log_prob = continuous_agent.choose_action(state)

            reward = torch.rand(n_env).to(params.DEVICE)
            new_state = torch.rand(n_env,2).to(params.DEVICE)
            done = (torch.ones(n_env) if t % 10 == 0 else torch.zeros(n_env)).to(params.DEVICE)

            discrete_agent.buffer.append((state,discrete_actions,reward,new_state,done))
            continuous_agent.buffer.append((state,continuous_actions,reward,new_state,done))

            state = new_state

    discrete_agent.update()
    continuous_agent.update()

    print("update done")
