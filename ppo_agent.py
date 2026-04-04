import torch
import torch.nn as nn
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.categorical import Categorical
from parameters import Params

params = Params(

                # environment/general training parameters 

                SEED = None,                     # seed used with torch
                MAX_TRAINING_STEPS = 100e6,      # 100M
                BUFFER_SIZE = 1000,              # size of episode buffer that triggers the update
                PRINT_FREQ_STEPS = 1000,         # after how many steps the logs should be printed during training
                UPDATE_PLOT_SAVE_FREQ = 10,      # after how many updates the avg return plot should be saved 
                N_EVAL_EPISODES = 10,            # how many episodes should be used for evaluation during training 
                GAMMA = 0.99,
                N_ENV = 64,
                DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),

                # agent parameters 

                PPO_EPS = 5e-2,                  # threshold for the ratio function
                SEPARATE_COV_PARAMS = True,      # if cov matrix should not be computed by the policy net
                DIAGONAL_COV_MATRIX = True,      # learn a diagonal or full cov matrix
                MODEL_NAME_POL = "policy.pt",    # how the new model will be saved
                MODEL_NAME_VAL = "value_net.pt",
                MIN_COV = 1e-2,                  # minimum value allowed for diagonal cov matrix
                VALUE_EPOCHS = 10,
                POLICY_EPOCHS = 10,
                VALUE_BATCH_SIZE = 128,          # for now these batches are made only along the time dimension
                POLICY_BATCH_SIZE = 128,         
                VALUE_LR = 3e-2,
                POLICY_LR = 1e-2,
                NUMERICAL_EPSILON = 1e-7,        # value for numerical stability
                BETA = 5e-3,                     # weight used for entropy
                ADVANTAGE_TYPE = "GAE",          # type of advantages GAE/TD/MC
                GAE_LAMBDA = 0.99,
                POLICY_METHOD = True,
                ALGO_NAME = "ppo"

               )


class PPOAgent:

    """
    implementation of a reinforcement learning agent that uses PPO algorithm

    This implementation can be used in environments with both 
    continuous and discrete action spaces

    with continuous actions the policy network will compute 
    the parameters (means and cov matrix) of the n-dimensional normal 
    distribution the actions will be sampled from

    with discrete actions the policy network will predict the logits
    (unnormalized scores) that will be used with categorical distribution 
    to compute the probability of each one of the possible n actions 

    this implementation supports the following types of advantage estimation
    - TD: A(t) = delta_t = R_t+1+gamma*V(S_t+1)-V(S_t)
    - MC: A(t) = G_t - V(S_t)
    - GAE: A(t) = delta_t + gamma * lambda * gae_t-1 (default)

    for state dependent and full cov matrix the net will not output exactly 
    all the elements but will only produce enough values such that they can 
    be used later to form a proper cov matrix since it has to be PSD

    when a policy network is loaded but cov matrix is not computed by the network 
    its elements are reinitialized in each new training session

    resources:
    https://arxiv.org/pdf/1707.06347
    https://arxiv.org/pdf/1506.02438
    """


    def __init__(self, parameters: Params):

        if parameters.SEED:
            torch.manual_seed(parameters.SEED)

        # extract the hardcoded values from parameters

        self.numerical_epsilon = parameters.NUMERICAL_EPSILON
        self.gamma = parameters.GAMMA
        self.epsilon = parameters.PPO_EPS
        self.beta = parameters.BETA
        self.advantage_type = parameters.ADVANTAGE_TYPE
        self.gae_lambda = parameters.GAE_LAMBDA
        self.diagonal_cov = parameters.DIAGONAL_COV_MATRIX
        self.separate_cov_params = parameters.SEPARATE_COV_PARAMS 
        self.min_cov = parameters.MIN_COV
        self.value_batch_size = parameters.VALUE_BATCH_SIZE
        self.policy_batch_size = parameters.POLICY_BATCH_SIZE
        self.value_epochs = parameters.VALUE_EPOCHS 
        self.policy_epochs = parameters.POLICY_EPOCHS 
        self.device = parameters.DEVICE
        self.n_env = parameters.N_ENV
        self.policy_lr = parameters.POLICY_LR
        self.value_lr = parameters.VALUE_LR

        # extract the other values added before calling the constructor

        self.obs_size = parameters.obs_size
        self.action_space_dim = parameters.action_space_dim
        self.continuous_actions = parameters.env_is_continuous
        self.value_model_checkpoint = parameters.value_model_checkpoint
        self.policy_model_checkpoint = parameters.policy_model_checkpoint

        self.buffer = []

        if self.continuous_actions == True:

            if not self.separate_cov_params: 

                # number of means + number of elements for covariance matrix
                if not self.diagonal_cov:
                    policy_net_output_dim = self.action_space_dim + self.action_space_dim**2 
                else:
                    policy_net_output_dim = 2*self.action_space_dim
            else:
                policy_net_output_dim = self.action_space_dim
                if not self.diagonal_cov:
                    self.cov_values = nn.Parameter(torch.rand(self.action_space_dim, self.action_space_dim).to(self.device))
                else:
                    self.log_var = nn.Parameter(torch.ones(self.action_space_dim).to(self.device))

        else:
            policy_net_output_dim = self.action_space_dim

        self.policy_net = nn.Sequential(
          nn.Linear(self.obs_size, 100),
          nn.LeakyReLU(),
          nn.Linear(100, 100),
          nn.LeakyReLU(),
          nn.Linear(100, policy_net_output_dim)).to(self.device)

        self.value_net = nn.Sequential(
          nn.Linear(self.obs_size, 100),
          nn.LeakyReLU(),
          nn.Linear(100, 100),
          nn.LeakyReLU(),
          nn.Linear(100, 1)).to(self.device)

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

        # group together all the trainable parameters used by the policy

        all_policy_params = list(self.policy_net.parameters())

        if self.continuous_actions == True:
            if self.separate_cov_params:
                if not self.diagonal_cov:
                    all_policy_params.append(self.cov_values)
                else:
                    all_policy_params.append(self.log_var)

        self.optim_policy = torch.optim.Adam(all_policy_params,
                            lr = self.policy_lr)

        self.optim_value = torch.optim.Adam(self.value_net.parameters(),
                           lr = self.value_lr)

        self.mse = torch.nn.MSELoss()


    def choose_action(self, obs):

        with torch.no_grad():

            # generate a distribution with the net, then sample from it 

            if self.continuous_actions:

                # create probability distribution (n-d gaussian) 

                # run the policy to get means and covariances
                policy_net_out = self.policy_net(obs)

                means = policy_net_out[:,:self.action_space_dim]

                if not self.separate_cov_params:
                
                    if not self.diagonal_cov:

                        # this can lead to errors due to numerical instability
                        cov_values = policy_net_out[:,self.action_space_dim:]\
                                     .reshape(-1,
                                              self.action_space_dim,
                                              self.action_space_dim)
                        cov = cov_values.mT @ cov_values + self.numerical_epsilon * torch.eye(self.action_space_dim).to(self.device)

                    else:

                        cov_values = policy_net_out[:,self.action_space_dim:]
                        cov = torch.stack([torch.diag(torch.exp(e).clamp(min=self.min_cov)) for e in cov_values])

                else:

                    if not self.diagonal_cov:

                        # this can lead to errors due to numerical instability
                        cov = self.cov_values.T @ self.cov_values + self.numerical_epsilon * torch.eye(self.action_space_dim).to(self.device)

                    else:

                        cov = torch.diag(torch.exp(self.log_var).clamp(min=self.min_cov))

                probs_distribution = MultivariateNormal(means,cov)

            else:

                # run the policy to get logits
                logits  = self.policy_net(obs)

                # create the discrete distribution
                probs_distribution = Categorical(logits=logits)

            # sample from prob distribution
            action = probs_distribution.sample()

            # return also the probability of the action sampled (for later)
            log_prob_action = probs_distribution.log_prob(action)

        return action, log_prob_action


    def choose_action_greedy(self, obs):

        with torch.no_grad():

            if self.continuous_actions:
    
                # get only the means as actions 
            
                policy_net_out = self.policy_net(obs)
                action = policy_net_out[:self.action_space_dim] 

            else:

                # compute probs and return the action with the highest one 

                logits  = self.policy_net(obs)
                probs_distribution = Categorical(logits=logits)
                action = probs_distribution.probs.argmax()

        return action


    def update(self):

        """
        update function, here the buffer filled with (s,a,r,s',d,logp(a))
        transitions is used to update value and policy networks using PPO
        (note that the buffer may contain transitions from one or more episodes)
        """
    
        T = len(self.buffer)
       
        # extract all the values from the buffer into tensors so they can be
        # processed in parallel

        states = torch.stack([t[0] for t in self.buffer])
        actions = torch.stack([t[1] for t in self.buffer])
        rewards = torch.stack([t[2] for t in self.buffer])
        next_states = torch.stack([t[3] for t in self.buffer])
        dones = torch.stack([t[4] for t in self.buffer])
        log_probs_old = torch.stack([t[5] for t in self.buffer])

        with torch.no_grad():

            # generalized advantage estimators

            if self.advantage_type == "GAE":

                advantages = torch.zeros(T,self.n_env, dtype=torch.float32).to(self.device)
                returns = torch.zeros(T,self.n_env, dtype=torch.float32).to(self.device)

                values = self.value_net(states).squeeze(-1)           # (T,n_env)
                next_values = self.value_net(next_states).squeeze(-1) # (T,n_env)
                
                gae = 0

                for t in reversed(range(T)):
                    
                    delta = rewards[t] + self.gamma * next_values[t] * (1.0 - dones[t].to(int)) - values[t]
                    gae = delta + self.gamma * self.gae_lambda * (1.0 - dones[t].to(int)) * gae
                    advantages[t] = gae
                    returns[t] = advantages[t] + values[t] 
    
            # td error advantage

            if self.advantage_type == "TD":

                values = self.value_net(states).squeeze(-1)
                next_values = self.value_net(next_states).squeeze(-1)
                returns = rewards + self.gamma * next_values * (1.0 - dones.to(int))
                returns = returns.to(torch.float32)
                TD_errors = returns - values
            
                advantages = TD_errors

            # monte carlo advantage (assuming the last timestep is the end of an episode)

            # in MC estimates we don't bootstrap and for this reason all the
            # data from the last incomplete episodes should be just ignored
            # but to keep the implementation simple and readable it will be not
            # hopefully it will be like a small source of noise and if the data
            # from incomplete episodes wil be just a small percentage of the 
            # buffer the model will still be able to learn

            if self.advantage_type == "MC":

                values = self.value_net(states).squeeze(-1)
                returns = torch.zeros(T, self.n_env, dtype=torch.float32).to(self.device)
                for t in reversed(range(T)):
                    returns[t] = rewards[t] + self.gamma * returns[t] * (1.0 - dones[t].to(int))

                advantages = returns - values

            # normalize advantages

            advantages = (advantages - advantages.mean()) / (advantages.std() + self.numerical_epsilon)
    
        # update value net (actually this should be called advantage net)

        # multiple update steps with minibatches
        for _ in range(self.value_epochs):

            # arrange the numbers from 0 to T randomly 
            indices = torch.randperm(T) 

            for start in range(0, T, self.value_batch_size):
                end = start + self.value_batch_size
                mb_indices = indices[start:end]
           
                mb_states = states[mb_indices]
                mb_returns = returns[mb_indices] 
           
                self.optim_value.zero_grad()
           
                value_pred = self.value_net(mb_states).squeeze(-1)

                loss_value = self.mse(value_pred, mb_returns)
                loss_value.backward()

                # gradient clipping for more stability
                torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), 0.5)

                self.optim_value.step()
        
        # update policy with the clipped objective (PPO)

        for _ in range(self.policy_epochs):

            # arrange the numbers from 0 to T randomly 
            indices = torch.randperm(T) 

            for start in range(0, T, self.policy_batch_size):

                end = start + self.policy_batch_size
                mb_indices = indices[start:end]
           
                mb_states = states[mb_indices]
                mb_actions = actions[mb_indices]
                mb_advantages = advantages[mb_indices]
                mb_log_probs_old = log_probs_old[mb_indices]

                self.optim_policy.zero_grad()

                # generate a new distribution with the new net

                if self.continuous_actions:

                    policy_net_out = self.policy_net(mb_states)

                    means = policy_net_out[:,:,:self.action_space_dim]

                    if not self.separate_cov_params: 

                        if not self.diagonal_cov:
                            cov_values = policy_net_out[:,:,self.action_space_dim:]\
                                        .reshape(-1,self.n_env,self.action_space_dim,
                                                    self.action_space_dim)
                            cov = cov_values.transpose(-2,-1) @ cov_values + self.numerical_epsilon * torch.eye(self.action_space_dim).to(self.device)
                        else:
                            cov = torch.diag_embed(torch.exp(policy_net_out[:,:,self.action_space_dim:]))

                    else:

                        # here cov will have shape (|A|,|A|) since it is made by
                        # a separate tensor but in distribution it will be 
                        # broadcasted to shape (B,n_env,|A|,|A|)

                        if not self.diagonal_cov:
                            # this can lead to errors due to numerical instability
                            cov = self.cov_values.T @ self.cov_values + self.numerical_epsilon * torch.eye(self.action_space_dim).to(self.device)
                        else:
                            cov = torch.diag(torch.exp(self.log_var)) 

                    distributions = MultivariateNormal(means,cov)
                    log_probs = distributions.log_prob(mb_actions)

                else:

                    logits = self.policy_net(mb_states)
                    distributions = Categorical(logits=logits)

                log_probs = distributions.log_prob(mb_actions)
                entropy = distributions.entropy()

                ratio = torch.exp(log_probs - mb_log_probs_old)

                ppo_objective = -1 * torch.min(ratio*mb_advantages,
                                               torch.clip(ratio,
                                                          1-self.epsilon,
                                                          1+self.epsilon)\
                                                *mb_advantages).mean()

                ppo_objective -= self.beta*entropy.mean()
    
                ppo_objective.backward()
                # gradient clipping for more stability
                torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.5)
                self.optim_policy.step()

        # clear experience buffer
        self.buffer = []


if __name__ == "__main__":

    print(f"using device {params.DEVICE}")
    params.value_model_checkpoint = None 
    params.policy_model_checkpoint = None
    params.env_is_continuous = False 
    params.obs_size = 2
    params.action_space_dim = 3

    discrete_agent = PPOAgent(params)

    print("discrete agent created")

    params.env_is_continuous = True 

    continuous_agent = PPOAgent(params)

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
            log_prob = torch.rand(n_env).to(params.DEVICE)

            discrete_agent.buffer.append((state,discrete_actions,reward,new_state,done,log_prob))
            continuous_agent.buffer.append((state,continuous_actions,reward,new_state,done,log_prob))

            state = new_state

    discrete_agent.update()
    continuous_agent.update()

    print("update done")

