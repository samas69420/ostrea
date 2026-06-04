import torch
import torch.nn as nn
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.categorical import Categorical
from utils.checkpoint import CheckpointHandler
from agents.base_agent import BaseAgent


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
                 continuous_actions,
                 lr,
                 diagonal_cov,
                 min_cov,
                 separate_cov_params,
                 norm_obs,
                 device,
                 numerical_epsilon):

        self.observation_is_3d_tensor = len(obs_size) == 3
        self.device = device
        self.separate_cov_params = separate_cov_params
        self.diagonal_cov = diagonal_cov
        self.continuous_actions = continuous_actions
        self.action_space_dim = action_space_dim
        self.norm_obs = norm_obs
        self.min_cov = min_cov
        self.numerical_epsilon = numerical_epsilon

        if self.norm_obs:

            # obs normalization stats

            self.obs_max = torch.nn.Parameter(torch.ones(1).to(self.device))
            self.obs_min = torch.nn.Parameter(torch.zeros(1).to(self.device))

            self.obs_max.requires_grad = False
            self.obs_min.requires_grad = False

        # compute net outputs

        if self.continuous_actions == True:

            if not self.separate_cov_params:

                # number of means + number of elements for covariance matrix
                if not self.diagonal_cov:
                    policy_net_output_dim = action_space_dim + action_space_dim**2
                else:
                    policy_net_output_dim = 2*action_space_dim
            else:
                policy_net_output_dim = action_space_dim

                if not self.diagonal_cov:
                    self.cov_values = nn.Parameter(torch.rand(action_space_dim, action_space_dim).to(self.device))
                else:
                    self.log_var = nn.Parameter(torch.ones(action_space_dim).to(self.device))

        else:
            policy_net_output_dim = action_space_dim

        all_params = []

        # compute net inputs

        if self.observation_is_3d_tensor:

            # add a shared convolutional encoder

            self.encoder = torch.nn.Sequential(
                               torch.nn.Conv2d(obs_size[0], 32, kernel_size = 4, stride = 4, padding = "valid"),
                               torch.nn.ReLU(),
                               torch.nn.Conv2d(32, 64, kernel_size = 3, stride = 2, padding = "valid"),
                               torch.nn.ReLU(),
                               torch.nn.Conv2d(64, 256, kernel_size = 3, padding = "valid"),
                               torch.nn.ReLU(),
                               torch.nn.Flatten()).to(self.device)

            all_params += list(self.encoder.parameters())

            # dynamically compute the size of encoded vector that will used as input to the MLPs
            with torch.no_grad():
                dummy_obs = torch.zeros(1, *obs_size).to(self.device)
                mlp_input = self.encoder(dummy_obs).shape[1]

        elif len(obs_size) == 1:
            # input is already a vector
            mlp_input = obs_size[0]

        self.policy_net = nn.Sequential(
          nn.Linear(mlp_input, 256),
          nn.LeakyReLU(),
          nn.Linear(256, 256),
          nn.LeakyReLU(),
          nn.Linear(256, 256),
          nn.LeakyReLU(),
          nn.Linear(256, policy_net_output_dim)).to(self.device)

        self.value_net = nn.Sequential(
          nn.Linear(mlp_input, 256),
          nn.LeakyReLU(),
          nn.Linear(256, 256),
          nn.LeakyReLU(),
          nn.Linear(256, 256),
          nn.LeakyReLU(),
          nn.Linear(256, 1)).to(self.device)

        all_params += list(self.policy_net.parameters())\
                    + list(self.value_net.parameters())

        if continuous_actions == True:
            if separate_cov_params:
                if not diagonal_cov:
                    all_params.append(self.cov_values)
                else:
                    all_params.append(self.log_var)

        self.optim = torch.optim.Adam(all_params,
                     lr = lr)


    def encode(self, obs, update_obs_stats = False):
        """
        turn observation in vector if it isn't already

        returns a tensor of shape:
        (n_env, n)    - training
        (T, n_env, n) - update
        (n)           - inference/eval

        where n is the vector size and T is the buffer size
        """

        obs = obs.to(torch.float32)

        if self.norm_obs:

            if update_obs_stats:
                self.obs_max.data.copy_(torch.max(self.obs_max, obs.max()))
                self.obs_min.data.copy_(torch.min(self.obs_min, obs.min()))

            # normalize obs linear
            obs = (obs-self.obs_min)/(self.obs_max-self.obs_min)

        if self.observation_is_3d_tensor:
            if len(obs.shape) == 5:
                # update | input shape = (T, n_env, C, W, H)
                T, n_env, c, w, h = obs.shape
                obs = obs.reshape(T * n_env, c, w, h)
                obs = self.encoder(obs)
                obs = obs.reshape(T, n_env, -1)

            elif len(obs.shape) == 4:
                # training | input shape = (n_env, C, W, H)
                obs = self.encoder(obs)

            elif len(obs.shape) == 3:
                # inference | input shape = (C, W, H)
                obs = obs.unsqueeze(0)
                obs = self.encoder(obs)
                obs = obs.squeeze(0)

        return obs


    def update_parameters(self, loss):
        """
        update the internal parameters of the approximators (networks weights)
        """

        self.optim.zero_grad()

        loss.backward()

        # gradient clipping for more stability
        for name, obj in self.__dict__.items():
            if isinstance(obj, torch.nn.Module):
                torch.nn.utils.clip_grad_norm_(obj.parameters(), 0.5)
            elif isinstance(obj, torch.nn.Parameter):
                torch.nn.utils.clip_grad_norm_(obj, 0.5)

        self.optim.step()


    def value(self, vec_obs):
        """
        compute the value of a observation using the value approximator
        takes only batched vector obs of shape (B, n_env, n)
        where B is the arbitrary batch size chosen to process the buffer
        """
        return self.value_net(vec_obs)


    def compute_action(self, vec_obs):
        """
        compute action deterministically for inference/eval
        takes only vector observation of shape (n)
        """

        if self.continuous_actions:

            # get only the means as actions

            policy_net_out = self.policy_net(vec_obs)
            action = policy_net_out[:self.action_space_dim]

        else:

            # compute probs and return the action with the highest one

            logits  = self.policy_net(vec_obs)
            probs_distribution = Categorical(logits=logits)
            action = probs_distribution.probs.argmax()

        return action


    def compute_distributions(self, vec_obs):
        """
        use the approximators to compute the probability distributions
        for the input
        takes only batched vector obs of shape: (B,n_env,n)
        where B is the arbitrary batch size chosen to process the buffer
        """
        n_samp,n_env,vec_size = vec_obs.shape

        if self.continuous_actions:

            # create probability distribution (n-d gaussian)

            # run the policy to get means and covariances
            policy_net_out = self.policy_net(vec_obs)

            means = policy_net_out[:,:,:self.action_space_dim]

            if not self.separate_cov_params:

                if not self.diagonal_cov:

                    # this can lead to errors due to numerical instability
                    cov_values = policy_net_out[:,:,self.action_space_dim:]\
                                 .reshape(-1,
                                          self.action_space_dim,
                                          self.action_space_dim)

                    cov = cov_values.mT @ cov_values + self.numerical_epsilon * torch.eye(self.action_space_dim).to(self.device)
                    cov = cov.reshape(-1,n_env,self.action_space_dim,self.action_space_dim)

                else:

                    cov_values = policy_net_out[:,:,self.action_space_dim:]
                    cov = torch.stack([torch.diag(torch.exp(e).clamp(min=self.min_cov)) for e in cov_values.reshape(-1,self.action_space_dim)])
                    cov = cov.reshape(-1,n_env,self.action_space_dim,self.action_space_dim)

            else:

                if not self.diagonal_cov:

                    # this can lead to errors due to numerical instability
                    cov = self.cov_values.T @ self.cov_values + self.numerical_epsilon * torch.eye(self.action_space_dim).to(self.device)

                else:

                    cov = torch.diag(torch.exp(self.log_var).clamp(min=self.min_cov))

            probs_distribution = MultivariateNormal(means,cov)

        else:

            # run the policy to get logits
            logits  = self.policy_net(vec_obs)

            # create the discrete distribution
            probs_distribution = Categorical(logits=logits)

        return probs_distribution


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


class PPOAgent(BaseAgent):

    """
    implementation of a reinforcement learning agent that uses PPO algorithm

    this implementation supports 1d and 3d continuous observation spaces

    this implementation can be used in environments with both
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


    def __init__(self, parameters):

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
        self.batch_size = parameters.BATCH_SIZE
        self.epochs = parameters.EPOCHS
        self.device = parameters.DEVICE
        self.n_env = parameters.N_ENV
        self.lr = parameters.LR
        self.policy_method = parameters.POLICY_METHOD
        self.squash_action = parameters.SQUASH_ACTION
        self.norm_obs = parameters.NORMALIZE_OBSERVATIONS

        # extract the other values added before calling the constructor

        self.obs_size = parameters.obs_size
        self.action_space_dim = parameters.action_space_dim
        self.continuous_actions = parameters.env_is_continuous
        self.checkpoint = parameters.checkpoint

        self.buffer = []

        # create the models
        # (MLPs for vector observations, CNN + MLPs for 3d-tensors observations)

        self.model = Model(self.obs_size,
                           self.action_space_dim,
                           self.continuous_actions,
                           self.lr,
                           self.diagonal_cov,
                           self.min_cov,
                           self.separate_cov_params,
                           self.norm_obs,
                           self.device,
                           self.numerical_epsilon)

        self.checkpoint_handler = CheckpointHandler(self)

        if self.checkpoint:
            self.load_checkpoint(self.checkpoint, self.device)
        else:
            print("no checkpoint, training new networks")

        self.loss_fn = torch.nn.MSELoss()


    def choose_action(self, obs):

        with torch.no_grad():

            obs = self.model.encode(obs) # out shape: [n_env, vec]

            # add a external dimension because all the methods used to work with
            # observations assume they are 3d tensors
            obs = obs.unsqueeze(dim=0)

            # generate a distribution with the net, then sample from it

            probs_distribution = self.model.compute_distributions(obs)

            action = probs_distribution.sample().squeeze()

            log_prob_action = probs_distribution.log_prob(action).squeeze()

            if self.continuous_actions and self.squash_action:

                log_prob_action -= torch.log(1-torch.tanh(action)**2+self.numerical_epsilon).sum(-1)
                action = torch.tanh(action)

                # add a clamp or in update when the action gets unsquashed the result could explode
                action = action.clamp(-1+self.numerical_epsilon,1-self.numerical_epsilon)

        return action, log_prob_action


    def choose_action_greedy(self, obs):

        with torch.no_grad():

            obs = self.model.encode(obs)

            action = self.model.compute_action(obs)

            if self.continuous_actions and self.squash_action:
                    action = torch.tanh(action)

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

        states = torch.stack([t[0] for t in self.buffer]).to(torch.float32)
        actions = torch.stack([t[1] for t in self.buffer])
        rewards = torch.stack([t[2] for t in self.buffer])
        next_states = torch.stack([t[3] for t in self.buffer]).to(torch.float32)
        terminated = torch.stack([t[4] for t in self.buffer])
        truncated = torch.stack([t[5] for t in self.buffer])
        log_probs_old = torch.stack([t[6] for t in self.buffer])

        with torch.no_grad():

            enc_states = self.model.encode(states, update_obs_stats = True)
            enc_next_states = self.model.encode(next_states, update_obs_stats = True)

            # generalized advantage estimators

            if self.advantage_type == "GAE":

                # note: according to the current version of the code the time
                # dimension and the env dimension should stay separated for GAEs
                # to be computed correctly

                advantages = torch.zeros(T,self.n_env, dtype=torch.float32).to(self.device)
                returns = torch.zeros(T,self.n_env, dtype=torch.float32).to(self.device)

                values = torch.zeros_like(returns).to(self.device) # (T,n_env)
                next_values = torch.zeros_like(returns).to(self.device) # (T,n_env)

                for index in range(0, T, self.batch_size):
                    end = min(T, index+self.batch_size)
                    values[index:end] = self.model.value(enc_states[index:end]).squeeze(-1)
                    next_values[index:end] = self.model.value(enc_next_states[index:end]).squeeze(-1)
                
                gae = 0

                for t in reversed(range(T)):
                    
                    dones = terminated[t].logical_or(truncated[t]).to(torch.float32)

                    # theoretically here there should be bootstrap with truncation
                    # but since according to the gymnasium's reset logic the next
                    # state i would find here as s_t_plus_1 is the first one
                    # of a new episode theres no bootstrap even if this is like
                    # considering terminal a non-terminal state
                    delta = rewards[t] + self.gamma * next_values[t] * (1.0 - dones) - values[t]

                    gae = delta + self.gamma * self.gae_lambda * (1.0 - dones) * gae
                    advantages[t] = gae
                    returns[t] = advantages[t] + values[t] 
    
            # td error advantage

            if self.advantage_type == "TD":

                returns = torch.zeros(T,self.n_env, dtype=torch.float32).to(self.device)
                values = torch.zeros_like(returns).to(self.device) # (T,n_env)
                next_values = torch.zeros_like(returns).to(self.device) # (T,n_env)

                dones = terminated.logical_or(truncated).to(torch.float32)

                for index in range(0, T, self.batch_size):
                    end = min(T, index+self.batch_size)
                    values[index:end] = self.model.value(enc_states[index:end]).squeeze(-1)
                    next_values[index:end] = self.model.value(enc_next_states[index:end]).squeeze(-1)

                returns = rewards + self.gamma * next_values * (1.0 - dones)
                returns = returns.to(torch.float32)
                TD_errors = returns - values
            
                advantages = TD_errors

            # monte carlo advantage (assuming the last timestep is the end of an episode)

            # MC estimates don't bootstrap and for this reason all the
            # data from the last incomplete episodes should be just ignored
            # but it will be not to keep the implementation simple and readable
            # hopefully it will be like a source of noise and if the data
            # from incomplete episodes will be just a small percentage of the
            # buffer the model will still be able to learn

            if self.advantage_type == "MC":

                returns = torch.zeros(T,self.n_env, dtype=torch.float32).to(self.device)
                values = torch.zeros_like(returns).to(self.device) # (T,n_env)

                for index in range(0, T, self.batch_size):
                    end = min(T, index+self.batch_size)
                    values[index:end] = self.model.value(enc_states[index:end]).squeeze(-1)

                for t in reversed(range(len(returns))):
                    dones = terminated[t].logical_or(truncated[t]).to(torch.float32)
                    next_return = returns[t+1] if t + 1 < T else 0.0
                    returns[t] = rewards[t] + self.gamma * next_return * (1.0 - dones[t])

                advantages = returns - values

            # normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + self.numerical_epsilon)
    
        # update nets

        # multiple update steps with minibatches
        for _ in range(self.epochs):

            indices = torch.randperm(T)

            for start in range(0, T, self.batch_size):

                end = min(T, start+self.batch_size)
                mb_indices = indices[start:end]

                mb_states = states[mb_indices]
                mb_returns = returns[mb_indices]
                mb_actions = actions[mb_indices]
                mb_advantages = advantages[mb_indices]
                mb_log_probs_old = log_probs_old[mb_indices]

                # compute value loss

                mb_states = self.model.encode(mb_states)

                value_pred = self.model.value(mb_states).squeeze(-1)
                loss_v = self.loss_fn(value_pred, mb_returns)

                # compute policy objective

                distributions = self.model.compute_distributions(mb_states)

                if self.continuous_actions:

                    if self.squash_action:

                        # here unsquashed actions are needed
                        unsquashed_actions = torch.atanh(mb_actions)
                        log_probs = distributions.log_prob(unsquashed_actions)
                        log_probs -= torch.log(1-mb_actions**2+self.numerical_epsilon).sum(-1)

                    elif not self.squash_action:

                        log_probs = distributions.log_prob(mb_actions)

                elif not self.continuous_actions:

                    log_probs = distributions.log_prob(mb_actions)

                # it doesn't really matter to correct entropy for squashed actions
                entropy = distributions.entropy()

                ratio = torch.exp(log_probs - mb_log_probs_old)

                ppo_objective = -torch.min(ratio*mb_advantages,
                                           torch.clip(ratio,
                                                      1-self.epsilon,
                                                      1+self.epsilon)\
                                            *mb_advantages).mean()

                loss = 0.5*loss_v + ppo_objective - self.beta*entropy.mean()
    
                self.model.update_parameters(loss)

        # clear experience buffer
        self.buffer = []
