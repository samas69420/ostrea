import torch
import torch.nn as nn
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.categorical import Categorical
from utils.checkpoint import CheckpointHandler
from utils.replaymemory import ReplayMemory
from agents.base_agent import BaseAgent
from networks.cnn import CNN
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
                 continuous_actions,
                 lr,
                 min_logvar,
                 max_logvar,
                 device,
                 tau,
                 double_q_net,
                 norm_obs,
                 alpha,
                 target_h,
                 numerical_epsilon):

        self.observation_is_3d_tensor = len(obs_size) == 3
        self.device = device
        self.continuous_actions = continuous_actions
        self.action_space_dim = action_space_dim
        self.min_logvar = min_logvar
        self.max_logvar = max_logvar
        self.numerical_epsilon = numerical_epsilon
        self.tau = tau
        self.double_q_net = double_q_net
        self.norm_obs = norm_obs
        self.alpha = alpha

        if self.norm_obs:

            # obs normalization stats, used as Parameter objs for the checkpoint handler

            self.obs_max = torch.nn.Parameter(torch.ones(1).to(self.device))
            self.obs_min = torch.nn.Parameter(torch.zeros(1).to(self.device))

            self.obs_max.requires_grad = False
            self.obs_min.requires_grad = False

        all_trainable_params = []

        # compute net inputs

        if self.observation_is_3d_tensor:

            # add convolutional encoders for policy and value
            # the value encoder is shared between the first and the second
            # value network (if used), this is done to save some memory

            self.value_encoder = CNN(in_channels = obs_size[0],
                                     device = self.device)

            self.target_value_encoder = CNN(in_channels = obs_size[0],
                                            device = self.device)

            self.policy_encoder = CNN(in_channels = obs_size[0],
                                      device = self.device)

            self.target_policy_encoder = CNN(in_channels = obs_size[0],
                                             device = self.device)

            self.target_policy_encoder.requires_grad = False
            self.target_value_encoder.requires_grad = False
            self.target_value_encoder.load_state_dict(self.value_encoder.state_dict())
            self.target_policy_encoder.load_state_dict(self.policy_encoder.state_dict())
            all_trainable_params += list(self.value_encoder.parameters())
            all_trainable_params += list(self.policy_encoder.parameters())

            # dynamically compute the size of encoded vector that will used as input to the MLPs
            # assuming the same output shape for both the encoders
            with torch.no_grad():
                dummy_obs = torch.zeros(1, *obs_size).to(self.device)
                policy_net_input_dim = self.policy_encoder(dummy_obs).shape[1]

        elif len(obs_size) == 1:
            # input is already a vector
            policy_net_input_dim = obs_size[0]

        if self.continuous_actions:
            policy_net_output_dim = 2*self.action_space_dim
            value_net_input_dim = policy_net_input_dim + self.action_space_dim
            value_net_output_dim = 1
        else:
            policy_net_output_dim = self.action_space_dim
            value_net_input_dim = policy_net_input_dim
            value_net_output_dim = self.action_space_dim

        self.policy_net = MLP(input_dim = policy_net_input_dim,
                              output_dim = policy_net_output_dim,
                              n_layers = 4,
                              hidden_dim = 256,
                              activation_constructor = nn.LeakyReLU,
                              device = self.device)

        self.value_net = MLP(input_dim = value_net_input_dim,
                             output_dim = value_net_output_dim,
                             n_layers = 4,
                             hidden_dim = 256,
                             activation_constructor = nn.LeakyReLU,
                             device = self.device)

        self.target_value_net = MLP(input_dim = value_net_input_dim,
                                    output_dim = value_net_output_dim,
                                    n_layers = 4,
                                    hidden_dim = 256,
                                    activation_constructor = nn.LeakyReLU,
                                    device = self.device)

        self.target_value_net.requires_grad = False
        self.target_value_net.load_state_dict(self.value_net.state_dict())

        all_trainable_params += list(self.policy_net.parameters())
        all_trainable_params += list(self.value_net.parameters())

        if alpha == "auto":
            self.log_alpha = torch.nn.Parameter(torch.zeros(1).to(self.device))
            all_trainable_params += [self.log_alpha]
        else:
            self.log_alpha = torch.log(torch.tensor(self.alpha).to(self.device))

        if self.double_q_net:

            self.sec_value_net = MLP(input_dim = value_net_input_dim,
                                     output_dim = value_net_output_dim,
                                     n_layers = 4,
                                     hidden_dim = 256,
                                     activation_constructor = nn.LeakyReLU,
                                     device = self.device)

            self.target_sec_value_net = MLP(input_dim = value_net_input_dim,
                                            output_dim = value_net_output_dim,
                                            n_layers = 4,
                                            hidden_dim = 256,
                                            activation_constructor = nn.LeakyReLU,
                                            device = self.device)

            self.target_sec_value_net.requires_grad = False
            self.target_sec_value_net.load_state_dict(self.sec_value_net.state_dict())
            [all_trainable_params.append(e) for e in self.sec_value_net.parameters()]

        self.optim = torch.optim.Adam(all_trainable_params, lr = lr)


    def encode(self, obs, target_net = False, update_obs_stats = False, policy = False):
        """
        turn observation in vector if it isn't already

        returns a tensor of shape:
        (n_env/B, n)    - training/update
        (n)             - inference/eval

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

            if policy and not target_net:
                encoder = self.policy_encoder
            elif policy and target_net:
                encoder = self.target_policy_encoder
            elif not policy and not target_net:
                encoder = self.value_encoder
            elif not policy and target_net:
                encoder = self.target_value_encoder

            if len(obs.shape) == 4:
                # training/update | input shape = (n_env/B, C, W, H)
                obs = encoder(obs)

            elif len(obs.shape) == 3:
                # inference | input shape = (C, W, H)
                obs = obs.unsqueeze(0)
                obs = encoder(obs)
                obs = obs.squeeze(0)

        return obs


    def compute_action(self, vec_obs):
        """
        compute action deterministically for inference/eval
        takes only vector observation of shape (n)
        """

        if self.continuous_actions:

            # get only the (squashed) means as actions

            policy_net_out = self.policy_net(vec_obs)
            means = policy_net_out[:self.action_space_dim]
            action = torch.tanh(means)

        else:

            # run the policy to get logits

            logits  = self.policy_net(vec_obs)
            probs_distribution = Categorical(logits=logits)
            action = probs_distribution.probs.argmax()

        return action


    def compute_distributions(self, obs):
        """
        use the approximator to compute the probability distributions
        for the input
        takes only batched vector obs of shape: (B,n)
        """

        if self.continuous_actions:

            # run the policy to get means and covariances
            policy_net_out = self.policy_net(obs)

            # compute the (unbounded) probability distribution
            means = policy_net_out[:,:self.action_space_dim]
            log_var = policy_net_out[:,self.action_space_dim:]
            var = torch.exp(torch.clamp(log_var, min = self.min_logvar, max = self.max_logvar))
            cov = torch.diag_embed(var)
            probs_distribution = MultivariateNormal(means,cov)

        else:

            # run the policy to get logits
            logits  = self.policy_net(obs)

            # create the discrete distribution
            probs_distribution = Categorical(logits=logits)

        return probs_distribution


    def value_disc(self, states, target_net=False):
        """
        use the value approximator (or its target version) to compute
        the value of state-action pair(s), for discrete actions
        returns TODO
        """

        if target_net:

            values = self.target_value_net(states)
            if self.double_q_net:
                sec_values = self.target_sec_value_net(states)
                return (values, sec_values)
            return values

        else:

            values = self.value_net(states)
            if self.double_q_net:
                sec_values = self.sec_value_net(states)
                return (values, sec_values)
            return values


    def value_cont(self, state_action, target_net=False):
        """
        use the value approximator (or its target version) to compute
        the value of state-action pair(s), for continuous actions
        returns TODO
        """

        if target_net:

            values = self.target_value_net(state_action).squeeze()
            if self.double_q_net:
                sec_values = self.target_sec_value_net(state_action).squeeze()
                return(values, sec_values)

        else:

            values = self.value_net(state_action).squeeze()
            if self.double_q_net:
                sec_values = self.sec_value_net(state_action).squeeze()
                return(values, sec_values)

        return values


    def reparam_sample(self, states):
        """
        sample a action using the reparametrization trick
        a = mu(state) + std(state)*noise
        """

        policy_net_out = self.policy_net(states)

        means = policy_net_out[:,:self.action_space_dim]
        log_var = policy_net_out[:,self.action_space_dim:]
        log_var = torch.clamp(log_var, min = self.min_logvar, max = self.max_logvar)
        var = torch.exp(log_var)
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(means)

        actions_unbounded = means + (std * eps)
        actions = torch.tanh(actions_unbounded)

        # compute also the (log)probability for each action sampled

        cov = torch.diag_embed(var)
        distributions = MultivariateNormal(means,cov)
        logprobs = distributions.log_prob(actions_unbounded)
        logprobs -= torch.log(1-torch.tanh(actions_unbounded)**2+self.numerical_epsilon).sum(-1)

        return actions, logprobs


    def update_parameters(self, value_loss, sac_objective, alpha_loss=None):
        """
        update the internal parameters of the approximators
        (networks weights + alpha) sequentially to prevent
        gradient interference
        """
        self.optim.zero_grad()

        # compute the gradient w.r.t. value net parameters
        # retain graph is needed because of the shared encoder
        value_loss.backward(retain_graph = True)

        # deactivate gradients for value net weights, this is necessary
        # because to compute the sac_objective the value network was used again
        # and without blocking the gradients the backward of sac_objective would
        # accumulate gradients also for the value net weights, corrupting
        # the ones already computed by value_loss.backward()
        #
        # by blocking the gradients for the value net parameters only the ones
        # w.r.t. policy net weights (and value net input layers since they're
        # needed for the chain rule) will be computed
        #
        # this was't a problem with two different optimizers because the gradients
        # computed with value_loss.backward() were used before the
        # sac_objective.backward() call and even after the call only the
        # optimizer liked to policy weights was used, the gradients w.r.t.
        # value weights were left unused and cleared before the next
        # value_loss.backward() call,
        #
        # this isn't a problem for ppo either because in ppo the gradients from
        # value and policy loss don't interfere with each other

        for p in self.value_net.parameters():
            p.requires_grad = False
        if self.double_q_net:
            for p in self.sec_value_net.parameters():
                p.requires_grad = False

        sac_objective.backward()

        for p in self.value_net.parameters():
            p.requires_grad = True
        if self.double_q_net:
            for p in self.sec_value_net.parameters():
                p.requires_grad = True

        if alpha_loss is not None:
            alpha_loss.backward()

        # gradient clipping for more stability
        for name, obj in self.__dict__.items():
            if isinstance(obj, torch.nn.Module):
                torch.nn.utils.clip_grad_norm_(obj.parameters(), 0.5)
            elif isinstance(obj, torch.nn.Parameter):
                torch.nn.utils.clip_grad_norm_(obj, 0.5)

        self.optim.step()

        # update target networks
        for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

        if self.double_q_net:
            for target_param, param in zip(self.target_sec_value_net.parameters(), self.sec_value_net.parameters()):
                target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

        if hasattr(self, 'policy_encoder'):
            for target_param, param in zip(self.target_policy_encoder.parameters(), self.policy_encoder.parameters()):
                target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

        if hasattr(self, 'value_encoder'):
            for target_param, param in zip(self.target_value_encoder.parameters(), self.value_encoder.parameters()):
                target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)


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


class SACAgent(BaseAgent):

    """
    implementation of a reinforcement learning agent that uses SAC algorithm

    this implementation can be used in environments with both 
    continuous and discrete action spaces

    with continuous actions the policy network will compute 
    the parameters (means and cov matrix) of the n-dimensional normal 
    distribution the actions will be sampled from

    with discrete actions the policy network will compute the logits
    (unnormalized scores) that will be used with categorical distribution
    to get the probability of each one of the possible n actions

    this implementation assumes actions in the range [-1,1]
    
    resources:
    https://arxiv.org/pdf/1801.01290
    https://arxiv.org/pdf/1812.05905
    """


    def __init__(self, parameters):

        # extract the hardcoded values from parameters

        self.gamma = parameters.GAMMA
        self.lr =  parameters.LR
        self.alpha = parameters.ALPHA
        self.memory_maxlen = parameters.MEMORY_MAXLEN
        self.memory_batch_size = parameters.MEMORY_BATCH_SIZE
        self.gradient_steps = parameters.GRADIENT_STEPS 
        self.warmup = parameters.WARMUP
        self.tau = parameters.TAU
        self.device = parameters.DEVICE
        self.numerical_epsilon = parameters.NUMERICAL_EPSILON
        self.max_logvar = parameters.MAX_LOGVAR
        self.min_logvar = parameters.MIN_LOGVAR
        self.policy_method = parameters.POLICY_METHOD
        self.double_q_net = parameters.USE_DOUBLE_Q_NET
        self.norm_obs = parameters.NORMALIZE_OBSERVATIONS

        # extract the other values added before calling the constructor

        self.obs_size = parameters.obs_size
        self.action_space_dim = parameters.action_space_dim
        self.continuous_actions = parameters.env_is_continuous
        self.checkpoint = parameters.checkpoint

        self.buffer = []
        self.tot_steps = 0

        self.memory = ReplayMemory(maxlen=self.memory_maxlen)

        if parameters.TARGET_H == "auto":
            self.target_h = torch.tensor(-self.action_space_dim).to(self.device) if self.continuous_actions \
                            else 0.5*torch.log(torch.tensor(self.action_space_dim)).to(self.device)
        else:
            self.target_h = torch.tensor(parameters.TARGET_H).to(self.device)

        self.model = Model(self.obs_size,
                           self.action_space_dim,
                           self.continuous_actions,
                           self.lr,
                           self.min_logvar,
                           self.max_logvar,
                           self.device,
                           self.tau,
                           self.double_q_net,
                           self.norm_obs,
                           self.alpha,
                           self.target_h,
                           self.numerical_epsilon)

        self.checkpoint_handler = CheckpointHandler(self)

        if self.checkpoint:
            self.load_checkpoint(self.checkpoint, self.device)
        else:
            print("no checkpoint, training new networks")

        self.loss_fn = torch.nn.MSELoss()


    def choose_action_greedy(self, obs):

        with torch.no_grad():

            obs = self.model.encode(obs, policy = True)

            action = self.model.compute_action(obs)

        return action


    def choose_action(self, obs):

        self.tot_steps += 1

        # sample from the actual distribution generated by the net
        
        with torch.no_grad():

            obs = self.model.encode(obs, policy = True) # out shape: [n_env, vec]

            probs_distribution = self.model.compute_distributions(obs)
            action = probs_distribution.sample()

            if self.continuous_actions:

                action = torch.tanh(action)

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

        self.update_memory()

        if len(self.memory) < self.memory_batch_size or \
            self.tot_steps < self.warmup:
            return

        T = len(self.buffer)

        for _ in range(self.gradient_steps):

            batch = self.memory.sample(self.memory_batch_size)

            states = torch.stack([t[0] for t in batch]).flatten(0,1)
            actions = torch.stack([t[1] for t in batch]).flatten(0,1) # bounded
            rewards = torch.stack([t[2] for t in batch]).flatten(0,1)
            next_states = torch.stack([t[3] for t in batch]).flatten(0,1)
            dones = torch.stack([t[4] for t in batch]).flatten(0,1)

            # encode with value encoder (with grad) + update obs stats for normalization
            val_enc_states = self.model.encode(states, policy = False, update_obs_stats = True)

            # update soft q function

            # sample next actions using the target policy in states
            # collected by behavior policy (without reparametrization
            # trick cause gradient is not needed here)

            if self.continuous_actions:

                # with continuous actions to compute the target it is necessary
                # to evaluate Q on sampled next actions
                # Q(S_t,A_t) <- R_t+1 + not_done*gamma*Q(S_t+1,A_t+1) - alpha*log(pi(S_t+1,A_t+1))
                # with double q:
                # Q1/2(S_t,A_t) <- R_t+1 + not_done*gamma*min(Q1(S_t+1,A_t+1),Q2(S_t+1,A_t+1)) - alpha*log(pi(S_t+1,A_t+1))

                with torch.no_grad():

                    # encode with policy encoder
                    pol_enc_next_states = self.model.encode(next_states, target_net = True, policy = True)

                    # encode with value encoder
                    val_enc_next_states = self.model.encode(next_states, target_net = True, policy = False)

                    next_probs_dist = self.model.compute_distributions(pol_enc_next_states)

                    next_actions_unbounded = next_probs_dist.sample()
                    next_action_log_probs = next_probs_dist.log_prob(next_actions_unbounded)

                    # apply tanh and adjust probability
                    next_actions = torch.tanh(next_actions_unbounded)
                    next_action_log_probs -= torch.log(1-torch.tanh(next_actions_unbounded)**2+self.numerical_epsilon).sum(-1)

                    next_s_a_pairs = torch.concat((val_enc_next_states, next_actions),dim=-1)

                    if self.double_q_net:
                        next_values1, next_values2 = self.model.value_cont(next_s_a_pairs, target_net = True)
                        next_values = torch.min(next_values1, next_values2)
                    else:
                        next_values = self.model.value_cont(next_s_a_pairs, target_net = True)

                    targets = rewards + self.gamma*(1-dones.to(int))*(next_values - torch.exp(self.model.log_alpha.detach()) * next_action_log_probs)
                    targets = targets.to(torch.float32)

                # compute predicted values

                s_a_pairs = torch.concat((val_enc_states,actions),dim=-1)
                if self.double_q_net:
                    Q1_s_a,Q2_s_a = self.model.value_cont(s_a_pairs)
                    value_loss = self.loss_fn(targets,Q1_s_a)+self.loss_fn(targets,Q2_s_a)
                else:
                    Q_s_a = self.model.value_cont(s_a_pairs)
                    value_loss = self.loss_fn(targets,Q_s_a)

            elif not self.continuous_actions:

                # with discrete action space there is no need to apply tanh
                # and also entropy and value of the next state can be computed
                # exactly using all actions instead of sampling
                # Q(S_t,A_t) <- R_t+1 + not_done*gamma*(sum_a(pi(S_t+1,A_t+1)*Q(S_t+1,A_t+1)) + alpha*entropy(pi(S_t+1,A_t+1)))

                with torch.no_grad():

                    # encode with policy encoder
                    pol_enc_next_states = self.model.encode(next_states, target_net = True, policy = True)

                    # encode with value encoder
                    val_enc_next_states = self.model.encode(next_states, target_net = True, policy = False)

                    next_probs_dist = self.model.compute_distributions(pol_enc_next_states)

                    entropy = next_probs_dist.entropy()
                    probs = next_probs_dist.probs

                    next_values = self.model.value_disc(val_enc_next_states, target_net = True)

                    if self.double_q_net:
                        next_values1, next_values2 = next_values
                        next_state_values1, next_state_values2 = ((probs*(next_values1)).sum(-1),
                                                                  (probs*(next_values2)).sum(-1))
                        next_state_values = torch.min(next_state_values1, next_state_values2)
                    else:
                        next_state_values = (probs*(next_values)).sum(-1)

                    targets = rewards + self.gamma*(1-dones.to(int))*(next_state_values + torch.exp(self.model.log_alpha.detach())*entropy)
                    targets = targets.to(torch.float32)

                if self.double_q_net:
                    Q1_s, Q2_s = self.model.value_disc(val_enc_states)
                    Q1_s_a = torch.gather(Q1_s,-1,actions.unsqueeze(1)).squeeze()
                    Q2_s_a = torch.gather(Q2_s,-1,actions.unsqueeze(1)).squeeze()
                    value_loss = self.loss_fn(targets,Q1_s_a)+self.loss_fn(targets,Q2_s_a)
                else:
                    Q_s = self.model.value_disc(val_enc_states)
                    Q_s_a = torch.gather(Q_s,-1,actions.unsqueeze(1)).squeeze()
                    value_loss = self.loss_fn(targets,Q_s_a)

            # update policy

            # encode with policy encoder (with grad)
            pol_enc_states = self.model.encode(states, policy = True)

            val_enc_states_det = self.model.encode(states, policy=False).detach()

            if self.continuous_actions:

                # sample actions using reparametrization trick

                # states here is not detached so the policy objective can affect the policy encoder
                actions, logprobs = self.model.reparam_sample(pol_enc_states)

                # states here is detached so the policy objective can not affect the value encoder 
                s_a_pairs = torch.concat((val_enc_states_det,actions),dim=-1)

                if self.double_q_net:
                    Q1_s_a,Q2_s_a = self.model.value_cont(s_a_pairs)
                    Q_s_a = torch.min(Q1_s_a.squeeze(-1),Q2_s_a.squeeze(-1))
                else:
                    Q_s_a = self.model.value_cont(s_a_pairs).squeeze(-1)

                sac_objective = (torch.exp(self.model.log_alpha.detach()) * logprobs - Q_s_a).mean()

            else:

                # with discrete action space also the objective can be computed
                # exactly considering all the actions instead of sampling one

                # states here is not detached so the policy objective can affect the policy encoder
                probs_dist = self.model.compute_distributions(pol_enc_states)
                entropy = probs_dist.entropy()
                probs = probs_dist.probs

                # states here is detached so the policy objective can not affect the value encoder
                values = self.model.value_disc(val_enc_states_det)

                if self.double_q_net:
                    values1, values2 = values
                    state_values1, state_values2 = ((probs*(values1)).sum(-1),
                                                    (probs*(values2)).sum(-1))
                    state_values = torch.min(state_values1, state_values2)
                else:
                    state_values = (probs*(values)).sum(-1)

                sac_objective = -(state_values + torch.exp(self.model.log_alpha.detach())*entropy).mean()

            # update temperature

            alpha_loss = None

            if self.alpha == "auto":
                if self.continuous_actions:
                    alpha_loss = (self.model.log_alpha * (-1*logprobs.detach() - self.target_h)).mean()
                else:
                    alpha_loss = (self.model.log_alpha * (entropy.detach() - self.target_h)).mean()

            self.model.update_parameters(value_loss, sac_objective, alpha_loss)
