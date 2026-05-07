import torch
import torch.nn as nn
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.categorical import Categorical
from utils.checkpoint import CheckpointHandler
from utils.replaymemory import ReplayMemory
from agents.base_agent import BaseAgent


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
        self.value_lr =  parameters.VALUE_LR
        self.policy_lr = parameters.POLICY_LR
        self.alpha_lr = parameters.ALPHA_LR
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
        self.double_q = parameters.USE_DOUBLE_Q_NET

        # extract the other values added before calling the constructor

        self.obs_size = parameters.obs_size
        self.action_space_dim = parameters.action_space_dim
        self.continuous_actions = parameters.env_is_continuous
        self.checkpoint = parameters.checkpoint

        if self.alpha == "auto":
            self.log_alpha = torch.nn.Parameter(torch.zeros(1).to(self.device))
        else:
            self.log_alpha = torch.log(torch.tensor(self.alpha).to(self.device))

        if self.continuous_actions:
            policy_net_output_dim = 2*self.action_space_dim 
            value_net_input_dim = self.obs_size + self.action_space_dim
            value_net_output_dim = 1
        else:
            policy_net_output_dim = self.action_space_dim
            value_net_input_dim = self.obs_size
            value_net_output_dim = self.action_space_dim

        if parameters.TARGET_H == "auto":
            self.target_h = torch.tensor(-self.action_space_dim).to(self.device) if self.continuous_actions \
            else 0.5*torch.log(torch.tensor(self.action_space_dim)).to(self.device)
        else:
            self.target_h = torch.tensor(parameters.TARGET_H).to(self.device)

        self.buffer = []
        self.tot_steps = 0

        self.memory = ReplayMemory(maxlen=self.memory_maxlen)

        self.policy_net = nn.Sequential(
          nn.Linear(self.obs_size, 100),
          nn.LeakyReLU(),
          nn.Linear(100, 100),
          nn.LeakyReLU(),
          nn.Linear(100, policy_net_output_dim)).to(self.device)

        self.value_net = nn.Sequential(
          nn.Linear(value_net_input_dim , 100),
          nn.LeakyReLU(),
          nn.Linear(100, 100),
          nn.LeakyReLU(),
          nn.Linear(100, value_net_output_dim)).to(self.device)

        self.target_value_net = nn.Sequential(
          nn.Linear(value_net_input_dim , 100),
          nn.LeakyReLU(),
          nn.Linear(100, 100),
          nn.LeakyReLU(),
          nn.Linear(100, value_net_output_dim)).to(self.device)

        self.target_value_net.load_state_dict(self.value_net.state_dict())
        value_nets_params = list(self.value_net.parameters())

        if self.double_q:

            self.sec_value_net = nn.Sequential(
              nn.Linear(value_net_input_dim , 100),
              nn.LeakyReLU(),
              nn.Linear(100, 100),
              nn.LeakyReLU(),
              nn.Linear(100, value_net_output_dim)).to(self.device)

            self.target_sec_value_net = nn.Sequential(
              nn.Linear(value_net_input_dim , 100),
              nn.LeakyReLU(),
              nn.Linear(100, 100),
              nn.LeakyReLU(),
              nn.Linear(100, value_net_output_dim)).to(self.device)

            [value_nets_params.append(e) for e in self.sec_value_net.parameters()]

            self.target_sec_value_net.load_state_dict(self.sec_value_net.state_dict())

        self.optim_policy = torch.optim.Adam(self.policy_net.parameters(),
                          lr = self.policy_lr)

        self.optim_value = torch.optim.Adam(value_nets_params,
                          lr = self.value_lr)

        if self.alpha == "auto":
            self.optim_temp = torch.optim.Adam([self.log_alpha],
                              lr = self.alpha_lr)

        self.checkpoint_handler = CheckpointHandler(self)

        if self.checkpoint:
            self.load_checkpoint(self.checkpoint, self.device)
        else:
            print("no checkpoint, training new networks")

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
                # with double q:
                # Q1/2(S_t,A_t) <- R_t+1 + not_done*gamma*min(Q1(S_t+1,A_t+1),Q2(S_t+1,A_t+1)) - alpha*log(pi(S_t+1,A_t+1))

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
                    if self.double_q:
                        min_q = torch.min(self.target_value_net(next_s_a_pairs).squeeze(),self.target_sec_value_net(next_s_a_pairs).squeeze())
                        targets = rewards + self.gamma*(1-dones.to(int))*(min_q - torch.exp(self.log_alpha.detach()) * next_action_log_probs)
                    else:
                        targets = rewards + self.gamma*(1-dones.to(int))*(self.target_value_net(next_s_a_pairs).squeeze() - torch.exp(self.log_alpha.detach()) * next_action_log_probs)
                    targets = targets.to(torch.float32) 

                s_a_pairs = torch.concat((states,actions),dim=-1)

                if self.double_q:
                    Q1_s_a = self.value_net(s_a_pairs).squeeze(-1)
                    Q2_s_a = self.sec_value_net(s_a_pairs).squeeze(-1)
                else:
                    Q_s_a = self.value_net(s_a_pairs).squeeze(-1)

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
                    if self.double_q:
                        next_values = torch.min((probs*(self.target_value_net(next_states))).sum(-1),(probs*(self.target_sec_value_net(next_states))).sum(-1))
                    else:
                        next_values = (probs*(self.target_value_net(next_states))).sum(-1)
                    targets = rewards + self.gamma*(1-dones.to(int))*(next_values + torch.exp(self.log_alpha.detach())*entropy)
                    targets = targets.to(torch.float32) 

                if self.double_q:
                    Q1_s = self.value_net(states)
                    Q1_s_a = torch.gather(Q1_s,-1,actions.unsqueeze(1)).squeeze()
                    Q2_s = self.sec_value_net(states)
                    Q2_s_a = torch.gather(Q2_s,-1,actions.unsqueeze(1)).squeeze()
                else:
                    Q_s = self.value_net(states)
                    Q_s_a = torch.gather(Q_s,-1,actions.unsqueeze(1)).squeeze()

            if self.double_q:
                loss = self.mse(targets,Q1_s_a)+self.mse(targets,Q2_s_a)
            else:
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
                if self.double_q:
                    Q1_s_a = self.value_net(s_a_pairs).squeeze(-1)
                    Q2_s_a = self.sec_value_net(s_a_pairs).squeeze(-1)
                    Q_s_a = torch.min(Q1_s_a,Q2_s_a)
                else:
                    Q_s_a = self.value_net(s_a_pairs).squeeze(-1)

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
                if self.double_q:
                    values = torch.min((probs*(self.value_net(states).detach())).sum(-1),(probs*(self.sec_value_net(states).detach())).sum(-1))
                else:
                    values = (probs*(self.value_net(states).detach())).sum(-1)
                sac_objective = -(values + torch.exp(self.log_alpha.detach())*entropy).mean()

            self.optim_policy.zero_grad()
            sac_objective.backward()
            self.optim_policy.step()

            # update alpha

            if self.alpha == "auto":

                # log_alpha is used instead of alpha because the algorithm is
                # actually learning the log itself, in this way it gets the
                # right gradient

                if self.continuous_actions:

                    alpha_loss = (self.log_alpha * (-1*logprobs.detach() - self.target_h)).mean()

                else:

                    # with discrete actions the entropy can be computed exactly
                    # in each state and the errors can be averaged

                    alpha_loss = (self.log_alpha * (entropy.detach() - self.target_h)).mean()

                self.optim_temp.zero_grad()
                alpha_loss.backward()
                self.optim_temp.step()

            # update target network

            for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
                target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

            if self.double_q:
                for target_param, param in zip(self.target_sec_value_net.parameters(), self.sec_value_net.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
