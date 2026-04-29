import torch
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--algo',  metavar = '<ppo/dql/...>', default = None, help = "choose an algorithm")
    args = parser.parse_args()

    if args.algo == "ppo":

        from agents.ppo_agent import PPOAgent as Agent
        from parameters.ppo_params import params

        print(f"using device {params.DEVICE}")

        params.checkpoint = None
        params.obs_size = 2
        params.action_space_dim = 3

        params.env_is_continuous = False 
        discrete_agent = Agent(params)

        print("discrete agent created")

        params.env_is_continuous = True 
        continuous_agent = Agent(params)

        print("continuous agent created")

        # test loop with random data, suppose a vectorized environment and 10 steps

        state = torch.rand(params.N_ENV,2).to(params.DEVICE)

        for t in range(10):

            with torch.no_grad():
            
                discrete_actions, disc_log_prob = discrete_agent.choose_action(state)
                continuous_actions, cont_log_prob = continuous_agent.choose_action(state)

                reward = torch.rand(params.N_ENV).to(params.DEVICE)
                new_state = torch.rand(params.N_ENV,2).to(params.DEVICE)
                done = (torch.ones(params.N_ENV) if t % 10 == 0 else torch.zeros(params.N_ENV)).to(params.DEVICE)
                log_prob = torch.rand(params.N_ENV).to(params.DEVICE)

                discrete_agent.buffer.append((state,discrete_actions,reward,new_state,done,log_prob))
                continuous_agent.buffer.append((state,continuous_actions,reward,new_state,done,log_prob))

                state = new_state

        discrete_agent.update()
        continuous_agent.update()

        print("update done")
        quit()

    elif args.algo == "dql":

        from agents.dql_agent import DQLAgent as Agent
        from parameters.dql_params import params

        print(f"using device {params.DEVICE}")

        params.checkpoint = None 
        params.env_is_continuous = False 
        params.obs_size = 2
        params.action_space_dim = 3
        params.MEMORY_BATCH_SIZE = 4

        agent = Agent(params)

        print("dql_agent created")

        for __ in range(10):

            state = torch.rand(params.N_ENV,2).to(params.DEVICE)

            for t in range(10):

                with torch.no_grad():
                
                    discrete_actions, _ = agent.choose_action(state)

                    reward = torch.rand(params.N_ENV).to(params.DEVICE)
                    new_state = torch.rand(params.N_ENV,2).to(params.DEVICE)
                    done = (torch.ones(params.N_ENV) if t % 10 == 0 else torch.zeros(params.N_ENV)).to(params.DEVICE)

                    agent.buffer.append((state,discrete_actions,reward,new_state,done,_))

                    state = new_state

            agent.update()

        print("update done")
        quit()

    elif args.algo == "sac":

        from agents.sac_agent import SACAgent as Agent
        from parameters.sac_params import params

        print(f"using device {params.DEVICE}")

        params.checkpoint = None
        params.obs_size = 2
        params.action_space_dim = 3
        params.WARMUP = 0
        params.MEMORY_BATCH_SIZE  = 5

        params.env_is_continuous = False 
        discrete_agent = Agent(params)

        print("discrete agent created")

        params.env_is_continuous = True 
        continuous_agent = Agent(params)

        print("continuous agent created")

        # test loop with random data, suppose a vectorized environment and 10 steps

        state = torch.rand(params.N_ENV,2).to(params.DEVICE)

        for t in range(10):

            with torch.no_grad():
            
                discrete_actions, disc_log_prob = discrete_agent.choose_action(state)
                continuous_actions, cont_log_prob = continuous_agent.choose_action(state)

                reward = torch.rand(params.N_ENV).to(params.DEVICE)
                new_state = torch.rand(params.N_ENV,2).to(params.DEVICE)
                done = (torch.ones(params.N_ENV) if t % 10 == 0 else torch.zeros(params.N_ENV)).to(params.DEVICE)

                discrete_agent.buffer.append((state,discrete_actions,reward,new_state,done))
                continuous_agent.buffer.append((state,continuous_actions,reward,new_state,done))

                state = new_state

        discrete_agent.update()
        continuous_agent.update()

        print("update done")
        quit()

    elif args.algo == "vpg":

        from agents.vpg_agent import VPGAgent as Agent
        from parameters.vpg_params import params

        print(f"using device {params.DEVICE}")

        params.checkpoint = None 
        params.obs_size = 2
        params.action_space_dim = 3

        params.env_is_continuous = False 
        discrete_agent = Agent(params)

        print("discrete agent created")

        params.env_is_continuous = True 
        continuous_agent = Agent(params)

        print("continuous agent created")

        # test loop with random data, suppose a vectorized environment and 10 steps

        state = torch.rand(params.N_ENV,2).to(params.DEVICE)

        for t in range(10):

            with torch.no_grad():
            
                discrete_actions, disc_log_prob = discrete_agent.choose_action(state)
                continuous_actions, cont_log_prob = continuous_agent.choose_action(state)

                reward = torch.rand(params.N_ENV).to(params.DEVICE)
                new_state = torch.rand(params.N_ENV,2).to(params.DEVICE)
                done = (torch.ones(params.N_ENV) if t % 10 == 0 else torch.zeros(params.N_ENV)).to(params.DEVICE)
                log_prob = torch.rand(params.N_ENV).to(params.DEVICE)

                discrete_agent.buffer.append((state,discrete_actions,reward,new_state,done,log_prob))
                continuous_agent.buffer.append((state,continuous_actions,reward,new_state,done,log_prob))

                state = new_state

        discrete_agent.update()
        continuous_agent.update()

        print("update done")
        quit()

    elif args.algo == "ddpg":

        from agents.ddpg_agent import DDPGAgent as Agent
        from parameters.ddpg_params import params

        print(f"using device {params.DEVICE}")

        params.checkpoint = None
        params.env_is_continuous = True
        params.obs_size = 2
        params.action_space_dim = 3
        params.WARMUP = 0
        params.MEMORY_BATCH_SIZE  = 5

        agent = Agent(params)

        print("agent created")

        # test loop with random data, suppose a vectorized environment and 10 steps

        state = torch.rand(params.N_ENV,2).to(params.DEVICE)

        for t in range(10):

            with torch.no_grad():
            
                actions, cont_log_prob = agent.choose_action(state)
                reward = torch.rand(params.N_ENV).to(params.DEVICE)
                new_state = torch.rand(params.N_ENV,2).to(params.DEVICE)
                done = (torch.ones(params.N_ENV) if t % 10 == 0 else torch.zeros(params.N_ENV)).to(params.DEVICE)
                agent.buffer.append((state,actions,reward,new_state,done))
                state = new_state

        agent.update()

        print("update done")
        quit()
    else:
        raise ValueError("invalid algo")


