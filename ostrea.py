import gymnasium
import torch
import time
import argparse
import os
from plotter import Plotter
import datetime
from gymnasium.wrappers import RecordVideo
from environments import environments_table 


def train_model(algo, environment, dry, model_file_pol, model_file_val, notes):

    def evaluate_policy(env, agent, scale_action, episodes=10):

        avg_reward = 0.

        with torch.no_grad():

            for _ in range(episodes):

                state, _ = env.reset()

                done = False

                while not done:

                    state_t = torch.FloatTensor(state).to(params.DEVICE)

                    action = agent.choose_action_greedy(state_t.squeeze()).unsqueeze(0)

                    if scale_action:
                        scaled_action = action*env.action_space.high.max()
                    else:
                        scaled_action = action

                    state, reward, term, trunc, _ = env.step(scaled_action.cpu().numpy())

                    avg_reward += reward.squeeze()

                    done = term.squeeze() or trunc.squeeze()

        return avg_reward / episodes

    def get_environments_train(shortname, n_envs):

        full_name = environments_table[shortname]["full"]
        is_continuous = environments_table[shortname]["is_continuous"]
        args = environments_table[shortname]["args"]

        if args:
            env = gymnasium.make_vec(full_name,**args,num_envs = n_envs)
            eval_env = gymnasium.make_vec(full_name,**args,num_envs = 1)

        else:
            env = gymnasium.make_vec(full_name,num_envs = n_envs)
            eval_env = gymnasium.make_vec(full_name,num_envs = 1)

        return env, eval_env, is_continuous

    if algo == "ppo":
        from ppo_agent import PPOAgent as Agent
        from ppo_agent import params
        bounded_actions = False
    elif algo == "dql":
        from dql_agent import DQLAgent as Agent
        from dql_agent import params
        bounded_actions = False
    elif algo == "sac":
        from sac_agent import SACAgent as Agent
        from sac_agent import params
        bounded_actions = True
    elif algo == "vpg":
        from vpg_agent import VPGAgent as Agent
        from vpg_agent import params
        bounded_actions = False
    elif algo == "ddpg":
        from ddpg_agent import DDPGAgent as Agent
        from ddpg_agent import params
        bounded_actions = True
    else:
        raise ValueError("invalid algo")

    env, eval_env, env_is_continuous = get_environments_train(environment, params.N_ENV)

    params.value_model_checkpoint = model_file_val
    params.policy_model_checkpoint = model_file_pol
    params.env_is_continuous = env_is_continuous
    params.obs_size = env.observation_space.shape[-1]
    params.action_space_dim = env.action_space.shape[-1] if env_is_continuous \
                                                       else env.action_space[0].n
    agent = Agent(params)

    if not dry:

        current_time = str(datetime.datetime.now()).replace(" ","__").replace(".","_").replace(":","_").replace("-","_")[:-7]
        dir_name = environment+"_"+params.ALGO_NAME+"_"+current_time

        # create dir for current training run
        os.mkdir(dir_name)

        plotter = Plotter(dir_name)
        
        if params.POLICY_METHOD:
            params.policy_checkpoint = model_file_pol
            params.policy_net = agent.policy_net
            params.policy_optimizer = agent.optim_policy

        params.value_checkpoint = model_file_val
        params.value_net = agent.value_net
        params.value_optimizer = agent.optim_value

        params.notes = notes
        
        params.save_summary(f"{dir_name}/summary.txt")
    
    num_steps = 0
    updates = 0
    best_score = -torch.inf
    eval_score = None

    scale_action = bounded_actions and env_is_continuous

    # env is reset only here because vectorized envs do it automatically after each episode 
    observation, info = env.reset() 

    while num_steps < params.MAX_TRAINING_STEPS:
        
        buffer_return = 0

        while len(agent.buffer) < params.BUFFER_SIZE:

            S_t = torch.tensor(observation, dtype=torch.float32).to(params.DEVICE)

            action,logprob = agent.choose_action(S_t)

            if scale_action:
                scaled_action = action*env.action_space.high.max()
            else:
                scaled_action = action

            observation, reward, terminated, truncated, info = env.step(scaled_action.cpu().numpy())

            S_t_plus_1 = torch.tensor(observation, dtype=torch.float32).to(params.DEVICE)

            episode_over = torch.tensor(terminated).logical_or(torch.tensor(truncated)).to(params.DEVICE)

            reward = torch.tensor(reward).to(params.DEVICE)

            agent.buffer.append((S_t, action, reward, S_t_plus_1, episode_over, logprob))

            buffer_return += reward

            num_steps += 1

            if num_steps % params.PRINT_FREQ_STEPS == 0:

                eval_score = evaluate_policy(eval_env, agent, scale_action, params.N_EVAL_EPISODES)

                avg_return = buffer_return.mean().item()
                print(f"steps:{num_steps} | avg undisc return: {avg_return:.2f} | updates: {updates} | last eval: {eval_score:.2f}")

                if eval_score > best_score:

                    best_score = eval_score

                    if not dry:
                        
                        if params.POLICY_METHOD:
                            if params.MODEL_NAME_POL:
                                model_path_pol = f"{dir_name}/{params.MODEL_NAME_POL}"
                                print(f"saving policy model in {model_path_pol}")
                                torch.save(agent.policy_net.state_dict(), model_path_pol)

                        if params.MODEL_NAME_VAL:
                            model_path_val = f"{dir_name}/{params.MODEL_NAME_VAL}"
                            print(f"saving value model in {model_path_val}")
                            torch.save(agent.value_net.state_dict(), model_path_val)

        if not dry:
            avg_return = buffer_return.mean().item()
            plotter.record({"avg_buffer_undisc_return": avg_return,
                            "x_label":f"{params.ALGO_NAME} updates",
                            "save_freq": params.UPDATE_PLOT_SAVE_FREQ})

        agent.update()
        updates += 1

    env.close()


def test_model(algo, environment, model_file_pol, model_file_val, n_runs, record):

    def get_environment_test(shortname, render_mode):

        full_name = environments_table[shortname]["full"]
        is_continuous = environments_table[shortname]["is_continuous"]
        args = environments_table[shortname]["args"]

        if args:
            env = gymnasium.make(full_name,**args, render_mode = render_mode)

        else:
            env = gymnasium.make(full_name, render_mode = render_mode)

        return env, is_continuous

    if algo == "ppo":
        from ppo_agent import PPOAgent as Agent
        from ppo_agent import params
        bounded_actions = False
    elif algo == "dql":
        from dql_agent import DQLAgent as Agent
        from dql_agent import params
        bounded_actions = False
    elif algo == "sac":
        from sac_agent import SACAgent as Agent
        from sac_agent import params
        bounded_actions = True
    elif algo == "vpg":
        from vpg_agent import VPGAgent as Agent
        from vpg_agent import params
        bounded_actions = False
    elif algo == "ddpg":
        from ddpg_agent import DDPGAgent as Agent
        from ddpg_agent import params
        bounded_actions = True
    else:
        raise ValueError("invalid algo")

    if model_file_pol  == None and params.POLICY_METHOD:
        raise ValueError("policy model file undeclared")
    if model_file_val == None:
        raise ValueError("value model file undeclared")

    print(f"testing models: \n policy: {args.modelpol} \n value: {args.modelval} \n for {args.test} episodes")

    # some environments go too fast and the render_fps in metadata doesn't help
    render_delay = False 
    if environment == "cheetah":
        render_delay = 0.05

    if record:
        render_mode = "rgb_array"
    else:
        render_mode = "human"

    env, env_is_continuous = get_environment_test(environment, render_mode)

    if record:
        
        video_folder = "./videos"
        print(f"recording episodes into {video_folder}")

        env = RecordVideo(
            env, 
            video_folder = video_folder, 
            episode_trigger = lambda episode_id: True
        )

    if render_delay and record:
        env.unwrapped.metadata["render_fps"] = 25

    params.DEVICE = torch.device("cpu") #overwrite DEVICE to use only cpu for tests

    params.value_model_checkpoint = model_file_val
    params.policy_model_checkpoint = model_file_pol
    params.env_is_continuous = env_is_continuous
    params.obs_size = env.observation_space.shape[-1]
    params.action_space_dim = env.action_space.shape[-1] if env_is_continuous \
                                                       else env.action_space.n

    agent = Agent(params)

    for e in range(n_runs):

        observation, info = env.reset()
        done = False
        total_reward = 0
    
        # run the episode
        with torch.no_grad():

            while not done:

                action = agent.choose_action_greedy(torch.tensor(observation).to(torch.float))

                if bounded_actions and env_is_continuous:
                    scaled_action = action*env.action_space.high.max()
                else:
                    scaled_action = action

                observation, reward, terminated, truncated, info = env.step(scaled_action.cpu().numpy())

                done = terminated or truncated

                total_reward += reward

                if render_delay and not record: 
                    time.sleep(render_delay)
    
            print(f"episode {e+1} - return {total_reward}")

    env.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
                    prog = "python ostrea.py",
                    description = "== OSTREA == \nscript to train and test various types of reinforcement learning agents in the environments provided by gymnasium library",
                    formatter_class = argparse.RawTextHelpFormatter,
                    epilog = "author: samas69420")
    
    parser.add_argument('-e', '--environment', metavar = '<cartpole/lander/...>', default = None, help = "what environment should be used")
    parser.add_argument('-mp', '--modelpol', metavar = 'MODELPATH', default = None, help = "load an existing model (policy)")
    parser.add_argument('-mv', '--modelval', metavar = 'MODELPATH', default = None, help = "load an existing model (value)")
    parser.add_argument('-a', '--algo',  metavar = '<ppo/dql/...>', default = None, help = "choose an algorithm")
    parser.add_argument('-l', '--list', action='store_true', help = "list all the currently supported algorithms and environments and exit")
    parser.add_argument('-r', '--record', action='store_true', help = "record a vieo of the episodes during testing")
    parser.add_argument('--test', metavar = 'N', default = None, help = "test N episodes then quit", type = int)
    parser.add_argument('--notes', default = None, help = "notes to include in the experiment summary", type = str)
    parser.add_argument('-d', '--dry',action='store_true', help = "don't save logs or models")

    args = parser.parse_args()

    if args.list:

        print("""
               ALGORITHMS:

               dql
               vpg
               ddpg
               ppo
               sac

               ENVIRONMENTS:

               cartpole
               lander
               lander_continuous
               cheetah
               humanoid
               ant
               walker
               bipedal
               bipedal_hardcore
               acrobot
               reacher
               mountaincar_continuous
               mountaincar
               pendulum
               pusher
               hopper
               humanoid_standup
               inverted_d_pendulum
               inverted_pendulum
               swimmer""".replace(" ", ""))

        quit()
    

    if not args.environment:
        raise ValueError("environment undeclared")

    if args.test:

        test_model(args.algo, args.environment, args.modelpol, args.modelval, args.test, args.record)
        quit()

    else:
        train_model(args.algo, args.environment, args.dry, args.modelpol, args.modelval, args.notes)
        quit()
