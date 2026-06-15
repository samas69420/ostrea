import gymnasium
import torch
import time
import argparse
import os
import datetime
from utils.plotter import Plotter
from gymnasium.wrappers import RecordVideo
from environments import environments_table 

# import to trigger the register function
from envs import *

def profile_model(algo, shortname):

    if algo == "ppo":
        from agents.ppo_agent import PPOAgent as Agent
        from parameters.ppo_params import params
    elif algo == "dql":
        from agents.dql_agent import DQLAgent as Agent
        from parameters.dql_params import params
    elif algo == "sac":
        from agents.sac_agent import SACAgent as Agent
        from parameters.sac_params import params
    elif algo == "vpg":
        from agents.vpg_agent import VPGAgent as Agent
        from parameters.vpg_params import params
    elif algo == "ddpg":
        from agents.ddpg_agent import DDPGAgent as Agent
        from parameters.ddpg_params import params
    else:
        raise ValueError("invalid algo")

    full_name = environments_table[shortname]["full"]
    args = environments_table[shortname]["args"]

    if args:
        dummy_env = gymnasium.make_vec(full_name,**args,num_envs = params.N_ENV, render_mode = None, vectorization_mode="async")
    else:
        dummy_env = gymnasium.make_vec(full_name,num_envs = params.N_ENV, render_mode = None, vectorization_mode="async")

    env_is_continuous = not (isinstance(dummy_env.action_space, gymnasium.spaces.Discrete) or \
                             isinstance(dummy_env.action_space, gymnasium.spaces.MultiDiscrete))

    params.DEVICE = torch.device("cpu")
    params.checkpoint = None
    params.env_is_continuous = env_is_continuous
    params.obs_size = dummy_env.observation_space.shape[1:]
    params.action_space_dim = dummy_env.action_space.shape[-1] if env_is_continuous \
                                                               else env.action_space[0].n
    agent = Agent(params)

    vec_obs_size = (params.N_ENV,*params.obs_size)

    # experimental_config needed to visualize the stack
    with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA],
                 profile_memory=True,
                 with_stack=True,
                 experimental_config=torch._C._profiler._ExperimentalConfig(verbose=True)) as prof:

        # emulate a update iteration

        while len(agent.buffer) < params.BUFFER_SIZE:

            S_t = torch.rand(vec_obs_size).to(params.DEVICE)

            action,logprob = agent.choose_action(S_t)

            action = action.to(params.DEVICE)
            if logprob:
                logprob = logprob.to(params.DEVICE)

            S_t_plus_1 = torch.rand(vec_obs_size).to(params.DEVICE)

            terminated = torch.zeros(params.N_ENV).to(params.DEVICE)
            truncated = torch.zeros(params.N_ENV).to(params.DEVICE)

            reward = torch.rand(params.N_ENV).to(params.DEVICE)

            agent.buffer.append((S_t, action, reward, S_t_plus_1, terminated, truncated, logprob))

        agent.update()

    prof.export_chrome_trace("trace.json")
    print("MEMORY USAGE CPU")
    print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=15))
    print("MEMORY USAGE GPU")
    print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=15))


def train_model(algo, environment, dry, checkpoint, notes):

    def evaluate_policy(env, agent, scale_action, episodes=10):

        if scale_action:
            low = torch.tensor(env.action_space.low).to(agent.device)
            high = torch.tensor(env.action_space.high).to(agent.device)

        avg_reward = 0.

        with torch.no_grad():

            for _ in range(episodes):

                state, _ = env.reset()

                done = False

                while not done:

                    state_t = torch.FloatTensor(state).to(params.DEVICE)

                    action = agent.choose_action_greedy(state_t.squeeze()).unsqueeze(0)

                    if scale_action:
                        # scale actions linearly to [low,high] assuming they are in the range [-1,1]
                        scaled_action = 0.5*((high-low)*action + (low+high))
                    else:
                        scaled_action = action

                    state, reward, term, trunc, _ = env.step(scaled_action.cpu().numpy())

                    avg_reward += reward.squeeze()

                    done = term.squeeze() or trunc.squeeze()

        return avg_reward / episodes

    def get_environments_train(shortname, n_envs):

        full_name = environments_table[shortname]["full"]
        args = environments_table[shortname]["args"]

        if args:
            env = gymnasium.make_vec(full_name,**args,num_envs = n_envs, render_mode = None, vectorization_mode="async")
            eval_env = gymnasium.make_vec(full_name,**args,num_envs = 1, render_mode = None)

        else:
            env = gymnasium.make_vec(full_name,num_envs = n_envs, render_mode = None, vectorization_mode="async")
            eval_env = gymnasium.make_vec(full_name,num_envs = 1, render_mode = None)

        is_continuous = not (isinstance(env.action_space, gymnasium.spaces.Discrete) or \
                             isinstance(env.action_space, gymnasium.spaces.MultiDiscrete))

        return env, eval_env, is_continuous

    if algo == "ppo":
        from agents.ppo_agent import PPOAgent as Agent
        from parameters.ppo_params import params
        bounded_actions = True if params.SQUASH_ACTION else False
    elif algo == "dql":
        from agents.dql_agent import DQLAgent as Agent
        from parameters.dql_params import params
        bounded_actions = False
    elif algo == "sac":
        from agents.sac_agent import SACAgent as Agent
        from parameters.sac_params import params
        bounded_actions = True
    elif algo == "vpg":
        from agents.vpg_agent import VPGAgent as Agent
        from parameters.vpg_params import params
        bounded_actions = False
    elif algo == "ddpg":
        from agents.ddpg_agent import DDPGAgent as Agent
        from parameters.ddpg_params import params
        bounded_actions = True
    else:
        raise ValueError("invalid algo")

    env, eval_env, env_is_continuous = get_environments_train(environment, params.N_ENV)

    params.checkpoint = checkpoint
    params.env_is_continuous = env_is_continuous
    params.obs_size = eval_env.observation_space.shape[1:]
    params.action_space_dim = env.action_space.shape[-1] if env_is_continuous \
                                                       else env.action_space[0].n
    agent = Agent(params)

    if not dry:

        # create dir for current training run
        current_time = str(datetime.datetime.now()).replace(" ","__").replace(".","_").replace(":","_").replace("-","_")[:-7]
        dir_name = environment+"_"+params.ALGO_NAME+"_"+current_time
        os.mkdir(dir_name)

        plotter = Plotter(dir_name)

        params.model_str = str(agent.model)
        params.notes = notes
        params.save_summary(f"{dir_name}/summary.txt")
    
    num_steps = 0
    updates = 0
    best_score = -torch.inf
    eval_score = None
    some_episode_was_truncated = False

    scale_action = bounded_actions and env_is_continuous
    if scale_action:
        low = torch.tensor(env.action_space.low).to(params.DEVICE)
        high = torch.tensor(env.action_space.high).to(params.DEVICE)

    # env is reset only here because vectorized envs do it automatically after each episode 
    observation, info = env.reset() 

    while num_steps < params.MAX_TRAINING_STEPS:
        
        buffer_return = 0

        while len(agent.buffer) < params.BUFFER_SIZE:

            S_t = torch.tensor(observation).to(params.DEVICE)

            action,logprob = agent.choose_action(S_t)

            if scale_action:
                # scale actions linearly to [low,high] assuming they are in the range [-1,1]
                scaled_action = 0.5*((high-low)*action + (low+high))
            else:
                scaled_action = action

            observation, reward, terminated, truncated, info = env.step(scaled_action.cpu().numpy())

            S_t_plus_1 = torch.tensor(observation).to(params.DEVICE)

            terminated = torch.tensor(terminated).to(params.DEVICE)
            truncated = torch.tensor(truncated).to(params.DEVICE)

            reward = torch.tensor(reward).to(params.DEVICE)

            agent.buffer.append((S_t, action, reward, S_t_plus_1, terminated, truncated, logprob))     #

            buffer_return += reward

            num_steps += 1

            if num_steps % params.CHECKPOINT_SAVE_FREQ == 0:

                if not dry:
                
                    checkpoint_path = f"{dir_name}/{params.CHECKPOINT_NAME}"
                    agent.save_checkpoint(checkpoint_path)

            if num_steps % params.PRINT_FREQ_STEPS == 0:

                eval_score = evaluate_policy(eval_env, agent, scale_action, params.N_EVAL_EPISODES)

                avg_return = buffer_return.mean().item()
                print(f"steps:{num_steps} | avg undisc return: {avg_return:.2f} | updates: {updates} | last eval: {eval_score:.2f}")

                if eval_score > best_score:

                    best_score = eval_score

                    if not dry:
                        
                        model_path = f"{dir_name}/{params.MODEL_NAME}"
                        agent.save_model(model_path)

        if not dry:
            avg_return = buffer_return.mean().item()
            plotter.record({"avg_buffer_undisc_return": avg_return,
                            "x_label":f"{params.ALGO_NAME} updates",
                            "save_freq": params.UPDATE_PLOT_SAVE_FREQ})

        agent.update()
        updates += 1

    env.close()


def test_model(algo, environment, checkpoint, n_runs, record):

    def get_environment_test(shortname, render_mode):

        full_name = environments_table[shortname]["full"]
        args = environments_table[shortname]["args"]

        if args:
            env = gymnasium.make(full_name,**args, render_mode = render_mode)

        else:
            env = gymnasium.make(full_name, render_mode = render_mode)

        is_continuous = not (isinstance(env.action_space, gymnasium.spaces.Discrete) or \
                             isinstance(env.action_space, gymnasium.spaces.MultiDiscrete))

        return env, is_continuous

    if algo == "ppo":
        from agents.ppo_agent import PPOAgent as Agent
        from parameters.ppo_params import params
        bounded_actions = True if params.SQUASH_ACTION else False
    elif algo == "dql":
        from agents.dql_agent import DQLAgent as Agent
        from parameters.dql_params import params
        bounded_actions = False
    elif algo == "sac":
        from agents.sac_agent import SACAgent as Agent
        from parameters.sac_params import params
        bounded_actions = True
    elif algo == "vpg":
        from agents.vpg_agent import VPGAgent as Agent
        from parameters.vpg_params import params
        bounded_actions = False
    elif algo == "ddpg":
        from agents.ddpg_agent import DDPGAgent as Agent
        from parameters.ddpg_params import params
        bounded_actions = True
    else:
        raise ValueError("invalid algo")

    if checkpoint == None:
        raise ValueError("checkpoint undeclared")

    print(f"testing model for {args.test} episodes")

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

    params.checkpoint = checkpoint
    params.env_is_continuous = env_is_continuous
    params.obs_size = env.observation_space.shape
    params.action_space_dim = env.action_space.shape[-1] if env_is_continuous \
                                                       else env.action_space.n

    agent = Agent(params)

    scale_action = bounded_actions and env_is_continuous
    if scale_action:
        low = torch.tensor(env.action_space.low).to(params.DEVICE)
        high = torch.tensor(env.action_space.high).to(params.DEVICE)

    for e in range(n_runs):

        observation, info = env.reset()
        done = False
        total_reward = 0
    
        # run the episode
        with torch.no_grad():

            while not done:

                action = agent.choose_action_greedy(torch.tensor(observation).to(torch.float))

                if scale_action:
                    # scale actions linearly to [low,high] assuming they are in the range [-1,1]
                    scaled_action = 0.5*((high-low)*action + (low+high))
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
    parser.add_argument('-c', '--checkpoint', metavar = 'CHECKPOINT_PATH', default = None, help = "load a checkpoint/model")
    parser.add_argument('-a', '--algo',  metavar = '<ppo/dql/...>', default = None, help = "choose an algorithm")
    parser.add_argument('-l', '--list', action='store_true', help = "list all the currently supported algorithms and environments and exit")
    parser.add_argument('-r', '--record', action='store_true', help = "record a vieo of the episodes during testing")
    parser.add_argument('--test', metavar = 'N', default = None, help = "test N episodes then quit", type = int)
    parser.add_argument('--notes', default = None, help = "notes to include in the experiment summary", type = str)
    parser.add_argument('-d', '--dry',action='store_true', help = "don't save logs or models")
    parser.add_argument('-p', '--profile',action='store_true', help = "profile memory usage for a update call")

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
               swimmer
               customcartpole""".replace(" ", ""))

        quit()

    if not args.environment:
        raise ValueError("environment undeclared")

    if args.profile:
        from torch.profiler import profile, ProfilerActivity, record_function
        profile_model(args.algo, args.environment)
        quit()

    if args.test:

        test_model(args.algo, args.environment, args.checkpoint, args.test, args.record)
        quit()

    else:
        train_model(args.algo, args.environment, args.dry, args.checkpoint, args.notes)
        quit()
