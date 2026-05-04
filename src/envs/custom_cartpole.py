import gymnasium as gym
import numpy as np


class CustomCartpole(gym.Env):

    """
    modded version of the cartpole environment that does not stop if the pole
    falls, the goal is to make the agent learn not only how to balance the pole 
    but also to swing it back up

    this is done mainly as an example of how to use a custom environment with 
    this script
    """


    metadata = {"render_modes": ["human", "rgb_array"]}


    def __init__(self,**kwargs):
        super(CustomCartpole).__init__()
        self.env = gym.make("CartPole-v1",**kwargs)
        self.action_space = self.env.action_space
        self.env.unwrapped.observation_space = gym.spaces.Box(-np.inf,+np.inf,(4,),np.float32)
        self.observation_space = self.env.observation_space
        self.x_threshold = 2.4 


    def step(self,action):
        observation, reward, terminated, truncated, info = self.env.step(action)

        x_position = observation[0]

        if abs(x_position) > self.x_threshold:
            terminated = True
        else:
            terminated = False
            # to suppress the damn warning
            self.env.unwrapped.steps_beyond_terminated = 1

        return (observation, reward, terminated, truncated, info)


    def reset(self,**kwargs):
        obs, info = self.env.reset(**kwargs)

        theta = self.np_random.uniform(-np.pi, np.pi)

        self.env.unwrapped.state = np.array([obs[0], obs[1], theta, obs[3]], dtype=np.float32)
        obs[2] = theta  

        return (obs,info)


    def render(self,**kwargs):
        return self.env.render(**kwargs)

gym.register(
    id="CustomCartpole",
    entry_point="envs.custom_cartpole:CustomCartpole",
    max_episode_steps=500)
