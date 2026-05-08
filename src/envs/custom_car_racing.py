import gymnasium as gym
import numpy as np


class CustomCarRacing(gym.Env):

    """
    modded version of the car racing environment that returns observations in
    the right shape according to NCHW convention used in pytorch intead of
    the HWC convention used in the original environment
    """


    metadata = {"render_modes": ["human", "rgb_array"]}


    def __init__(self,**kwargs):
        super(CustomCarRacing).__init__()
        self.env = gym.make("CarRacing-v3",**kwargs)
        self.action_space = self.env.action_space
        self.observation_space = gym.spaces.Box(0,255,(3,96,96),np.uint8)


    def step(self,action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        observation = observation.swapaxes(0,2)
        observation = observation.swapaxes(1,2)
        return (observation, reward, terminated, truncated, info)


    def reset(self,**kwargs):
        observation, info = self.env.reset(**kwargs)
        observation = observation.swapaxes(0,2)
        observation = observation.swapaxes(1,2)
        return (observation,info)


    def render(self,**kwargs):
        return self.env.render(**kwargs)


gym.register(
    id="CustomCarRacing",
    entry_point="envs.custom_car_racing:CustomCarRacing")


if __name__ == "__main__":
    env = CustomCarRacing(render_mode = "human")
    env.reset()
    for _ in range(100):
        action = env.action_space.sample()
        o,r,te,tr,i = env.step(action)
    env.close()
