import gymnasium as gym
import numpy as np


class CustomCarRacing(gym.Env):

    """
    modded version of the car racing environment that returns observations in
    the right shape according to NCHW convention used in pytorch intead of
    the HWC convention used in the original environment
    """


    metadata = {"render_modes": ["human", "rgb_array"]}


    def __init__(self, past_frames = 4, **kwargs):
        super().__init__()
        self.env = gym.make("CarRacing-v3",**kwargs)
        self.action_space = self.env.action_space
        self.observation_space = gym.spaces.Box(0,255,(past_frames,96,96),np.uint8)
        self.obs = np.zeros((past_frames,96,96))
        self.render_mode = kwargs['render_mode']


    def step(self,action):
        frame, reward, terminated, truncated, info = self.env.step(action)
        frame = frame.swapaxes(0,2)
        frame = frame.swapaxes(1,2)
        frame = frame.mean(axis=0).astype(np.uint8)
        new_obs = np.zeros_like(self.obs, dtype=np.uint8)
        new_obs[1:] = self.obs[:-1]
        new_obs[0] = frame
        self.obs = new_obs
        return (self.obs, reward, terminated, truncated, info)


    def reset(self,**kwargs):
        self.obs = np.zeros(self.observation_space.shape, dtype=np.uint8)
        frame, info = self.env.reset(**kwargs)
        frame = frame.swapaxes(0,2)
        frame = frame.swapaxes(1,2)
        frame = frame.mean(axis=0).astype(np.uint8)
        self.obs[0] = frame
        return (self.obs,info)


    def render(self):
        return self.env.render()


gym.register(
    id="CustomCarRacing",
    entry_point="envs.custom_car_racing:CustomCarRacing",
    max_episode_steps = 1000)


if __name__ == "__main__":
    env = CustomCarRacing(render_mode = "human")
    env.reset()
    for _ in range(100):
        action = env.action_space.sample()
        o,r,te,tr,i = env.step(action)
    env.close()
