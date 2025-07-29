import torch
import numpy as np
import gymnasium as gym
from PIL import Image
from ale_py import ALEInterface


class AtariWrapper(gym.Wrapper):

    def __init__(self, env, image_shape=(84, 84), frame_skip=4, no_lose=True):
        super().__init__(env)
        self.image_shape = image_shape
        self.frame_skip = frame_skip

        obs_shape = (frame_skip, self.image_shape[0], self.image_shape[1])
        self.observation_space = gym.spaces.Box(shape=obs_shape, low=0, high=1, dtype=np.float32)
        self.no_lose = no_lose

    def reset(self):
        observations = []

        obs, info = self.env.reset()
        obs = self._process_observations(obs)
        observations.append(obs)

        SPACE_ACTION = 1

        for i in range(self.frame_skip - 1):
            obs, reward, terminated, truncated, info = self.env.step(SPACE_ACTION)
            obs = self._process_observations(obs)
            observations.append(obs)

        observation = np.stack(observations)

        return observation, info

    def step(self, action):
        observations = []
        total_reward = 0
        for i in range(self.frame_skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            obs = self._process_observations(obs)
            observations.append(obs)
            total_reward += reward

        observation = np.stack(observations)

        if(self.no_lose and info['lives'] < 5):
            terminated = True

        total_reward = float(np.clip(total_reward, -1.0, 1.0))

        return observation, total_reward, terminated, truncated, info


    def _process_observations(self, obs):
        image = Image.fromarray(obs)
        image = image.convert('L')
        image = image.resize((self.image_shape[1], self.image_shape[0]))
        image_array = np.array(image).astype(np.float32)
        image_array /= 255
        return image_array
