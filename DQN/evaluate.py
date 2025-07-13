import torch
import numpy as np
import gymnasium as gym
from PIL import Image
from ale_py import ALEInterface
from env import AtariWrapper
from network import AtariCNN
from agent import AtariAgent
import time


def evaluate(env_name, agent, device, games_count=10):
    env = AtariWrapper(gym.make(env_name, render_mode="rgb_array"))
    scores = []
    for i in range(games_count):
        obs, info = env.reset()
        done = False
        total_reward = 0

        while not done:
            action_index = agent.choose_action(torch.from_numpy(obs).unsqueeze(0).to(device), eps=0.05)

            obs, reward, done, truncated, info = env.step(action_index)
            total_reward += reward

        scores.append(total_reward)

    mean_scores = sum(scores)/len(scores)

    print(f"Mean score: {mean_scores:.1f}")


