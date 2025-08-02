import torch
import numpy as np
import gymnasium as gym
from PIL import Image
from ale_py import ALEInterface
from env import AtariWrapper
from network import AtariCNN
from agent import AtariAgent
import time
from evaluate import evaluate

env_name = "BreakoutNoFrameskip-v4"
env = AtariWrapper(gym.make(env_name, render_mode="human"), no_lose=False)
nb_actions = env.action_space.n

total_games = 3

device = torch.device("cpu")

network = AtariCNN(nb_actions=nb_actions)
network.to(device)
agent = AtariAgent(nb_actions=nb_actions, network=network, device=device, eps=0.05)
agent.load_model("lol.model")

for i in range(total_games):
    obs, info = env.reset()
    done = False

    while not done:
        action_index = agent.choose_action(torch.from_numpy(obs).unsqueeze(0).to(device))
        obs, reward, done, truncated, info = env.step(action_index)

        if done:
            break

