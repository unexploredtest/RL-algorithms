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
env = AtariWrapper(gym.make(env_name, render_mode="rgb_array"))
nb_actions = env.action_space.n

total_frames = 5000000

device = torch.device("cpu")

network = AtariCNN(nb_actions=nb_actions)
network.to(device)
agent = AtariAgent(nb_actions=nb_actions, network=network)

frame = 0

start_frame = time.time()
obs, info = env.reset()
done = False

losses = []

while frame < total_frames:
    if done:
        obs, info = env.reset()

    action_index = agent.choose_action(torch.from_numpy(obs).unsqueeze(0).to(device))
    obs_next, reward, done, truncated, info = env.step(action_index)

    
    agent.store_transition(torch.from_numpy(obs).to(device),
        torch.tensor([action_index], dtype=torch.int64).to(device),
        torch.tensor([reward]).to(device),
        torch.tensor([done]).to(device),
        torch.from_numpy(obs_next).to(device))

    loss = agent.learn()
    if(loss != None):
        losses.append(loss)

    obs = obs_next

    frame += 1
    time_spent = time.time() - start_frame
    if(frame % 1000 == 0):
        print(f"Frame {frame}, Average loss {np.mean(losses[-200:]):.3f}, took {time_spent:.1f} secs")
        evaluate(env_name, agent, device)
        start_frame = time.time()

    


