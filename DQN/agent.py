from network import AtariCNN
from replay_mem import ReplayMemory

import random
import torch
import torch.nn as nn
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np


class AtariAgent:
    def __init__(self, nb_actions=4, network=None, lr=0.00025, gamma=0.99, eps=1.0,
        eps_fframe=1e6, eps_final=0.1, minibatch_size=32, min_training_step=1000,
        max_mem=1000000):   

        self.nb_actions = nb_actions
        if(network == None):
            network = AtariCNN(nb_actions)
        self.network = network
        self.optim = torch.optim.RMSprop(network.parameters(), lr=lr, alpha=0.99, eps=1e-08)
        self.minibatch_size = minibatch_size

        self.eps = eps
        self.eps_final = eps_final
        self.eps_step = (eps - eps_final) / eps_fframe
        self.gamma = gamma
        self.min_training_step = min_training_step

        self.replay_memory = ReplayMemory(max_mem=max_mem)

    def store_transition(self, obs, action, reward, done, next_obs):
        self.replay_memory.append(obs, action, reward, done, next_obs)

    def choose_action(self, obs, eps=None):
        if(eps == None):
            eps = self.eps

        if(random.random() < eps):
            return random.randint(0, self.nb_actions - 1)
        else:
            with torch.no_grad():
                action_values = self.network(obs)
                return torch.argmax(action_values).item()

    def learn(self):
        if(not self.replay_memory.size > self.min_training_step):
            return

        obss, actions, rewards, dones, next_obss = self.replay_memory.sample(self.minibatch_size)

        ys = rewards + 0.0
        ys[dones == 0] += self.gamma * torch.max(self.network(next_obss))
        qvals = self.network(obss)
        ys_p = qvals[torch.arange(qvals.size(0), device=qvals.device), actions]

        loss = F.mse_loss(ys, ys_p)

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        self.eps = max(self.eps - self.eps_step, self.eps_final)