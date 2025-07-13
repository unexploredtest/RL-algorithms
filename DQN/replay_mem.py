import random
import numpy as np
import torch

class ReplayMemory:
    def __init__(self, max_mem=1000000):
        self.size = 0
        self.current_idx = 0
        self.full = False
        self.max_mem = max_mem
        self.mem = []

    def append(self, obs, action, reward, done, next_obs):
        obs = obs.unsqueeze(0)
        if(next_obs != None):
            next_obs = next_obs.unsqueeze(0)
        
        if(self.full):
            self.mem[self.current_idx] = (obs, action, reward, done, next_obs)
        else:
            self.mem.append((obs, action, reward, done, next_obs))
            if(self.current_idx >= self.max_mem - 1):
                self.full = True
            self.size += 1
        
        self.current_idx = (self.current_idx + 1) % self.max_mem
        self.size += 1

    def sample(self, minibatch_size=32):
        batch = random.sample(self.mem[:self.size], minibatch_size)
        
        obss, actions, rewards, dones, next_obss = zip(*batch)

        obss = torch.cat(obss)
        actions = torch.cat(actions)
        rewards = torch.cat(rewards)
        dones = torch.cat(dones)
        next_obss = torch.cat(next_obss)

        return obss, actions, rewards, dones, next_obss

