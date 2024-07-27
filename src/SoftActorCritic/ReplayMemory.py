import torch
import random
import os
import pickle
import numpy as np

class ReplayMemory:
    def __init__(self, capacity, seed, gpu):
        random.seed(seed)
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        self.device = torch.device(f"cuda:{gpu}")

    def push(self, state, action, reward, next_state, done):
        reward = np.array([reward])
        done = np.array([done])
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (torch.tensor(state, dtype=torch.float, device=self.device),
                                      torch.tensor(action, dtype=torch.float, device=self.device),
                                      torch.tensor(reward, dtype=torch.float, device=self.device),
                                      torch.tensor(next_state, dtype=torch.float, device=self.device),
                                      torch.tensor(done, dtype=torch.float, device=self.device))
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        state = torch.stack(state).to(self.device)
        action = torch.stack(action).to(self.device)
        reward = torch.stack(reward).to(self.device)
        next_state = torch.stack(next_state).to(self.device)
        done = torch.stack(done).to(self.device)
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)

    def save_buffer(self, save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        save_path = "{}/sac_buffer.pkl".format(save_dir)
        print('Saving sac_buffer.pkl to {}'.format(save_path))

        with open(save_path, 'wb') as f:
            pickle.dump(self.buffer, f)

    def load_buffer(self, save_path):
        print('Loading buffer from {}'.format(save_path))

        with open(save_path, "rb") as f:
            self.buffer = pickle.load(f)
            self.position = len(self.buffer) % self.capacity
