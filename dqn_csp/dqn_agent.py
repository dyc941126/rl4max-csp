import os.path
import random

import torch
from torch_geometric.data import Data, Batch

from dqn_csp.env import Environment
from dqn_csp.memory import Mem
import torch.nn.functional as F

class DQNAgent:
    def __init__(self, train_list, valid_list, model, target_model, optimizer, model_pth='../models', device='cpu', capacity=100000, validation_interval=50, epsilon=.9, gamma=.99):
        self.train_list = train_list
        self.valid_list = valid_list
        self.model = model
        self.target_model = target_model
        self.device = torch.device(device)

        self.model.to(self.device)
        self.target_model.to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())

        self.model_pth = model_pth
        if not os.path.exists(model_pth):
            os.makedirs(model_pth)

        self.optimizer = optimizer
        self.memory = Mem(capacity)
        self.validation_interval = validation_interval
        self.epsilon = epsilon
        self.gamma = gamma

    def train(self, nb_epoch, batch_size=32):
        for ep in range(nb_epoch):
            pth = random.choice(self.train_list)
            # pth = '../problems/train/125.xml'
            env = Environment(pth, device=self.device)
            env.reset(True)
            sar = []
            self.model.eval()
            total_r = 0
            losses = [0]
            while True:
                if ep != 0:
                    losses.append(self._learn(batch_size))
                x, edge_index, edge_attr, scatter_index, scatter_norm, action_space = env.observe()
                s_prime = Data(x, edge_index, edge_attr, scatter_index=scatter_index, scatter_norm=scatter_norm, action_space=action_space)
                if len(sar) != 0:
                    self.memory.add(sar[0], sar[1], sar[2], s_prime, False)
                if random.random() < self.epsilon:
                    q_value = self.model.inference(Data(x, edge_index, edge_attr), scatter_index, scatter_norm, action_space)
                    action = q_value.argmax().item()
                else:
                    action = random.choice([i for i in range(len(action_space))])
                s = Data(x, edge_index, edge_attr, scatter_index=scatter_index, scatter_norm=scatter_norm)
                r, done = env.act(action)
                total_r += r
                sar = [s, action_space[action], r]
                if done:
                    self.memory.add(s, action_space[action], r, s_prime, True)  # we don't care s'
                    break
            print(ep, '{:.4f}'.format(-total_r / len(env.functions)), '{:.4f}'.format(sum(losses) / len(losses)), sep='\t')
            if ep % self.validation_interval == 0 and ep != 0:
                reward = 0
                for pth in self.valid_list:
                    env = Environment(pth, device=self.device)
                    env.reset()
                    while True:
                        x, edge_index, edge_attr, scatter_index, scatter_norm, action_space = env.observe()
                        q_value = self.model.inference(Data(x, edge_index, edge_attr), scatter_index, scatter_norm,
                                                       action_space)
                        action = q_value.argmax().item()
                        r, done = env.act(action)
                        reward += r
                        if done:
                            break
                reward /= len(self.valid_list)
                tag = int(ep / self.validation_interval)
                print(f'Validate {tag}: {-reward:.2f}')
                torch.save(self.model.state_dict(), f'{self.model_pth}/{tag}.pth')

    def _learn(self, batch_size):
        self.optimizer.zero_grad()
        s, a, r, s_prime, done = self.memory.sample(batch_size)
        batch = Batch.from_data_list(s)
        batch_scatter_index, batch_scatter_norm = [], []
        for d in s:
            batch_scatter_index.append(d.scatter_index)
            batch_scatter_norm.append(d.scatter_norm)
        batch_scatter_norm = torch.cat(batch_scatter_norm, dim=0)
        self.model.eval()
        pred = self.model(batch, batch_scatter_index, batch_scatter_norm, a)

        targets = []
        for i in range(len(s_prime)):
            if done[i]:
                targets.append(0)
            else:
                sp = s_prime[i]
                q_value = self.target_model.inference(sp, sp.scatter_index, sp.scatter_norm, sp.action_space)
                targets.append(q_value.max().item())
        targets = torch.tensor(r, dtype=torch.float32, device=self.device) + self.gamma * torch.tensor(targets, dtype=torch.float32, device=self.device)
        targets.unsqueeze_(1)

        self.model.train()
        loss = F.mse_loss(pred, targets)
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
        self.optimizer.step()
        self._soft_update()
        return loss.item()

    def _soft_update(self, tau=.001):
        for t_param, param in zip(self.target_model.parameters(), self.model.parameters()):
            new_param = tau * param.data + (1.0 - tau) * t_param.data
            t_param.data.copy_(new_param)
