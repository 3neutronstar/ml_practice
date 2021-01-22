import torch
import os
import copy
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import gym
import torch
import numpy as np
import torch.optim as optim
import random
from collections import namedtuple
from copy import deepcopy
from torch.utils.tensorboard import SummaryWriter
configs = {
    'gamma': 0.99,
    'lr': 0.0001,
    'num_lanes': 2,
    'model': 'normal',
    'file_name': '4x4grid',
    'tl_rl_list': ['n_1_1'],
    'laneLength': 300.0,
    'num_cars': 1800,
    'flow_start': 0,
    'flow_end': 3000,
    'sim_start': 0,
    'max_steps': 3000,
    'num_epochs': 1000,

}
configs['device']=torch.device('cpu')
configs['action_space'] = 2*len(configs['tl_rl_list'])
configs['action_size'] = 1*len(configs['tl_rl_list'])
configs['state_space'] = 4*len(configs['tl_rl_list'])

class RLAlgorithm():
    def __init__(self, configs):
        super().__init__()
        self.configs = configs

    def get_action(self, state):
        '''
        return action (torch Tensor (1,action_space))
        상속을 위한 함수
        '''
        raise NotImplementedError

    def update_hyperparams(self,epoch):
        '''
        상속을 위한 함수
        '''
        raise NotImplementedError

    def update_tensorboard(self,writer,epoch):
        '''
        상속을 위한 함수
        '''
        raise NotImplementedError



Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """전환 저장"""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[int(self.position)] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


def merge_dict(d1, d2):
    merged = copy.deepcopy(d1)
    for key in d2.keys():
        if key in merged.keys():
            raise KeyError
        merged[key] = d2[key]
    return merged

class PolicyNet(nn.Module):
    def __init__(self, configs):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(configs['state_space'], 40)
        self.fc2 = nn.Linear(40, 30)
        self.fc3 = nn.Linear(30, configs['action_space'])
        self.running_loss = 0

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1)
        return x


class Trainer(RLAlgorithm):
    def __init__(self, configs):
        #self.configs = merge_dict(configs, DEFAULT_CONFIG)
        self.configs=configs
        self.model = PolicyNet(self.configs)
        self.gamma = self.configs['gamma']
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.configs['lr'])
        self.saved_log_probs = []
        self.rewards = []
        self.eps = torch.tensor(np.finfo(np.float32).eps.item())

    def get_action(self, state, reward):
        self.rewards.append(reward)
        state = state.float()
        print('state', state)
        probs = self.model(state)
        print('probs', probs)
        m = Categorical(probs)
        action = m.sample()
        print('action', action)
        self.saved_log_probs.append(m.log_prob(action))
        return action

    def get_loss(self):
        return self.running_loss

    def update(self, done=False):
        R = 0  # Return
        policy_loss = []
        returns = []
        for r in self.rewards[::-1]:  # end기준으로 최근 reward부터 옛날로
            R = r + self.gamma * R
            returns.insert(0, R)  # 앞으로 내용 삽입(실제로는 맨뒤부터 삽입해서 t=0까지 삽입)
            # 내용순서는 t=0,1,2,3,4,...T)

        returns = torch.tensor(returns, device=self.configs['device'])
        returns = (returns - returns.mean()) / (returns.std() + self.eps)

        for log_prob, R in zip(self.saved_log_probs, returns):
            policy_loss.append(-log_prob * R)

        self.optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()

        self.running_loss = policy_loss
        policy_loss.backward()
        self.optimizer.step()
        del self.rewards[:]
        del self.saved_log_probs[:]


number_of_episodes = 5000

step_size_initial = 1
step_size_decay = 1

# INITIALIZATION

evol = []
env = gym.make('CartPole-v0')
step_size = step_size_initial
learner = Trainer(configs)
t = 0
reward = 0
for epoch in range(number_of_episodes):
    state = env.reset()
    t = 0
    done = False
    state = torch.from_numpy(state).reshape(1, -1)
    Return = 0
    while not done:
        t += 1
        action = learner.get_action(state, reward)

        next_state, reward, done, _ = env.step(action.item())
        next_state = torch.from_numpy(next_state).reshape(1, -1)
        if done:
            next_state = None
        print("{} {} {} {}".format(state, action, reward, next_state))
        learner.update()
        state = next_state
        Return += reward
    learner.update_hyperparams(epoch)


    print('Episode ' + str(epoch) + ' ended in ' +
          str(t) + ' time steps'+'reward: ', str(Return))
