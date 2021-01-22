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
    'decay_rate':1000

}
configs['device']=torch.device('cpu')
configs['action_space'] = 2*len(configs['tl_rl_list'])
configs['action_size'] = 1*len(configs['tl_rl_list'])
configs['state_space'] = 4*len(configs['tl_rl_list'])


DEFAULT_CONFIG={
    'actior_lr':0.001,
    'critic_lr':0.001,
}


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



def merge_dict(d1, d2):
    merged = copy.deepcopy(d1)
    for key in d2.keys():
        if key in merged.keys():
            raise KeyError
        merged[key] = d2[key]
    return merged

class ActorNet(nn.Module):
    def __init__(self, configs):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(configs['state_space'], 40)
        self.fc2 = nn.Linear(40, 30)
        self.fc3 = nn.Linear(30, configs['action_space'])
        self.running_loss = 0

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=0)
        return x


class CriticNet(nn.Module):
    def __init__(self, configs):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(configs['state_space'], 40)
        self.fc2 = nn.Linear(40, 30)
        self.fc3 = nn.Linear(30, configs['action_space'])
        self.running_loss = 0

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=0)
        return x


class Trainer(RLAlgorithm):
    def __init__(self, configs):
        self.configs = merge_dict(configs, DEFAULT_CONFIG)
        self.configs=configs
        self.actor=ActorNet(configs)
        self.critic=CriticNet(configs)
        self.gamma = self.configs['gamma']
        self.lr=configs['lr']
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr)
        self.loss=nn.MSELoss()
        self.running_loss=0

    def get_action(self, state):
        prob=self.actor(state)
        m=Categorical(prob)
        action=m.sample()
        return action

    def update(self, done=False):
        self.optimizer.zero_grad()

        loss.backward()
        self.optimizer.step()
        
    
    
    def update_hyperparams(self, epoch):

        # decay learning rate
        if self.lr > 0.01*self.lr:
            self.lr = self.lr_decay_rate*self.lr

    def update_tensorboard(self, writer, epoch):
        writer.add_scalar('episode/loss', self.running_loss/self.configs['max_steps'],
                          self.configs['max_steps']*epoch)  # 1 epoch마다
        writer.add_scalar('hyperparameter/lr', self.lr,
                          self.configs['max_steps']*epoch)

        action_distribution = torch.cat(self.action, 0)
        writer.add_histogram('hist/episode/action_distribution', action_distribution,
                             epoch)  # 1 epoch마다
        self.action = tuple()
        # clear
        self.running_loss = 0

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
score=0
for epoch in range(number_of_episodes):
    state = env.reset()
    t = 0
    done = False
    state = torch.from_numpy(state).float().to(configs['device'])
    Return = 0
    while not done:
        t += 1
        action = learner.get_action(state)

        next_state, reward, done, _ = env.step(action.item())
        next_state = torch.from_numpy(next_state).float()
        if done:
            next_state = None
        
        
        state = next_state
        Return += reward
    learner.update()
    learner.update_hyperparams(epoch)
    score+=Return
    if epoch %10==0:
        print('Episode ' + str(epoch) + ' ended in ' +
            str(t) + ' time steps'+'reward: ', str(score/10))
        score=0
env.close()
