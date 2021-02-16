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

Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state', 'log_prob', 'done'))


class ReplayMemory(object):

    def __init__(self, configs, capacity=100000):
        self.capacity = capacity
        self.configs = configs
        self.memory = []
        self.position = 0

    def push(self, *args):
        """전환 저장"""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[int(self.position)] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def batch(self):
        transitions = self.memory
        batch = Transition(*zip(*transitions))

        # 최종 상태가 아닌 마스크를 계산하고 배치 요소를 연결합니다.
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=self.configs['device'], dtype=torch.bool)

        #non_final_mask.reshape(self.configs['batch_size'], 1)
        # non_final_next_states = torch.tensor([s for s in batch.next_state
        #                                       if s is not None]).reshape(-1, 1)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None], dim=0)
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action, dim=0)  # 안쓰지만 배치함
        # reward_batch = torch.cat(torch.tensor(batch.reward, dim=0)
        reward_batch = torch.tensor(batch.reward)
        next_state_batch = torch.cat(batch.state)
        log_prob_batch = torch.cat(batch.log_prob)
        done_batch = torch.tensor(batch.done)

        return state_batch, action_batch, reward_batch, next_state_batch, log_prob_batch, done_batch

    def __len__(self):
        return len(self.memory)

    def clear_memory(self):
        del self.memory[:]
        self.position = 0


configs = {
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
configs['device'] = torch.device('cpu')
configs['action_space'] = 2*len(configs['tl_rl_list'])
configs['action_size'] = 1*len(configs['tl_rl_list'])
configs['state_space'] = 4*len(configs['tl_rl_list'])


DEFAULT_CONFIG = {
    'gamma': 0.99,
    'lr': 0.002,
    'decay_rate': 0.98,
    'actior_lr': 0.001,
    'critic_lr': 0.001,
    'actor_layers': [30, 30],
    'critic_layers': [30, 30],
    'k_epochs': 4,
    'eps_clip': 0.2,
    'lr_decay_rate': 0.98,
    'gae': 0.95

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

    def update_hyperparams(self, epoch):
        '''
        상속을 위한 함수
        '''
        raise NotImplementedError

    def update_tensorboard(self, writer, epoch):
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


class Net(nn.Module):
    def __init__(self, configs):
        super(Net, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(configs['state_space'], configs['actor_layers'][0]),
            nn.Tanh(),
            nn.Linear(configs['actor_layers'][0], configs['actor_layers'][0]),
            nn.Tanh(),
            nn.Linear(configs['actor_layers'][1], configs['action_space']),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(configs['state_space'], configs['critic_layers'][0]),
            nn.Tanh(),
            nn.Linear(configs['critic_layers'][0],
                      configs['critic_layers'][1]),
            nn.Tanh(),
            # Q value 는 1개, so softmax 사용 x
            nn.Linear(configs['critic_layers'][1], 1),
        )

    def forward(self):
        raise NotImplementedError


class Trainer(RLAlgorithm):
    def __init__(self, configs):
        self.configs = merge_dict(configs, DEFAULT_CONFIG)
        self.model = Net(self.configs).to(configs['device'])
        self.experience_replay = ReplayMemory(configs)
        self.optimizer =\
            torch.optim.Adam(self.model.parameters(), lr=self.configs['lr'])
        self.eps_clip = self.configs['eps_clip']
        self.lr = self.configs['lr']
        self.lr_decay_rate = self.configs['lr_decay_rate']
        self.criterion = nn.MSELoss()
        self.running_loss = 0
        self.gae = self.configs['gae']

    def get_action(self, state):  # old로 계산
        probability = self.model.actor(state)
        distributions = Categorical(probability)
        action = distributions.sample()
        # print("state:", state)
        self.log_probs = distributions.log_prob(
            action)  # 이거 의미 자신의 action을 선택한 것의 확률
        # print("action:", action)

        return action  # item으로 step에 적용

    def save_replay(self, state, action, reward, next_state, done):
        self.experience_replay.push(
            state, action, reward, next_state, self.log_probs, done)

    def evaluate(self, state, action):  # 현재로 계산
        action_probs = self.model.actor(state)
        distributions = Categorical(action_probs)

        action_logprobs = distributions.log_prob(action)
        distributions_entropy = distributions.entropy()
        state_value = self.model.critic(state)
        return action_logprobs, torch.squeeze(state_value), distributions_entropy

    def update(self):
        state, action, reward, next_state, log_probs, done = self.experience_replay.batch()
        self.experience_replay.clear_memory()

        for _ in range(self.configs['k_epochs']):
            td_target = reward + \
                self.configs['gamma']*self.model.critic(next_state)
            delta = (td_target-self.model.critic(state)).detach()
            print(delta)
            advantage = 0.0
            for delta_t in reversed(range(delta.size()[1])):
                print(delta_t)
                advantage = self.configs['gamma'] * \
                    self.gae*advantage+delta[delta_t]
                advantage_list.insert(0, advantage)

            advantage = torch.cat(advantage_list, dtype=torch.float)
            action_probs = self.model.actor(state)
            distributions = Categorical(action_probs)
            action_logprobs = distributions.log_prob(action)
            ratio = torch.exp(action_logprobs-log_probs)

            surr1 = ratio*advantage
            surr2 = torch.clamp(ratio, 1-self.eps_clip,
                                1+self.eps_clip)*advantage
            loss = -torch.min(surr1, surr2) + \
                f.criterion(self.model.critic(state), td_target.detach())

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

    def update_hyperparams(self, epoch):

        # decay learning rate
        if self.lr > 0.01*self.configs['lr']:
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


# INITIALIZATION
evol = []

number_of_episodes = 5000

step_size_initial = 1
step_size_decay = 1
update_timesteps = 2000
env = gym.make('CartPole-v0')
step_size = step_size_initial
learner = Trainer(configs)
t = 0
reward = 0
for e in range(number_of_episodes):
    state = env.reset()
    t = 0
    done = False
    state = torch.from_numpy(state).reshape(1, -1).float()
    Return = 0
    while not done:
        t += 1
        action = learner.get_action(state)
        next_state, reward, done, _ = env.step(action.item())
        # print(state, action, reward, next_state, done)
        next_state = torch.from_numpy(next_state).reshape(1, -1).float()
        learner.save_replay(state, action, reward, next_state, done)

        state = next_state
        Return += reward
        if done:
            break

    learner.update()
    # print("{} {} {} {}".format(state, action, reward, next_state))
    if e % 50 == 0:
        learner.update_hyperparams(e)

    print('Episode ' + str(e) + ' ended in ' +
          str(t) + ' time steps'+'reward: ', str(Return))
