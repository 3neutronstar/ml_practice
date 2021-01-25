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

}


class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]


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


class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.dones = []

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.dones[:]


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
        self.memory = Memory()
        self.optimizer =\
            torch.optim.Adam(self.model.parameters(), lr=self.configs['lr'])
        self.model_old = Net(self.configs).to(configs['device'])
        self.model_old.load_state_dict(self.model.state_dict())
        self.eps_clip = self.configs['eps_clip']
        self.lr = self.configs['lr']
        self.lr_decay_rate = self.configs['lr_decay_rate']
        self.criterion = nn.MSELoss()
        self.running_loss = 0

    def get_action(self, state):  # old로 계산
        probability = self.model_old.actor(state)
        distributions = Categorical(probability)
        action = distributions.sample()
        # print("state:", state)

        self.memory.states.append(state)
        self.memory.actions.append(action)
        self.memory.logprobs.append(distributions.log_prob(action))  # 이거 의미
        # print("action:", action)

        return action  # item으로 step에 적용

    def evaluate(self, state, action):  # 현재로 계산
        action_probs = self.model.actor(state)
        distributions = Categorical(action_probs)

        action_logprobs = distributions.log_prob(action)
        distributions_entropy = distributions.entropy()
        state_value = self.model.critic(state)
        return action_logprobs, torch.squeeze(state_value), distributions_entropy

    def update(self):
        rewards = []
        discounted_reward = 0
        for reward, done in zip(reversed(self.memory.rewards), reversed(self.memory.dones)):
            if done:
                discounted_reward = 0
            discounted_reward = reward + \
                (self.configs['gamma']*discounted_reward)
            rewards.insert(0, discounted_reward)  # 앞으로 삽입
            print("discount reward", discounted_reward)

        # normalizing the reward
        rewards = torch.tensor(
            rewards, dtype=torch.float).to(configs['device'])
        rewards = (rewards-rewards.mean()) / (rewards.std()+1e-5)  # epsilon

        # list to tensor
        old_states = torch.stack(self.memory.states).to(
            configs['device']).detach()  # no grads by detach
        old_actions = torch.stack(self.memory.actions).to(
            configs['device']).detach()
        old_logprobs = torch.stack(self.memory.logprobs).to(
            configs['device']).detach()
        # print("old1:", old_states)
        # print("old2:", old_actions)
        # print("old3:", old_logprobs)
        for _ in range(self.configs['k_epochs']):
            # evaluate old actions and values
            logprobs, state_values, distributions_entropy = self.evaluate(
                old_states, old_actions)

            # find the ratio (pi_theta/pi_theta_old)
            ratios = torch.exp(logprobs-old_logprobs.detach())

            # surrogate loss
            advantages = rewards-state_values.detach()
            surr1 = ratios*advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip,
                                1+self.eps_clip)*advantages
            loss = -torch.min(surr1, surr2)+0.5*self.criterion(state_values,
                                                               rewards)-0.01*distributions_entropy
            # optimize K epochs
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
        self.model_old.load_state_dict(self.model.state_dict())

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
time_steps = 0
for e in range(number_of_episodes):
    state = env.reset()
    t = 0
    done = False
    state = torch.from_numpy(state).reshape(1, -1).float()
    Return = 0
    while not done:
        time_steps += 1
        t += 1
        action = learner.get_action(state)
        next_state, reward, done, _ = env.step(action.item())
        # print(state, action, reward, next_state, done)
        next_state = torch.from_numpy(next_state).reshape(1, -1).float()
        learner.memory.rewards.append(reward)
        learner.memory.dones.append(done)

        if time_steps % update_timesteps == 0:
            learner.update()
            learner.memory.clear_memory()
            time_steps = 0

        state = next_state
        Return += reward
        if done:
            break
        # print("{} {} {} {}".format(state, action, reward, next_state))
    if e % 50 == 0:
        learner.update_hyperparams(e)

    print('Episode ' + str(e) + ' ended in ' +
          str(t) + ' time steps'+'reward: ', str(Return))
