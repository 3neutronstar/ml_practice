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
    'lr': 0.01,
    'decay_rate': 0.98,
    'actior_lr': 0.001,
    'critic_lr': 0.001,

}
MAX_STEPS = 200  # 1에피소드 당 최대 단계 수
NUM_EPISODES = 1000  # 최대 에피소드 수

NUM_PROCESSES = 32  # 동시 실행 환경 수
NUM_ADVANCED_STEP = 5  # 총 보상을 계산할 때 Advantage 학습을 할 단계 수
# A2C 손실함수 계산에 사용되는 상수
value_loss_coef = 0.5
entropy_coeff = 0.01
max_grad_norm = 0.5


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


class RolloutStorage(object):
    def __init__(self, num_steps, num_processes, state_space, configs):
        self.states = torch.zeros(num_steps+1, num_processes, state_space)
        self.masks = torch.ones(num_steps+1, num_processes, 1)
        self.rewards = torch.zeros(num_steps, num_processes, 1)
        self.actions = torch.zeros(num_steps, num_processes, 1).long()
        self.gamma = configs['gamma']
        self.returns = torch.zeros(num_steps+1, num_processes, 1)
        self.index = 0

    def insert(self, state, action, reward, mask):
        self.states[self.index+1].copy_(state)
        self.masks[self.index+1].copy_(mask)
        self.actions[self.index].copy_(action)
        self.rewards[self.index].copy_(reward)

        self.index = (self.index+1) % NUM_ADVANCED_STEP  # index값 업데이트

    def after_update(self):
        self.states[0].copy_(self.states[-1])
        self.masks[0].copy_(self.masks[-1])

    def compute_returns(self, next_value):
        self.returns[-1] = next_value
        for ad_step in reversed(range(self.rewards.size(0))):
            self.returns[ad_step] = self.gamma*self.returns[ad_step+1] * \
                self.masks[ad_step+1]+self.rewards[ad_step]


class Net(nn.Module):
    def __init__(self, configs):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(configs['state_space'], 32)
        self.fc2 = nn.Linear(32, 32)
        self.actor = nn.Linear(32, configs['action_space'])
        self.critic = nn.Linear(32, 1)  # Qvalue
        self.running_loss = 0

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        critic_output = self.critic(x)
        actor_output = self.actor(x)
        return critic_output, actor_output

    def act(self, x):
        _, actor_output = self(x)  # forward
        # dim=1 이면 같은 행(행동 종류에 대해서 sofmax)
        action_prob = F.softmax(actor_output, dim=1)
        action = action_prob.multinomial(num_samples=1)
        #action = Categorical(action_prob).sample()
        return action

    def get_state_value(self, x):
        state_value, _ = self(x)
        return state_value

    def evaluate_action(self, x, actions):  # x가 process수 만큼 들어옴
        state_value, actor_output = self(x)
        log_prob = F.log_softmax(actor_output, dim=1)
        # batch중에서 자신이 선택했던 action의 위치의 값을 반환Q(s,a')
        action_log_prob = log_prob.gather(1, actions)

        probs = F.softmax(actor_output, dim=1)
        entropy = -(log_prob*probs).sum(-1).mean()  # 전체 process수 의 평균
        return state_value, action_log_prob, entropy


class Trainer(RLAlgorithm):
    def __init__(self, configs):
        self.configs = merge_dict(configs, DEFAULT_CONFIG)
        self.actor_critic = Net(configs)
        self.actor_critic.train()
        self.gamma = self.configs['gamma']
        self.lr = self.configs['lr']
        self.optimizer = torch.optim.Adam(
            self.actor_critic.parameters(), lr=self.lr)
        self.action_size = self.configs['action_size']
        self.running_loss = 0
        self.state_space = self.configs['state_space']
        self.rollouts = RolloutStorage(
            NUM_ADVANCED_STEP, NUM_PROCESSES, self.state_space, self.configs)

    def get_action(self, state):
        action = self.actor_critic.act(state)
        return action

    def update(self):
        state_space = self.rollouts.states.size()[2:]
        num_steps = NUM_ADVANCED_STEP
        num_processes = NUM_PROCESSES

        state_values, action_log_prob, entropy = self.actor_critic.evaluate_action(
            self.rollouts.states[:-1].view(-1, self.state_space), self.rollouts.actions.view(-1, self.action_size))

        state_values = state_values.view(num_steps, num_processes, 1)
        action_log_prob = action_log_prob.view(num_steps, num_processes, 1)

        advantages = self.rollouts.returns[:-1] - state_values

        value_loss = advantages.pow(2).mean()  # MSE
        action_gain = (action_log_prob*advantages.detach()
                       ).mean()  # detach로 상수화
        total_loss = (value_loss*value_loss_coef - action_gain -
                      entropy*entropy_coeff)  # cross entropy

        self.optimizer.zero_grad()

        total_loss.backward()
        # 경사 clipping, 너무 한번에 크게 변화하지 않도록
        nn.utils.clip_grad_norm_(self.actor_critic.parameters(), max_grad_norm)
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


number_of_episodes = 5000

step_size_initial = 1
step_size_decay = 1

state_space = configs['state_space']
action_space = configs['action_space']

# INITIALIZATION

evol = []
env = [gym.make('CartPole-v0') for i in range(NUM_PROCESSES)]
step_size = step_size_initial
learner = Trainer(configs)

state = torch.zeros(NUM_PROCESSES, state_space)

Return = torch.zeros([NUM_PROCESSES, 1])
final_rewards = torch.zeros([NUM_PROCESSES, 1])
episode_rewards = torch.zeros([NUM_PROCESSES, 1])  # 현재 에피소드의 보상
state_torch = torch.zeros([NUM_PROCESSES, state_space])  # Numpy 배열
reward_torch = torch.zeros([NUM_PROCESSES, 1])  # Numpy 배열
done_torch = torch.zeros([NUM_PROCESSES, 1])  # Numpy 배열
each_step = torch.zeros(NUM_PROCESSES)  # 각 환경의 단계 수를 기록
episode = 0  # 환경 0의 에피소드 수

state = torch.tensor([env[i].reset() for i in range(NUM_PROCESSES)]).float()

# advanced 학습에 사용되는 객체 learner.rollouts 첫번째 상태에 현재 상태를 저장
learner.rollouts.states[0].copy_(state)  # 동시에 같은 상황 주고 시작

for j in range(number_of_episodes*NUM_PROCESSES):
    for step in range(NUM_ADVANCED_STEP):
        with torch.no_grad():
            actions = learner.get_action(
                learner.rollouts.states[step])

        for i in range(NUM_PROCESSES):
            s, r, d, _ = env[i].step(actions[i].item())
            state_torch[i], reward_torch[i], done_torch[i] = torch.from_numpy(
                s).float(), torch.tensor(r), torch.tensor(d)
            if done_torch[i]:
                if i == 0:
                    print('%d Episode: Finished after %d steps' % (
                        episode, each_step[i]+1))
                    episode += 1
                if each_step[i] < 195:
                    reward_torch[i] = -1.0
                else:
                    reward_torch[i] = 1.0

                each_step[i] = 0
                state_torch[i] = torch.tensor([env[i].reset()]).float()
            else:
                reward_torch[i] = 0.0
                each_step[i] += 1

        episode_rewards += reward_torch
        # done 이면 0 done이 아니면 mask를 1로
        masks = torch.FloatTensor([[0.0] if done_ else [1.0]
                                   for done_ in done_torch])

        # 마지막 에피소드의 총 보상을 업데이트
        final_rewards *= masks  # done이 false이면 1을 곱하고, true이면 0을 곱해 초기화
        # done이 false이면 0을 더하고, true이면 episode_rewards를 더해줌
        final_rewards += (1 - masks) * episode_rewards

        # 에피소드의 총보상을 업데이트
        episode_rewards *= masks  # done이 false인 에피소드의 mask는 1이므로 그대로, true이면 0이 됨

        # 현재 done이 true이면 모두 0으로
        state *= masks
        state = state_torch  # next_state가 state로 이전

        # 메모리 객체에 현 단계의 transition을 저장
        learner.rollouts.insert(state, actions, reward_torch, masks)

    # advanced 학습 for문 끝

    # advanced 학습 대상 중 마지막 단계의 상태로 예측하는 상태가치를 계산

    with torch.no_grad():
        next_value = learner.actor_critic.get_state_value(
            learner.rollouts.states[-1]).detach()
        # learner.rollouts.observations의 크기는 torch.Size([6, 16, 4])

    # 모든 단계의 할인총보상을 계산하고, rollouts의 변수 returns를 업데이트
    learner.rollouts.compute_returns(next_value)

    # 신경망 및 rollout 업데이트
    learner.update()
    learner.rollouts.after_update()

    # 환경 갯수를 넘어서는 횟수로 200단계를 버텨내면 성공
    if final_rewards.sum().numpy() >= NUM_PROCESSES:
        print('연속성공')
        break
