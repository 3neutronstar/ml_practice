import torch
import gc
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import random
from collections import namedtuple
import gym
import numpy as np
configs = {
    'action_space': 1,
    'gamma': 0.99,
    'tau': 0.95,
    'fc': [20, 30],
    'experience_replay_size': 1e5,
    'device': 'cpu',
    'batch_size': 32,
    'actor_lr': 0.001,
    'critic_lr': 0.0001,
}
WEIGHTS_FINAL_INIT = 3e-3
BIAS_FINAL_INIT = 3e-4


Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state', 'done'))


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


class Actor(nn.Module):
    def __init__(self, input_size, output_size, configs):
        super(Actor, self).__init__()
        self.configs = configs
        self.action_space = self.configs['action_space']
        self.fc1 = nn.Linear(input_size, self.configs['fc'][0])
        self.ln1 = nn.LayerNorm(self.configs['fc'][0])

        self.fc2 = nn.Linear(self.configs['fc'][0], self.configs['fc'][1])
        self.ln2 = nn.LayerNorm(self.configs['fc'][1])

        self.mu = nn.Linear(self.configs['fc'][1], output_size)

        nn.init.uniform_(self.mu.weight, -WEIGHTS_FINAL_INIT,
                         WEIGHTS_FINAL_INIT)
        nn.init.uniform_(self.mu.bias, -BIAS_FINAL_INIT, BIAS_FINAL_INIT)

    def forward(self, inputs):
        x = inputs
        x = self.fc1(x)
        x = self.ln1(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = self.ln2(x)
        x = F.relu(x)

        mu = torch.tanh(self.mu(x))
        return mu


class Critic(nn.Module):
    def __init__(self, input_size, output_size, configs):
        super(Critic, self).__init__()
        self.configs = configs
        self.action_space = self.configs['action_space']

        self.fc1 = nn.Linear(input_size, self.configs['fc'][0])
        self.ln1 = nn.LayerNorm(self.configs['fc'][0])

        self.fc2 = nn.Linear(self.configs['fc'][0]+1, self.configs['fc'][1])
        self.ln2 = nn.LayerNorm(self.configs['fc'][1])

        self.Value = nn.Linear(self.configs['fc'][1], output_size)

        nn.init.uniform_(self.Value.weight, -WEIGHTS_FINAL_INIT,
                         WEIGHTS_FINAL_INIT)
        nn.init.uniform_(self.Value.bias, -BIAS_FINAL_INIT, BIAS_FINAL_INIT)

    def forward(self, inputs, actions):
        x = inputs
        x = self.fc1(x)
        x = self.ln1(x)
        x = F.relu(x)
        x = torch.cat((x, actions), dim=1)
        x = self.fc2(x)
        x = self.ln2(x)
        x = F.relu(x)

        V = self.Value(x)
        return V


class OrnsteinUhlenbeckActionNoise:
    def __init__(self, mu, sigma, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def noise(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt \
            + self.sigma * np.sqrt(self.dt) * \
            np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(
            self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)


class DDPG(object):
    def __init__(self, configs):
        self.configs = configs
        self.gamma = self.configs['gamma']
        self.tau = self.configs['tau']
        # self.action_space = configs['action_space']
        # self.state_space = configs['state_space']
        self.action_space = 1
        self.state_space = 3
        self.actor = Actor(self.state_space, self.action_space,
                           self.configs).to(self.configs['device'])
        self.actor_target = Actor(self.state_space, self.action_space,
                                  self.configs).to(self.configs['device'])
        self.critic = Critic(self.state_space, self.action_space,
                             self.configs).to(self.configs['device'])
        self.critic_target = Critic(self.state_space, self.action_space,
                                    self.configs).to(self.configs['device'])

        # initializing with hard update
        self.hard_update(self.actor_target, self.actor)
        self.hard_update(self.critic_target, self.critic)
        # batch
        self.memory = ReplayMemory(self.configs['experience_replay_size'])

        # optim
        self.actor_optim = optim.Adam(
            self.actor.parameters(), lr=self.configs['actor_lr'])
        self.critic_optim = optim.Adam(
            self.critic.parameters(), lr=self.configs['critic_lr'])

        # noise
        self.action_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(
            self.action_space), sigma=0.2*np.ones(self.action_space))

        # clamp
        self.action_space_high = self.configs['action_space'].high[0]
        self.action_space_low = self.configs['action_space'].low[0]

    def get_action(self, state):
        self.actor.eval()
        mu = self.actor(state.float())
        self.actor.train()
        mu = mu.data

        if self.action_noise is not None:
            noise = torch.Tensor(self.action_noise.noise()).to(
                self.configs['device'])
            mu += noise

        mu = mu.clamp(self.action_space_high, self.action_space_low)
        return mu

    def update(self):
        if len(self.memory) <= self.configs['batch_size']:
            return
        transitions = self.memory.sample(self.configs['batch_size'])
        batch = Transition(*zip(*transitions))

        state_batch = torch.cat(batch.state).to(self.configs['device'])
        action_batch = torch.cat(batch.action).to(self.configs['device'])
        reward_batch = torch.cat(batch.reward).to(self.configs['device'])
        done_batch = torch.cat(batch.done).to(self.configs['device'])
        next_state_batch = torch.cat(
            batch.next_state).to(self.configs['device'])

        # get action and the state value from each target
        next_action_batch = self.actor_target(next_state_batch)
        next_state_action_values = self.critic_target(
            next_state_batch, next_action_batch.detach())

        # calc target
        reward_batch = reward_batch.unsqueeze(1)
        done_batch = done_batch.unsqueeze(1)
        expected_values = reward_batch + \
            (~done_batch) * \
            self.configs['gamma']*next_state_action_values

        # critic network update
        self.critic_optim.zero_grad()
        state_action_batch = self.critic(state_batch, action_batch)
        value_loss = F.mse_loss(state_action_batch, expected_values.detach())
        value_loss.backward()
        self.critic_optim.step()

        # actor network update
        self.actor_optim.zero_grad()
        policy_loss = -self.critic(state_batch, self.actor(state_batch))
        policy_loss = policy_loss.mean()
        policy_loss.backward()
        self.actor_optim.step()

        # update target
        self.soft_update(self.actor_target, self.actor)
        self.soft_update(self.critic_target, self.critic)

        return value_loss.item(), policy_loss.item()

    def soft_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau)

    def hard_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

    def save_replay(self, state, action,  reward, next_state, done):
        reward = torch.tensor(reward, device=self.configs['device']).view(1)
        done = torch.tensor(done, device=self.configs['device']).view(1)
        self.memory.push(state, action, reward, next_state, done)

    def target_update(self):
        self.hard_update(self.actor_target, self.actor)
        self.hard_update(self.critic_target, self.critic)


number_of_episodes = 5000

step_size_initial = 1
step_size_decay = 1

# INITIALIZATION

evol = []
env = gym.make('Pendulum-v0')
configs['action_space'] = env.action_space
print(env.action_space)
print(env.observation_space)
step_size = step_size_initial
learner = DDPG(configs)
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

        next_state, reward, done, _ = env.step(action)
        next_state = torch.from_numpy(next_state).reshape(1, -1)
        # print("{} {} {} {}".format(state, action, reward, next_state))
        learner.save_replay(state, action, reward, next_state, done)
        loss = learner.update()
        state = next_state
        Return += reward
    # learner.update_hyperparams(e)
    # if e % 10 == 0:
    #     learner.target_update()

    print('Episode ' + str(e) + ' ended in ' +
          str(t) + ' time steps'+'reward: ', str(Return))
    print("loss: {}".format(loss))
