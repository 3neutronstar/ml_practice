import torch
import torch.nn as nn
from torch.distributions import Categorical
import gym
import copy
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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


def merge_dict(d1, d2):
    merged = copy.deepcopy(d1)
    for key in d2.keys():
        if key in merged.keys():
            raise KeyError
        merged[key] = d2[key]
    return merged


class ActorCritic(nn.Module):
    def __init__(self, memory, configs):
        super(ActorCritic, self).__init__()
        self.memory = memory
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

    def act(self, state):
        state = torch.from_numpy(state).float().to(device)
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        action = dist.sample()

        self.memory.states.append(state)
        self.memory.actions.append(action)
        self.memory.logprobs.append(dist.log_prob(action))

        return action.item()

    def evaluate(self, state, action):
        action_probs = self.actor(state)
        dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()

        state_value = self.critic(state)

        return action_logprobs, torch.squeeze(state_value), dist_entropy


class PPO:
    def __init__(self, configs):
        self.memory = Memory()
        self.configs = merge_dict(configs, DEFAULT_CONFIG)
        self.gamma = self.configs['gamma']
        self.eps_clip = self.configs['eps_clip']
        self.lr = self.configs['lr']
        self.lr_decay_rate = self.configs['lr_decay_rate']
        self.K_epochs = self.configs['k_epochs']

        self.policy = ActorCritic(self.memory, self.configs).to(device)
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=self.lr, )
        self.policy_old = ActorCritic(self.memory, self.configs).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.criterion = nn.MSELoss()

    def update(self):
        # Monte Carlo estimate of state rewards:
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.memory.rewards), reversed(self.memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards:
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        # convert list to tensor
        old_states = torch.stack(self.memory.states).to(device).detach()
        old_actions = torch.stack(self.memory.actions).to(device).detach()
        old_logprobs = torch.stack(self.memory.logprobs).to(device).detach()

        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            # Evaluating old actions and values :
            logprobs, state_values, dist_entropy = self.policy.evaluate(
                old_states, old_actions)

            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss:
            # cumulative reward이므로 state value와 빼면 advantage가나옴
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip,
                                1+self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5 * \
                self.criterion(state_values, rewards) - 0.01*dist_entropy

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())

    def get_action(self, state):
        action = self.policy_old.act(state)
        return action


def main():
    ############## Hyperparameters ##############
    env_name = "CartPole-v0"
    # creating environment
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = 2
    render = False
    solved_reward = 230         # stop training if avg_reward > solved_reward
    log_interval = 20           # print avg reward in the interval
    max_episodes = 50000        # max training episodes
    max_timesteps = 300         # max timesteps in one episode
    n_latent_var = 64           # number of variables in hidden layer
    update_timestep = 2000      # update policy every n timesteps
    lr = 0.002
    betas = (0.9, 0.999)
    gamma = 0.99                # discount factor
    K_epochs = 4                # update policy for K epochs
    eps_clip = 0.2              # clip parameter for PPO
    random_seed = None
    #############################################

    if random_seed:
        torch.manual_seed(random_seed)
        env.seed(random_seed)

    ppo = PPO(configs)
    print(lr, betas)

    # logging variables
    running_reward = 0
    avg_length = 0
    timestep = 0
    EPISODE = 10
    # training loop
    for i_episode in range(1, max_episodes+1):
        state = env.reset()
        for t in range(max_timesteps):

            # Running policy_old:
            action = ppo.get_action(state)
            state, reward, done, _ = env.step(action)

            # Saving reward and is_terminal:
            ppo.memory.rewards.append(reward)
            ppo.memory.is_terminals.append(done)

            # update if its time

            running_reward += reward
            if render:
                env.render()
            if done:
                break

        if i_episode % EPISODE == 0:
            ppo.update()
            ppo.memory.clear_memory()
            timestep = 0
        avg_length += t

        # stop training if avg_reward > solved_reward
        if running_reward > (log_interval*solved_reward):
            print("########## Solved! ##########")
            torch.save(ppo.policy.state_dict(),
                       './PPO_{}.pth'.format(env_name))
            break

        # logging
        if i_episode % log_interval == 0:
            avg_length = int(avg_length/log_interval)
            running_reward = int((running_reward/log_interval))

            print('Episode {} \t avg length: {} \t reward: {}'.format(
                i_episode, avg_length, running_reward))
            running_reward = 0
            avg_length = 0


if __name__ == '__main__':
    main()
