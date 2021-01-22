import gym  # gym space 기반
import torch
from torch.distributions import Categorical
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

learning_rate = 0.0001
gamma = 0.99


class Policy(nn.Module):
    def __init__(self, learning_rate, gamma):
        super(Policy, self).__init__()
        self.data = []
        self.gamma = gamma
        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, 2)
        # policy optimizing based
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=0)
        return x

    def put_data(self, item):
        self.data.append(item)  # list append

    def train_net(self):
        reward = 0
        self.optimizer.zero_grad()  # reset the gradient
        for r, prob in self.data[::-1]:  # from the last data
            reward = r+self.gamma*reward  # decreasing
            loss = -torch.log(prob)*reward
            loss.backward()
        self.optimizer.step()
        self.data = []  # make empty


def main():
    env = gym.make('CartPole-v1')  # environment setting
    policy = Policy(learning_rate, gamma)
    score = 0.0
    print_interval = 20
    total_episodes = 10000
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for num_episodes in range(total_episodes):
        state = env.reset()
        done = False

        while not done:
            prob = policy(torch.from_numpy(state).float().to(
                device))  # get the probability from policy
            m = Categorical(prob)  # probability distribution
            action = m.sample()  # sample the action from the distribution

            # return the data from the environment that is led by sampled action
            next_state, reward, done, info = env.step(action.item())
            # save the data for the Monte Carlo method at the end of episode
            policy.put_data((reward, prob[action]))
            state = next_state
            score += reward
            # env.render()  # show all the results

        policy.train_net()  # learning from the saving data in the episodes
        if num_episodes % print_interval == 0 and num_episodes != 0:
            print("num of episodes: {}, avg_score: {}".format(
                num_episodes, score/print_interval))
            score = 0.0

    env.close()


if __name__ == '__main__':
    main()
