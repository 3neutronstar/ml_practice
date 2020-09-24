# package import
import numpy as np
import matplotlib.pyplot as pyplot
%matplotlib inline
import gym
import torch

ENV='CartPole-v0' # task name
GAMMA=0.99 # Discount Rate
MAX_STEPS=200 #  max steps per 1 epsiode
NUM_EPISODES=1000 # max episodes
NUM_PROCESSES=32 # the number of environments that activate simulataneously
NUM_ADVANCED_STEP=5 # the number of steps that Advantage Learning when calculate total reward(return)

# Hyperparameter that uses calculate loss function in A2C
value_loss_coef=0.5
entropy_coef=0.01
max_grad_norm=0.5

class RolloutStorage(object):
    '''Memory class that uses in advantage learning'''
    def __init__(slef,num_steps,num_processes,obs_shape):
        self.observations=torch.zeros(num_steps+1,num_processes,4)
        self.masks=torch.zeros(num_steps+1,num_processes,1)
        
