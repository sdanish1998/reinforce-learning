import torch
import torch.nn as nn
import numpy as np

def initialize_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight.data)

def make_network(state_dim, action_dim, hidden_sizes):
    '''Initializes Q network.'''
    layers = []
    layers.append(nn.Linear(state_dim, hidden_sizes[0]))
    for i in range(1, len(hidden_sizes)):
        layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))

    layers.append(nn.ReLU())
    layers.append(nn.Linear(hidden_sizes[-1], action_dim))

    network = nn.Sequential(*layers).apply(initialize_weights)
    return network

@torch.no_grad()
def update_target(net, target_net, tau):
    '''Update the target parameters using a soft update given by the parameter tau.'''
    for target_param, param in zip(target_net.parameters(), net.parameters()):
        target_param.copy_((1 - tau) * target_param + tau * param)

def get_eps(eps_param, t):
    eps = eps_param ** t
    if eps <= 0.001: return 0.001
    return eps
