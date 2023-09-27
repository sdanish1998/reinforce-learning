import torch
from torch import nn
from utils import make_network
import numpy as np

class QNetwork(nn.Module):
    def __init__(self,
                 gamma,
                 state_dim,
                 action_dim,
                 hidden_sizes=[10, 10]):
        super().__init__()
        self.gamma = gamma
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_sizes = hidden_sizes

        # neural net architecture
        self.network = make_network(state_dim, action_dim, hidden_sizes)

    def forward(self, states):
        '''Returns the Q values for each action at each state.'''
        qs = self.network(states)
        return qs

    def get_max_q(self, states):
        '''Compute the max Q value for each state in the batch.'''
        qs = self.forward(states)
        max_qs, _ = torch.max(qs, 1)
        return max_qs

    def get_action(self, state, eps):
        '''Get the action at a given state according to an epsilon greedy method.'''
        if np.random.uniform() < eps:
            # Select a random action
            return p.random.randint(self.action_dim)
        else:
            # Get the Q-values for the state
            state = torch.FloatTensor(state).unsqueeze(0)
            qs = self.network(state)
            # Select the action with the highest Q-value
            _, action = torch.max(qs, 1)
            return action.item()

    @torch.no_grad()
    def get_targets(self, rewards, next_states, dones):
        '''Get the next Q function targets, as given by the Bellman optimality equation for Q functions.'''
        with torch.no_grad():
            # Compute the max Q values for the next states
            max_q = self.get_max_q(next_states)
            # Compute the targets for the Q function
            targets = rewards + (1-dones) * self.gamma * max_q
        return targets
