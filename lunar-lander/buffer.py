import torch
from collections import deque, namedtuple
import random

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayBuffer(object):
    '''Replay buffer that stores online (s, a, r, s', d) transitions for training.'''

    def __init__(self, maxsize=100000):
        '''Initialize the buffer using the given parameters.'''
        self.buffer = deque(maxlen = maxsize)

    def __len__(self):
        '''Return the length of the buffer (i.e. the number of transitions).'''
        return len(self.buffer)

    def add_experience(self, state, action, reward, next_state, done):
        '''Add (s, a, r, s', d) to the buffer.'''
        self.buffer.append(Transition(state, action, reward, next_state, done))

    def sample(self, batch_size):
        '''Samples 'batch_size' transitions from the buffer and returns a tuple of torch
        tensors representing the states, actions, rewards, next states, and terminal signals.'''
        transitions = random.sample(self.buffer, batch_size)
        states = torch.stack([t.state for t in transitions])
        actions = torch.stack([t.action for t in transitions])
        rewards = torch.tensor([t.reward for t in transitions], dtype=torch.float32)
        next_states = torch.stack([t.next_state for t in transitions])
        dones = torch.tensor([t.done for t in transitions], dtype=torch.float32)
        return states, actions, rewards, next_states, dones
