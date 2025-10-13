"""
Base model components used across different experiments.
Contains the DQN and ReplayMemory implementations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple
import random
from torch.autograd import Variable

# Define if CUDA is available
USE_CUDA = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if USE_CUDA else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if USE_CUDA else torch.ByteTensor
Tensor = FloatTensor

# Define the Transition tuple for memory storage
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory:
    """
    Replay Memory for storing past transitions.
    Used for experience replay in DQN training.
    """
    def __init__(self, capacity):
        """Initialize replay memory with given capacity."""
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Save a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """Sample a batch of transitions."""
        return random.sample(self.memory, batch_size)

    def __len__(self):
        """Return the current size of memory."""
        return len(self.memory)

class BaseDQN(nn.Module):
    """
    Base DQN class that implements common functionality.
    Specific architectures should inherit from this.
    """
    def __init__(self):
        super(BaseDQN, self).__init__()
        
    def init_hidden(self, size):
        """Initialize hidden state for RNN-based models."""
        return (Variable(torch.randn((1, size, 64))),
                Variable(torch.randn((1, size, 64))))

class LSTMDQN(BaseDQN):
    """LSTM-based DQN architecture."""
    def __init__(self, n_actions):
        super(LSTMDQN, self).__init__()
        self.lstm1 = nn.LSTM(5, 64)
        self.lin1 = nn.Linear(64, 256)
        self.lin2 = nn.Linear(256, 64)
        self.lin3 = nn.Linear(64, 32)
        self.lin4 = nn.Linear(32, 16)
        self.head = nn.Linear(16, n_actions)
        self.hidden = self.init_hidden(100)
        self.fixhidden = self.init_hidden(100)

    def forward(self, x):
        x = x.view(2, -1, 5)
        self.hidden = self.init_hidden(x.size(1))
        nperiod = x.size(0)
        lstm_out, self.hidden = self.lstm1(x, self.hidden)
        x = torch.relu(lstm_out[nperiod-1])
        x = torch.relu(self.lin1(x))
        x = torch.relu(self.lin2(x))
        x = torch.relu(self.lin3(x))
        x = torch.relu(self.lin4(x))
        return self.head(x.view(x.size(0), -1))

class FCNNDQN(BaseDQN):
    """Fully Connected Neural Network DQN architecture."""
    def __init__(self, n_actions):
        super(FCNNDQN, self).__init__()
        self.lin1 = nn.Linear(7, 256)
        self.lin1c = nn.Linear(256, 128)
        self.lin2 = nn.Linear(128, 64)
        self.lin3 = nn.Linear(64, 32)
        self.lin4 = nn.Linear(32, 16)
        self.head = nn.Linear(16, n_actions)

    def forward(self, x):
        x = torch.relu(self.lin1(x))
        x = torch.relu(self.lin1c(x))
        x = torch.relu(self.lin2(x))
        x = torch.relu(self.lin3(x))
        x = torch.relu(self.lin4(x))
        return self.head(x.view(x.size(0), -1))
