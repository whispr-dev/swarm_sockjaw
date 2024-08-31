import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import random
from collections import namedtuple

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Define the Fuzzy Logic Layer
class FuzzyLogicLayer(nn.Module):
    def __init__(self, input_size, output_size):
        super(FuzzyLogicLayer, self).__init__()
        self.weights = nn.Parameter(torch.randn(input_size, output_size))
        self.bias = nn.Parameter(torch.zeros(output_size))

    def forward(self, x):
        membership_values = torch.sigmoid(torch.mm(x, self.weights) + self.bias)
        return membership_values

# Define the GAN with Fuzzy Logic
class SimpleGAN(nn.Module):
    def __init__(self):
        super(SimpleGAN, self).__init__()
        self.fc1 = nn.Linear(100, 128)
        self.fuzzy_layer = FuzzyLogicLayer(128, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fuzzy_layer(x)
        x = self.fc2(x)
        return x

# Define Transition for Reinforcement Learning
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

# Define Network Environment for the jellyfish AI
class NetworkEnvironment:
    def __init__(self, num_nodes, connectivity_prob):
        self.num_nodes = num_nodes
        self.connectivity_prob = connectivity_prob
        self.adj_matrix = self._generate_adj_matrix()

    def _generate_adj_matrix(self):
        adj_matrix = np.zeros((self.num_nodes, self.num_nodes))
        for i in range(self.num_nodes):
            for j in range(i+1, self.num_nodes):
                if np.random.rand() < self.connectivity_prob:
                    adj_matrix[i, j] = 1
                    adj_matrix[j, i] = 1
        return adj_matrix

    def get_neighbors(self, node):
        return np.nonzero(self.adj_matrix[node])[0]

# Define the Jellyfish Agent
class JellyfishAgent:
    def __init__(self, env, gan_brain):
        self.env = env
        self.gan_brain = gan_brain
        self.state = np.random.randint(0, env.num_nodes)

    def get_state(self):
        return self.state

    def take_action(self):
        neighbors = self.env.get_neighbors(self.state)
        if len(neighbors) == 0:
            return self.state, 0  # Stay if no neighbors
        noise = torch.randn(1, 100).to(device)
        action_probs = self.gan_brain(noise).detach().cpu().numpy().flatten()
        action = neighbors[np.argmax(action_probs[:len(neighbors)])]
        self.state = action
        return action, 1  # Return the action and reward

# Define the GAN and Fuzzy Logic Controller
class GANController:
    def __init__(self, gan_brain):
        self.gan_brain = gan_brain

    def make_decision(self, state):
        noise = torch.randn(1, 100).to(device)
        decision = self.gan_brain(noise)
        return decision.argmax().item()

# Define the training loop for the jellyfish's brain
def train_jellyfish(agent, num_epochs):
    for epoch in range(num_epochs):
        state = agent.get_state()
        total_reward = 0
        while True:
            action, reward = agent.take_action()
            total_reward += reward
            state = action
            if reward == 1:  # If the agent has moved to a new node
                break
        print(f"Epoch {epoch+1}/{num_epochs}, Total Reward: {total_reward}")

# Initialize the environment and agent
num_nodes = 10
connectivity_prob = 0.5
env = NetworkEnvironment(num_nodes, connectivity_prob)

# Initialize the GAN brain
gan_brain = SimpleGAN().to(device)

# Initialize the Jellyfish Agent
jellyfish = JellyfishAgent(env, gan_brain)

# Train the Jellyfish Agent
num_epochs = 100
train_jellyfish(jellyfish, num_epochs)
