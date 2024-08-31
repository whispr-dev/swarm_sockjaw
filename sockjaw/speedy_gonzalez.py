import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from collections import namedtuple
import ctypes
import threading
import random
import os

# Get the current directory of the Python script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Specify the path to rng.so relative to the current directory
rng_lib_path = os.path.join(current_dir, "rng.so")

# Load the custom RNG library
rng_lib = ctypes.CDLL(rng_lib_path)


# Define custom RNG functions
rng64 = rng_lib.rng64
rng128 = rng_lib.rng128
jump = rng_lib.jump

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class NetworkEnvironment:
    def __init__(self, num_nodes, connectivity_prob):
        self.num_nodes = num_nodes
        self.connectivity_prob = connectivity_prob
        self.adj_matrix = self._generate_adj_matrix()

    def _generate_adj_matrix(self):
        adj_matrix = np.zeros((self.num_nodes, self.num_nodes))
        for i in range(self.num_nodes):
            for j in range(i+1, self.num_nodes):
                if rng64() < self.connectivity_prob:  # Use rng64 for connectivity probability
                    adj_matrix[i, j] = 1
                    adj_matrix[j, i] = 1
        return adj_matrix

    def get_neighbors(self, node):
        return np.nonzero(self.adj_matrix[node])[0]

class Agent:
    def __init__(self, env):
        self.env = env
        self.state = rng64() % env.num_nodes  # Initial state using rng64

    def get_state(self):
        return self.state

    def take_action(self, action):
        neighbors = self.env.get_neighbors(self.state)
        if action in neighbors:
            self.state = action
            return action, 1  # Return the action and reward
        else:
            return self.state, 0  # Stay in the same node and return zero reward

class QNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class DQN:
    def __init__(self, q_network, optimizer, gamma, batch_size, replay_buffer_capacity):
        self.q_network = q_network
        self.optimizer = optimizer
        self.gamma = gamma
        self.batch_size = batch_size
        self.replay_buffer_capacity = replay_buffer_capacity
        self.replay_buffer = []

    def select_action(self, state, epsilon):
        if rng64() < epsilon:  # Use rng64 for epsilon
            return rng64() % self.q_network.fc2.out_features  # Random action
        else:
            with torch.no_grad():
                q_values = self.q_network(torch.tensor(state, dtype=torch.float32))
                return q_values.argmax().item()

    def update_q_network(self):
        if len(self.replay_buffer) < self.batch_size:
            return  # Not enough samples for training

        transitions = random.sample(self.replay_buffer, self.batch_size)
        batch = Transition(*zip(*transitions))
        state_batch = torch.tensor(batch.state, dtype=torch.float32)
        action_batch = torch.tensor(batch.action)
        next_state_batch = torch.tensor(batch.next_state, dtype=torch.float32)
        reward_batch = torch.tensor(batch.reward, dtype=torch.float32)

        q_values = self.q_network(state_batch)
        q_values = q_values.gather(1, action_batch.unsqueeze(1)).squeeze(1)

        next_q_values = self.q_network(next_state_batch).max(1)[0].detach()
        target_q_values = reward_batch + self.gamma * next_q_values

        loss = F.mse_loss(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def simulate_epochs(self, agent, num_epochs):
        for _ in range(num_epochs):
            state = agent.get_state()
            total_reward = 0
            while True:
                action = agent.take_action(state)[0]  # Only need the action from take_action
                next_state, reward = agent.take_action(action)
                self.replay_buffer.append(Transition(state, action, next_state, reward))
                total_reward += reward
                state = next_state
                if reward == 1:  # If the goal state is reached
                    break

    def train(self, num_epochs, epsilon_start, epsilon_end, epsilon_decay):
        agent = Agent(env)  # Create the agent here
        epsilon = epsilon_start
        for epoch in range(num_epochs):
            self.simulate_epochs(agent, 1)  # Simulate one epoch
            self.update_q_network()
            epsilon = max(epsilon_end, epsilon * epsilon_decay)
            print(f"Epoch {epoch+1}/{num_epochs}, Epsilon: {epsilon}")

# Initialize environment
num_nodes = 5
connectivity_prob = 0.5
env = NetworkEnvironment(num_nodes, connectivity_prob)

# Initialize Q-network
input_size = num_nodes
hidden_size = 16
output_size = num_nodes
q_network = QNetwork(input_size, hidden_size, output_size)

# Initialize optimizer
learning_rate = 0.001
optimizer = torch.optim.Adam(q_network.parameters(), lr=learning_rate)

# Initialize DQN agent
gamma = 0.99
batch_size = 32
replay_buffer_capacity = 10000
dqn_agent = DQN(q_network, optimizer, gamma, batch_size, replay_buffer_capacity)

# Train the agent
num_epochs = 100

# Define parameter ranges
epsilon_start_range = (0.5, 1.0)
epsilon_end_range = (0.01, 0.1)
epsilon_decay_range = (0.95, 0.99)

# Define the number of parameter combinations to try
num_combinations = 10

# Initialize variables to store the best parameters and performance
best_parameters = None
best_performance = float('-inf')  # Initialize with negative infinity

# Iterate over the number of parameter combinations
for _ in range(num_combinations):
# Randomly sample values for epsilon_start, epsilon_end, and epsilon_decay
    epsilon_start = rng64() * (epsilon_start_range[1] - epsilon_start_range[0]) + epsilon_start_range[0]
    epsilon_end = rng64() * (epsilon_end_range[1] - epsilon_end_range[0]) + epsilon_end_range[0]
    epsilon_decay = rng64() * (epsilon_decay_range[1] - epsilon_decay_range[0]) + epsilon_decay_range[0]

    # Initialize and train the agent with the sampled parameters
    dqn_agent.train(num_epochs, epsilon_start, epsilon_end, epsilon_decay) 

    # Evaluate the performance of the trained agent
    # You can define your performance metric here, such as the average total reward
    performance = ...  # Calculate performance metric

    # Check if the current set of parameters gives better performance than the previous best
    if performance > best_performance:
        best_parameters = (epsilon_start, epsilon_end, epsilon_decay)
        best_performance = performance

# Print the best parameters and performance
print("Best Parameters:", best_parameters)
print("Best Performance:", best_performance)
