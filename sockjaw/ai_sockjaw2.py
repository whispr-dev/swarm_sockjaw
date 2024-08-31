import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple

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
                if np.random.rand() < self.connectivity_prob:
                    adj_matrix[i, j] = 1
                    adj_matrix[j, i] = 1
        return adj_matrix

    def get_neighbors(self, node):
        return np.nonzero(self.adj_matrix[node])[0]

class Agent:
    def __init__(self, env):
        self.env = env
        self.state = np.random.randint(0, env.num_nodes)  # Initial state is a random node

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
        if random.random() < epsilon:
            return random.randint(0, self.q_network.output_size - 1)  # Random action
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

    def simulate_episodes(self, agent, env, num_episodes):
        for _ in range(num_episodes):
            state = agent.get_state()
            total_reward = 0
            while True:
                action = agent.take_action(state)[0]  # Only need the action from take_action
                next_state, reward = agent.take_action(action)
                self.replay_buffer.append(Transition(state, action, next_state, reward))
                total_reward += reward
              


            epsilon = max(epsilon_end, epsilon * epsilon_decay)
            print(f"Episode {episode+1}/{num_episodes}, Total Reward: {total_reward}, Epsilon: {epsilon}")


# Initialize environment
num_nodes = 50
connectivity_prob = 5
env = NetworkEnvironment(num_nodes, connectivity_prob)

# Initialize Q-network
input_size = num_nodes
hidden_size = 64
output_size = num_nodes
q_network = QNetwork(input_size, hidden_size, output_size)

# Initialize optimizer
learning_rate = 0.001
optimizer = torch.optim.Adam(q_network.parameters(), lr=learning_rate)

# Initialize DQN agent
gamma = 0.99
batch_size = 32
replay_buffer_capacity = 10000
agent = DQN(env, q_network, optimizer, gamma, batch_size, replay_buffer_capacity)

# Train the agent
num_episodes = 500

import random

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
    epsilon_start = random.uniform(*epsilon_start_range)
    epsilon_end = random.uniform(*epsilon_end_range)
    epsilon_decay = random.uniform(*epsilon_decay_range)

    # Initialize and train the agent with the sampled parameters
    agent = DQN(env, q_network, optimizer, gamma, batch_size, replay_buffer_capacity)
    agent.train(num_episodes, epsilon_start, epsilon_end, epsilon_decay)

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
