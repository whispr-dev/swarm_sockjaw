import numpy as np

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

# Example usage
num_nodes = 5
connectivity_prob = 0.5
env = NetworkEnvironment(num_nodes, connectivity_prob)
print("Adjacency Matrix:")
print(env.adj_matrix)
for i in range(num_nodes):
    print(f"Neighbors of node {i}: {env.get_neighbors(i)}")

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
            print(f"Agent moved to node {action}")
        else:
            print("Invalid action! Agent stays in the same node.")


# Example usage
agent = Agent(env)
print(f"Initial state: {agent.get_state()}")
agent.take_action(2)  # Move to node 2
agent.take_action(4)  # Try to move to node 4, which is not a neighbor


import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


import random
from collections import namedtuple

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class DQN:
    def __init__(self, env, q_network, optimizer, gamma, batch_size, replay_buffer_capacity):
        self.env = env
        self.q_network = q_network
        self.optimizer = optimizer
        self.gamma = gamma
        self.batch_size = batch_size
        self.replay_buffer_capacity = replay_buffer_capacity
        self.replay_buffer = []

    def select_action(self, state, epsilon):
        if random.random() < epsilon:
            return random.choice(self.env.get_neighbors(state))
        else:
            with torch.no_grad():
                q_values = self.q_network(torch.tensor(state, dtype=torch.float32))
                return q_values.argmax().item()

    def update_q_network(self, transitions):
        states, actions, next_states, rewards = zip(*transitions)
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        rewards = torch.tensor(rewards, dtype=torch.float32)

        q_values = self.q_network(states)
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        next_q_values = self.q_network(next_states).max(1)[0].detach()
        target_q_values = rewards + self.gamma * next_q_values

        loss = F.mse_loss(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()



    def train(self, num_episodes, epsilon_start, epsilon_end, epsilon_decay):
        epsilon = epsilon_start
        for episode in range(num_episodes):
            state = self.env.get_state()
            total_reward = 0
            while True:
                action = self.select_action(state, epsilon)
                next_state = action
                reward = 1 if action == self.env.num_nodes - 1 else 0
                self.replay_buffer.append(Transition(state, action, next_state, reward))
                if len(self.replay_buffer) > self.replay_buffer_capacity:
                    self.replay_buffer.pop(0)

                if len(self.replay_buffer) >= self.batch_size:
                    batch = random.sample(self.replay_buffer, self.batch_size)
                    self.update_q_network(batch)

                total_reward += reward
                state = next_state
                if action == self.env.num_nodes - 1:
                    break

            epsilon = max(epsilon_end, epsilon * epsilon_decay)
            print(f"Episode {episode+1}/{num_episodes}, Total Reward: {total_reward}, Epsilon: {epsilon}")


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
agent = DQN(env, q_network, optimizer, gamma, batch_size, replay_buffer_capacity)

# Train the agent
num_episodes = 100
epsilon_start = 1.0
epsilon_end = 0.1
epsilon_decay = 0.99
agent.train(num_episodes, epsilon_start, epsilon_end, epsilon_decay)
