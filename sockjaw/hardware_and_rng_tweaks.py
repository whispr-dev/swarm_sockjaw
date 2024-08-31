import numpy as np
import ctypes

# Load the shared library
rng_lib = ctypes.CDLL('./rng.so')

class NetworkEnvironment:
    def __init__(self, num_nodes, connectivity_prob):
        self.num_nodes = num_nodes
        self.connectivity_prob = connectivity_prob
        self.adj_matrix = self._generate_adj_matrix()

    def _generate_adj_matrix(self):
        adj_matrix = np.zeros((self.num_nodes, self.num_nodes))
        for i in range(self.num_nodes):
            for j in range(i+1, self.num_nodes):
                if rng_lib.random() < self.connectivity_prob:  # Use RNG from shared library
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
        self.state = rng_lib.random_int(0, env.num_nodes)  # Use RNG from shared library for initial state

    def get_state(self):
        return self.state

    def take_action(self, action):
        neighbors = self.env.get_neighbors(self.state)
        if action in neighbors:
            self.state = action
            print(f"Agent moved to node {action}")
        else:
            print("Invalid action! Agent stays in the same node.")

# Rest of the code remains the same...
