import numpy as np

class Rule30CA:
    def __init__(self, shape, initial_state):
        self.shape = shape
        self.grid = initial_state.copy()
    
    def apply_rule(self):
        next_state = np.zeros_like(self.grid)
        for i in range(1, len(self.grid) - 1):
            left = self.grid[i - 1]
            center = self.grid[i]
            right = self.grid[i + 1]
            # Rule 30: 111 -> 0, 110 -> 0, 101 -> 0, 100 -> 1
            #         011 -> 1, 010 -> 1, 001 -> 1, 000 -> 0
            next_state[i] = left ^ (center | right)
        self.grid = next_state
    
    def get_neighbors(self, x, y, z):
        neighbors = []
        for i in range(-1, 2):
            for j in range(-1, 2):
                for k in range(-1, 2):
                    if i == 0 and j == 0 and k == 0:
                        continue
                    nx = (x + i) % self.shape[0]
                    ny = (y + j) % self.shape[1]
                    nz = (z + k) % self.shape[2]
                    neighbors.append(self.grid[nx, ny, nz])
        return neighbors

def generate_rule30(initial_state, generations):
    ca = Rule30CA(initial_state.shape, initial_state)
    states = [initial_state]
    for _ in range(generations):
        ca.apply_rule()
        states.append(ca.grid.copy())
    return np.array(states)

# Example usage:
# Assuming you want a 1D CA for simplicity:
width = 101
generations = 50
initial_state = np.zeros(width, dtype=int)
initial_state[width // 2] = 1

ca = generate_rule30(initial_state, generations)
print(ca)  # To visualize the states, consider using matplotlib as shown earlier.
