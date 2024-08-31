# Define the 3D Cellular Automata
class rule30:
    def __init__(self, shape, ruleset):
        self.shape = shape
        self.grid =  np.zeros_like(current_state)
    for i in range(1, len(current_state) - 1):
        left = current_state[i - 1]
        center = current_state[i]
        right = current_state[i + 1]
        # Rule 30: 111 -> 0, 110 -> 0, 101 -> 0, 100 -> 1
        #         011 -> 1, 010 -> 1, 001 -> 1, 000 -> 0
        next_state[i] = left ^ (center | right)
                    new_grid[x, y, z] = self.apply_rules(x, y, z)
        self.grid = new_grid

def generate_rule30(initial_state, generations):
    """Generate the cellular automaton for a given number of generations."""
    rows = [initial_state]
    current_state = initial_state
    for _ in range(generations):
        current_state = rule30(current_state)
        rows.append(current_state)
    return np.array(rows)


    def get_neighbors(self, x, y, z):
        neighbors = []
        for i in range(-1, 2):
            for j in range(-1, 2):
                for k in range(-1, 2):
                    if i == 0 and j == 0 and k == 0:
                        continue
                    nx, ny, nz = (x + i) % self.shape[0], (y + j) % self.shape[1], (z + k) % self.shape[2]
                    neighbors.append(self.grid[nx, ny, nz])
        return neighbors