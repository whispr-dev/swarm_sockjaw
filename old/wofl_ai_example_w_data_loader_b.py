import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# Example data loader (random data for illustration)
data = torch.randn(100, 1, 28, 28).to(device)
labels = torch.ones(100, 1).to(device)
data_loader = DataLoader(TensorDataset(data, labels), batch_size=10)

#  Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Define the CNN
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Define the LSTM
class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[-1])
        return out

# Define the Transformer
class SimpleTransformer(nn.Module):
    def __init__(self, input_dim, nhead, num_encoder_layers, dim_feedforward, output_dim):
        super(SimpleTransformer, self).__init__()
        self.transformer = nn.Transformer(d_model=input_dim, nhead=nhead, num_encoder_layers=num_encoder_layers, dim_feedforward=dim_feedforward)
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, src, tgt):
        transformer_out = self.transformer(src, tgt)
        out = self.fc(transformer_out)
        return out

# Define the 3D Cellular Automata
class GameOfLife3D:
    def __init__(self, shape, ruleset):
        self.shape = shape
        self.grid = np.random.choice([0, 1], size=shape)
        self.ruleset = ruleset

    def step(self):
        new_grid = np.zeros(self.shape)
        for x in range(self.shape[0]):
            for y in range(self.shape[1]):
                for z in range(self.shape[2]):
                    new_grid[x, y, z] = self.apply_rules(x, y, z)
        self.grid = new_grid

    def apply_rules(self, x, y, z):
        neighbors = self.get_neighbors(x, y, z)
        live_neighbors = np.sum(neighbors)
        if self.grid[x, y, z] == 1:
            if live_neighbors in self.ruleset['survive']:
                return 1
            else:
                return 0
        else:
            if live_neighbors in self.ruleset['birth']:
                return 1
            else:
                return 0

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

# Define the Ganglion with CA
class GanglionWithCA(nn.Module):
    def __init__(self, num_models, ca_shape, ca_ruleset):
        super(GanglionWithCA, self).__init__()
        self.clock = torch.zeros(1).to(device)
        self.num_models = num_models
        self.sync_signals = torch.zeros(num_models).to(device)
        self.feedback_signals = torch.zeros(num_models).to(device)
        self.ca = GameOfLife3D(ca_shape, ca_ruleset)

    def forward(self):
        # Update the clock
        self.clock += 1

        # Step the cellular automata
        self.ca.step()

        # Generate synchronization signals based on the CA grid
        for i in range(self.num_models):
            x, y, z = np.unravel_index(i, self.ca.shape)
            self.sync_signals[i] = self.ca.grid[x, y, z]

        return self.sync_signals

    def receive_feedback(self, feedback):
        self.feedback_signals = feedback
        # Adjust clock or sync logic based on feedback if necessary

# Define the synchronized model wrapper
class SyncedModel(nn.Module):
    def __init__(self, base_model, model_id, ganglion):
        super(SyncedModel, self).__init__()
        self.base_model = base_model
        self.model_id = model_id
        self.ganglion = ganglion

    def forward(self, x):
        sync_signal = self.ganglion.sync_signals[self.model_id]
        
        # Use the sync signal to modify the model's behavior if needed
        if sync_signal % 2 == 0:
            x = F.relu(self.base_model(x))
        else:
            x = F.sigmoid(self.base_model(x))
        
        # Generate feedback signal
        feedback = torch.tensor([x.sum().item()]).to(device)  # Example feedback logic
        
        return x, feedback

# Define the crossover layer
class CrossoverLayer(nn.Module):
    def __init__(self, in_features, out_features, num_crosstalk):
        super(CrossoverLayer, self).__init__()
        self.num_crosstalk = num_crosstalk
        self.linear = nn.Linear(in_features, out_features)
        self.crosstalk_weights = nn.Parameter(torch.randn(num_crosstalk, in_features, out_features))

    def forward(self, x):
        base_output = self.linear(x)
        crosstalk_output = torch.zeros_like(base_output)
        for i in range(self.num_crosstalk):
            crosstalk_output += F.linear(x, self.crosstalk_weights[i])
        return base_output + crosstalk_output

# Define the Master Controller with CA
class MasterControllerWithCA(nn.Module):
    def __init__(self, models, ganglion):
        super(MasterControllerWithCA, self).__init__()
        self.models = nn.ModuleList(models)
        self.ganglion = ganglion
        self.crossover_layer = CrossoverLayer(in_features=128, out_features=128, num_crosstalk=3)

    def forward(self, x):
        sync_signals = self.ganglion()
        outputs = []
        feedbacks = []
        
        for model in self.models:
            out, feedback = model(x)
            outputs.append(out)
            feedbacks.append(feedback)

        # Send feedback to the ganglion
        self.ganglion.receive_feedback(torch.stack(feedbacks))
        
        # Apply crossover layer to combine outputs
        combined_output = torch.cat(outputs, dim=1)
        combined_output = self.crossover_layer(combined_output)
        
        return combined_output

# Define the Generator
class Generator(nn.Module):
    def __init__(self, master_controller):
        super(Generator, self).__init__()
        self.master_controller = master_controller
        self.fc = nn.Linear(128, 784)

    def forward(self, x):
        x = self.master_controller(x)
        x = F.relu(self.fc(x))
        return x

# Define the Discriminator
class Discriminator(nn.Module):
    def __init__(self, master_controller):
        super(Discriminator, self).__init__()
        self.master_controller = master_controller
        self.fc = nn.Linear(128, 1)

    def forward(self, x):
        x = self.master_controller(x)
        x = torch.sigmoid(self.fc(x))
        return x

# Training loop
def train_gan_with_ca(generator, discriminator, data_loader, num_epochs, criterion, optimizer_g, optimizer_d, ganglion):
    for epoch in range(num_epochs):
        for real_data, _ in data_loader:
            real_data = real_data.to(device)
            # Train Discriminator
            optimizer_d.zero_grad()
            real_output = discriminator(real_data)
            real_loss = criterion(real_output, torch.ones_like(real_output))

            noise = torch.randn(real_data.size(0), 100).to(device)
            fake_data = generator(noise)
            fake_output = discriminator(fake_data.detach())
            fake_loss = criterion(fake_output, torch.zeros_like(fake_output))

            d_loss = real_loss + fake_loss
            d_loss.backward()
            optimizer_d.step()

            # Train Generator
            optimizer_g.zero_grad()
            fake_output = discriminator(fake_data)
            g_loss = criterion(fake_output, torch.ones_like(fake_output))
            g_loss.backward()
            optimizer_g.step()

            # Update the ganglion
            ganglion()

        print(f'Epoch {epoch}, Generator Loss: {g_loss.item()}, Discriminator Loss: {d_loss.item()}')

