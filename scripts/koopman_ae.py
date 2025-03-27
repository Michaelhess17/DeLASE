import torch
from torch import nn
import numpy as np

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, depth=2, activation="tanh", final_activation=None, device='cuda'):
        super(MLP, self).__init__()
        self.device = device
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        if activation not in ["tanh", "relu", None]:
            raise ValueError("Activation must be either 'tanh' or 'relu'")
        else:
            if activation == "tanh":
                self.activation = nn.Tanh
            else:
                self.activation = nn.ReLU
        
        if final_activation not in ["tanh", "relu", None]:
            raise ValueError("Final activation must be either 'tanh' or None")
        else:
            if final_activation == "tanh":
                self.final_activation = nn.Tanh
            elif final_activation == "relu":
                self.final_activation = nn.ReLU
            else:
                self.final_activation = None
                
        self.layers = []
        for i in range(depth):
            if i == 0:
                self.layers.append(nn.Linear(input_dim, hidden_dim))
                if self.activation is not None:
                    self.layers.append(self.activation())
            elif i == depth - 1:
                self.layers.append(nn.Linear(hidden_dim, output_dim))
                if self.final_activation is not None:
                    self.layers.append(self.final_activation())
            else:
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))
                if self.activation is not None:
                    self.layers.append(self.activation())
        
        self.model = nn.Sequential(*self.layers).to(self.device)
        
    def forward(self, x):
        return self.model(x)

class KoopmanAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, depth=2, device='cuda'):
        super(KoopmanAE, self).__init__()
        self.device = device
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.depth = depth

        self.encoder = MLP(input_dim, hidden_dim, output_dim, depth, activation="tanh", final_activation=None, device=device)
        self.decoder = MLP(output_dim, hidden_dim, input_dim, depth, activation="tanh", final_activation=None, device=device)

        self.A = nn.Linear(output_dim, output_dim).to(self.device)

    def forward(self, x):
        x = self.encoder(x)
        y = self.A(x)
        x_out = self.decoder(y)
        return x_out

def generate_oscillator_data(n_samples=1000, n_features=50, n_oscillators=3):
    """Generate high-dimensional oscillator data"""
    t = np.linspace(0, 10, n_samples)
    base_signals = np.zeros((n_samples, n_oscillators))
    
    # Generate base oscillators with different frequencies
    for i in range(n_oscillators):
        freq = 1 + i * 0.5  # Different frequencies
        base_signals[:, i] = np.sin(2 * np.pi * freq * t)
    
    # Create random mixing matrix
    mixing_matrix = np.random.randn(n_oscillators, n_features)
    
    # Generate high-dimensional data
    data = base_signals @ mixing_matrix
    # Normalize
    data = (data - data.mean(0)) / data.std(0)
    return data

def train_koopman_ae(model, data, n_epochs=1000, batch_size=32, learning_rate=1e-3, verbose=True):
    """Train the KoopmanAE model"""
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    # Prepare data
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data).float().to(model.device)
    X = data[:-1]  # All but last sample
    Y = data[1:]   # All but first sample

    train_dataset = torch.utils.data.TensorDataset(X, Y)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    losses = []
    
    for epoch in range(n_epochs):
        epoch_loss = 0
        n_batches = len(train_loader)
        
        for (x_batch, y_batch) in train_loader:

            y_pred = model(x_batch)
            
            # Compute loss
            loss = criterion(y_pred, y_batch)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        epoch_loss /= n_batches
        losses.append(epoch_loss)
        
        if ((epoch + 1) % 100 == 0) and verbose:
            print(f'Epoch [{epoch+1}/{n_epochs}], Loss: {epoch_loss:.6f}')
    
    return losses

def predict(model, data, n_steps=100):
    """Predict future states using the KoopmanAE model"""
    model.eval()
    with torch.no_grad():
        x = data[:1]
        predictions = [x]
        for _ in range(n_steps):
            x = model(predictions[-1])
            predictions.append(x)
        predictions = torch.stack(predictions[1:]).squeeze()
    return predictions

def evaluate_mse(model, test_data, n_steps):
    """Evaluate MSE on test data"""
    model.eval()
    with torch.no_grad():
        predictions = predict(model, test_data, n_steps=test_data.shape[0])
        mse = torch.mean((predictions[:test_data.shape[0]-1] - test_data[1:n_steps+1])**2).item()
    return mse

