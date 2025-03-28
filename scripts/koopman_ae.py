import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
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
    def __init__(self, input_dim, hidden_dim, output_dim, depth=2, device='cuda', num_subjects=1):
        super(KoopmanAE, self).__init__()
        self.device = device
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.depth = depth

        self.encoder = MLP(input_dim, hidden_dim, output_dim, depth, activation="tanh", final_activation=None, device=device)
        self.decoder = MLP(output_dim, hidden_dim, input_dim, depth, activation="tanh", final_activation=None, device=device)

        # self.A = nn.ModuleList([nn.Linear(output_dim, output_dim) for _ in range(num_subjects)]).to(self.device)
        self.A = nn.Parameter(torch.randn(num_subjects, output_dim, output_dim).to(self.device))

    def forward(self, x, subject_indices=None):
        # select the As we'll use
        if subject_indices is not None:
            A = self.A[subject_indices]
        else:
            A = self.A[0]

        x = self.encoder(x)
        y = torch.bmm(A, x.unsqueeze(-1)).squeeze(-1)
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

def train_koopman_ae(model, X, Y, subject_indices, n_epochs=1000, batch_size=32, learning_rate=1e-3, verbose=True):
    """Train the KoopmanAE model"""
    writer = SummaryWriter()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    

    train_dataset = torch.utils.data.TensorDataset(X, Y, subject_indices)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True) 
    
    losses = []
    
    for epoch in range(n_epochs):
        epoch_loss = 0
        n_batches = len(train_loader)
        
        for (x_batch, y_batch, subject_indices_batch) in train_loader:

            y_pred = model(x_batch, subject_indices_batch)
            
            # Compute loss
            loss = criterion(y_pred, y_batch)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        epoch_loss /= n_batches
        losses.append(epoch_loss)
        
        writer.add_scalar("Loss/train", epoch_loss, epoch)
        
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

