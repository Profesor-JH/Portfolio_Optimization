import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Define the portfolio optimization model
class Portfolio(nn.Module):
    def __init__(self, input_size, output_size):
        super(Portfolio, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_size),
            nn.Softmax(dim=-1)  # Use Softmax to ensure allocations sum up to 1
        )

    def forward(self, x):
        return self.network(x)

# Define the loss function
def loss_function(predictions, targets, lambda_reg=0.5, diversification_reg=0.1):
    """
    Loss function: Mean return - lambda_reg * portfolio variance + diversification_reg * diversification penalty
    """
    
    portfolio_return = torch.mean(torch.sum(predictions * targets, dim=1))
    portfolio_variance = torch.var(torch.sum(predictions * targets, dim=1))
    
    # New term to encourage diversification
    diversification_penalty = diversification_reg * torch.var(predictions)
    total_loss = -portfolio_return + lambda_reg * portfolio_variance + diversification_penalty
    
    return total_loss

# Train the model
def train_model(model, data_loader, optimizer, epochs=100):
    model.train()
    for epoch in range(epochs):
        for inputs, targets in data_loader:
            optimizer.zero_grad()
            predictions = model(inputs)
            loss = loss_function(predictions, targets)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch + 1}, Loss: {loss.item()}')
