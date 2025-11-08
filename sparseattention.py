import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

import torch
import torch.nn as nn

class SparseAttention(nn.Module):
    def __init__(self, input_dim, sparse_dim, num_heads):
        super(SparseAttention, self).__init__()
        self.input_dim = input_dim
        self.sparse_dim = sparse_dim
        self.num_heads = num_heads
        
        self.query = nn.Linear(input_dim, input_dim * num_heads)
        self.key = nn.Linear(input_dim, input_dim * num_heads)
        self.value = nn.Linear(input_dim, input_dim * num_heads)
        self.output = nn.Linear(input_dim * num_heads, input_dim)
        
        self.sparse_pattern = self._create_sparse_pattern()
    
    def _create_sparse_pattern(self):
        # Create a fixed sparse pattern
        # For simplicity, we randomly select sparse_dim elements out of input_dim
        indices = torch.randperm(self.input_dim)[:self.sparse_dim]
        sparse_pattern = torch.zeros(self.input_dim)
        sparse_pattern[indices] = 1
        return sparse_pattern
    
    def forward(self, x):
        batch_size = x.size(0)
        num_elements = 1
        
        # Reshape input for multi-head attention
        x = x.view(batch_size, num_elements, 1, self.input_dim)
        
        # Compute query, key, and value
        query = self.query(x)  # (batch_size, num_elements, num_heads, input_dim)
        key = self.key(x)  # (batch_size, num_elements, num_heads, input_dim)
        value = self.value(x)  # (batch_size, num_elements, num_heads, input_dim)
        
        # Apply sparse pattern to key and value
        key_sparse = key * self.sparse_pattern
        value_sparse = value * self.sparse_pattern
        
        # Compute attention scores
        scores = torch.matmul(query, key_sparse.transpose(2, 3))  # (batch_size, num_elements, num_heads, num_elements)
        
        # Apply softmax to get attention weights
        attn_weights = torch.softmax(scores, dim=-1)
        
        # Attend to sparse elements by applying attention weights
        attended_sparse = torch.matmul(attn_weights, value_sparse)  # (batch_size, num_elements, num_heads, input_dim)
        
        # Aggregate attended sparse elements across heads
        attended = torch.sum(attended_sparse, dim=2)  # (batch_size, num_elements, input_dim)
        
        # Reshape for output
        attended = attended.view(batch_size, num_elements, self.input_dim * self.num_heads)
        
        # Apply linear transformation for final output
        output = self.output(attended)  # (batch_size, num_elements, input_dim)
        
        return output


# Define a simple classifier using SparseAttention
class Classifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Classifier, self).__init__()
        self.attention = SparseAttention(input_dim, sparse_dim=100, num_heads=4)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = self.attention(x)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Prepare the data
train_dataset = MNIST(root='data/dataset/mnist', train=True, transform=ToTensor(), download=True)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Create the model and optimizer
model = Classifier(input_dim=784, hidden_dim=256, output_dim=10)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(10):
    for images, labels in train_loader:
        optimizer.zero_grad()
        output = model(images.view(-1, 784))
        loss = nn.CrossEntropyLoss()(output, labels)
        loss.backward()
        optimizer.step()
    
    print(f"Epoch {epoch+1}: Loss = {loss.item()}")
