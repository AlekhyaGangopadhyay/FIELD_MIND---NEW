import torch
import torch.nn as nn
import torch.nn.functional as F

class SciSenseProjection(nn.Module):
    """
    Standard projection layer that maps intermediate hidden representations
    to the unified 4096-dimensional SciSense embedding space.
    """
    def __init__(self, hidden_dim, embedding_dim=4096):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, embedding_dim)
        self.layer_norm = nn.LayerNorm(embedding_dim)
        
    def forward(self, x):
        x = self.proj(x)
        x = self.layer_norm(x)
        # L2 normalize embeddings so they reside on a unit hypersphere (for cosine similarity)
        return F.normalize(x, p=2, dim=-1)

class GasEncoder(nn.Module):
    """
    Encoder for gas concentration readings.
    Default input dimension is 6: Methane, CO, LPG, Smoke, NOx, and CO2.
    """
    def __init__(self, input_dim=6, hidden_dim=128, embedding_dim=4096):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.projection = SciSenseProjection(hidden_dim, embedding_dim)
        
    def forward(self, x):
        # Ensure input tensor is float
        x = x.float()
        hidden = self.net(x)
        return self.projection(hidden)

class EnvironmentalEncoder(nn.Module):
    """
    Encoder for microclimate/environmental sensor readings.
    Default input dimension is 4: Temperature, Humidity, Pressure, and Occupancy.
    """
    def __init__(self, input_dim=4, hidden_dim=128, embedding_dim=4096):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.projection = SciSenseProjection(hidden_dim, embedding_dim)
        
    def forward(self, x):
        x = x.float()
        hidden = self.net(x)
        return self.projection(hidden)

class VibrationEncoder(nn.Module):
    """
    Encoder for seismic/blast vibration characteristics.
    Default input dimension is 15: Spatial coordinates, offset, charges, and SD parameters.
    """
    def __init__(self, input_dim=15, hidden_dim=256, embedding_dim=4096):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.projection = SciSenseProjection(hidden_dim, embedding_dim)
        
    def forward(self, x):
        x = x.float()
        hidden = self.net(x)
        return self.projection(hidden)

class UltrasonicEncoder(nn.Module):
    """
    Encoder for ultrasonic robot distance readings.
    Input dimension is configurable (supports 2, 4, or 24 sensors).
    """
    def __init__(self, input_dim=24, hidden_dim=256, embedding_dim=4096):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.projection = SciSenseProjection(hidden_dim, embedding_dim)
        
    def forward(self, x):
        x = x.float()
        hidden = self.net(x)
        return self.projection(hidden)
