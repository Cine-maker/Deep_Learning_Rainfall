import torch
import torch.nn as nn

class EmbeddingModule(nn.Module):
    def __init__(self, num_features, embedding_size):
        super().__init__()
        self.linear = nn.Linear(num_features, embedding_size)

    def forward(self, x):
        # Reshape input into dimensions (batch_size, num_features, num_time_series, num_time_steps)
        x = x.permute(0, 3, 1, 2)   

        # Apply feature-wise linear embedding
        x = self.linear(x)
        
        return x   

class LTCN(nn.Module):
    def __init__(self, embedding_size, num_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(embedding_size, num_channels, kernel_size=(1,5))
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=(1, 3))

    def forward(self, x):
        x = x.unsqueeze(1)  
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        
        # Keep only last timestep 
        x = x[..., -1:]  
        
        return x

class GLFormer(nn.Module):
    def __init__(self, embedding_size): 
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(embedding_size, nhead=4)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
    def forward(self, x):
        x = x.permute(1, 0, 2, 3)    
        x = self.transformer(x) 
        return x.permute(1, 0, 2, 3)
    
class LightCTS(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = EmbeddingModule(num_features=2, embedding_size=64)   
        self.ltcn = LTCN(embedding_size=64, num_channels=32)
        self.glformer = GLFormer(embedding_size=64)
        
        self.linear = nn.Linear(64, 1) 

    def forward(self, x):
        x = self.embedding(x) 
        x = self.ltcn(x) 
        x = self.glformer(x)
        
        # Flatten across time series dimension
        x = torch.flatten(x, start_dim=1)  
        return self.linear(x)   

model = LightCTS()  

# Rest is data preprocessing and training loop