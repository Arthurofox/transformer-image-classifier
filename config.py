# config.py
import torch

class Config:
    # Model parameters 
    img_size = 64           
    patch_size = 8          
    in_channels = 3
    num_classes = 10        
    embed_dim = 192         
    depth = 4             
    num_heads = 3         
    mlp_ratio = 2.0       
    dropout = 0.1
    
    # Training parameters
    batch_size = 32        
    learning_rate = 3e-4
    weight_decay = 0.03
    epochs = 50            
    
    # Device configuration
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    # Dataset parameters
    num_workers = 2         