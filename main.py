# main.py
import torch
from model import VisionTransformer
from config import Config
from dataset import get_transform
from trainer import Trainer
from torchvision import datasets
import os
from tqdm import tqdm

def main():
    config = Config()
    
    # Create datasets
    train_dataset = datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=get_transform(is_train=True)
    )
    val_dataset = datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=get_transform(is_train=False)
    )
    
    # Initialize model
    model = VisionTransformer(
        img_size=config.img_size,
        patch_size=config.patch_size,
        in_channels=config.in_channels,
        num_classes=config.num_classes,
        embed_dim=config.embed_dim,
        depth=config.depth,
        num_heads=config.num_heads,
        mlp_ratio=config.mlp_ratio,
        dropout=config.dropout
    )
    
    # Initialize trainer
    trainer = Trainer(model, train_dataset, val_dataset, config)
    
    # Training loop
    best_val_acc = 0
    for epoch in range(config.epochs):
        train_loss, train_acc = trainer.train_epoch(epoch)
        val_loss, val_acc = trainer.validate(epoch)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'val_acc': val_acc,
            }, 'model_best.pth')
        
        print(f'\nSummary Epoch: {epoch+1}/{config.epochs}')
        print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%')
        print(f'Best Val Acc: {best_val_acc:.2f}%\n')

if __name__ == '__main__':
    main()