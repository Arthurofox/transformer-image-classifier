# trainer.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
from pathlib import Path

class Trainer:
    def __init__(self, model, train_dataset, val_dataset, config):
        self.model = model.to(config.device)
        self.config = config
        
        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=2
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=2
        )
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        # Create progress bar
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.config.epochs} [Train]')
        
        for inputs, targets in pbar:
            inputs, targets = inputs.to(self.config.device), targets.to(self.config.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{total_loss/len(pbar):.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
            
        return total_loss / len(self.train_loader), 100. * correct / total

    def validate(self, epoch):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        # Create progress bar for validation
        pbar = tqdm(self.val_loader, desc=f'Epoch {epoch+1}/{self.config.epochs} [Valid]')
        
        with torch.no_grad():
            for inputs, targets in pbar:
                inputs, targets = inputs.to(self.config.device), targets.to(self.config.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{total_loss/len(pbar):.4f}',
                    'acc': f'{100.*correct/total:.2f}%'
                })
                
        return total_loss / len(self.val_loader), 100. * correct / total