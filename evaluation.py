# evaluation.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from model import VisionTransformer
from config import Config
from dataset import get_transform

class Evaluator:
    def __init__(self, model, test_dataset, config):
        self.model = model.to(config.device)
        self.config = config
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=2
        )
        self.criterion = nn.CrossEntropyLoss()

    def evaluate(self):
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for inputs, targets in tqdm(self.test_loader, desc="Evaluating"):
                inputs = inputs.to(self.config.device)
                targets = targets.to(self.config.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                total_loss += loss.item()
                
                _, predicted = outputs.max(1)
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        # Calculate metrics
        test_loss = total_loss / len(self.test_loader)
        accuracy = (np.array(all_predictions) == np.array(all_targets)).mean() * 100
        
        # Generate confusion matrix
        cm = confusion_matrix(all_targets, all_predictions)
        
        return {
            'test_loss': test_loss,
            'accuracy': accuracy,
            'predictions': all_predictions,
            'targets': all_targets,
            'confusion_matrix': cm
        }

    def plot_confusion_matrix(self, cm, class_names):
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png')
        plt.close()

def main():
    config = Config()
    
    # Load test dataset
    test_dataset = datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=get_transform(is_train=False)
    )
    
    # Load model
    model = VisionTransformer(
        img_size=config.img_size,
        patch_size=config.patch_size,
        in_channels=config.in_channels,
        num_classes=config.num_classes,
        embed_dim=config.embed_dim,
        depth=config.depth,
        num_heads=config.num_heads,
        mlp_ratio=config.mlp_ratio,
        dropout=0.0  # No dropout during evaluation
    )
    
    # Load trained weights
    checkpoint = torch.load('model_best.pth', map_location=config.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Initialize evaluator and run evaluation
    evaluator = Evaluator(model, test_dataset, config)
    results = evaluator.evaluate()
    
    # Print results
    print(f"Test Loss: {results['test_loss']:.4f}")
    print(f"Test Accuracy: {results['accuracy']:.2f}%")
    
    # Plot confusion matrix
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                  'dog', 'frog', 'horse', 'ship', 'truck']  # CIFAR-10 classes
    evaluator.plot_confusion_matrix(results['confusion_matrix'], class_names)
    
    # Generate detailed classification report
    print("\nClassification Report:")
    print(classification_report(results['targets'], results['predictions'],
                              target_names=class_names))

if __name__ == '__main__':
    main()