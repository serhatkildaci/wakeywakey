"""
Training module for wake word detection models.

Provides comprehensive training pipeline with experiment tracking,
hyperparameter optimization, and advanced training strategies.
"""

import os
import time
import json
import pickle
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable, Tuple
import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import matplotlib.pyplot as plt

# Optional dependencies
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

logger = logging.getLogger(__name__)


class Trainer:
    """
    Comprehensive trainer for wake word detection models.
    
    Features experiment tracking, hyperparameter optimization, model selection,
    and advanced training strategies like early stopping and learning rate scheduling.
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        use_wandb: bool = False,
        project_name: str = "wakeywakey"
    ):
        """
        Initialize trainer.
        
        Args:
            config: Training configuration dictionary
            use_wandb: Whether to use Weights & Biases logging
            project_name: W&B project name
        """
        self.config = config
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        self.project_name = project_name
        
        # Training state
        self.model: Optional[nn.Module] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
        self.criterion: Optional[nn.Module] = None
        
        # Data loaders
        self.train_loader: Optional[DataLoader] = None
        self.val_loader: Optional[DataLoader] = None
        self.test_loader: Optional[DataLoader] = None
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'lr': []
        }
        
        # Best model tracking
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.best_model_state = None
        self.patience_counter = 0
        
        # Device setup
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize W&B if enabled
        if self.use_wandb:
            self._init_wandb()
        
        logger.info(f"Trainer initialized on {self.device}")
    
    def _init_wandb(self):
        """Initialize Weights & Biases logging."""
        try:
            wandb.init(
                project=self.project_name,
                config=self.config,
                name=f"wakeywakey_{self.config.get('model_type', 'unknown')}_{int(time.time())}"
            )
            logger.info("W&B logging initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize W&B: {e}")
            self.use_wandb = False
    
    def setup_model(self, model: nn.Module):
        """
        Setup model for training.
        
        Args:
            model: PyTorch model to train
        """
        self.model = model.to(self.device)
        
        # Log model info
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        logger.info(f"Model setup: {total_params:,} total params, {trainable_params:,} trainable")
        
        if self.use_wandb:
            wandb.watch(self.model)
    
    def setup_optimizer(self):
        """Setup optimizer based on configuration."""
        if self.model is None:
            raise ValueError("Model must be setup before optimizer")
        
        optimizer_type = self.config.get('optimizer', 'adam')
        lr = self.config.get('learning_rate', 0.001)
        weight_decay = self.config.get('weight_decay', 1e-5)
        
        if optimizer_type.lower() == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                betas=self.config.get('betas', (0.9, 0.999))
            )
        elif optimizer_type.lower() == 'adamw':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        elif optimizer_type.lower() == 'sgd':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=lr,
                momentum=self.config.get('momentum', 0.9),
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")
        
        logger.info(f"Optimizer setup: {optimizer_type} with lr={lr}")
    
    def setup_scheduler(self):
        """Setup learning rate scheduler."""
        if self.optimizer is None:
            raise ValueError("Optimizer must be setup before scheduler")
        
        scheduler_type = self.config.get('scheduler', 'cosine')
        
        if scheduler_type == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.get('epochs', 100),
                eta_min=self.config.get('min_lr', 1e-6)
            )
        elif scheduler_type == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.get('step_size', 30),
                gamma=self.config.get('gamma', 0.1)
            )
        elif scheduler_type == 'plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=self.config.get('factor', 0.5),
                patience=self.config.get('scheduler_patience', 5),
                min_lr=self.config.get('min_lr', 1e-6)
            )
        elif scheduler_type is None:
            self.scheduler = None
        else:
            raise ValueError(f"Unknown scheduler: {scheduler_type}")
        
        logger.info(f"Scheduler setup: {scheduler_type}")
    
    def setup_criterion(self):
        """Setup loss criterion."""
        criterion_type = self.config.get('criterion', 'bce')
        
        if criterion_type == 'bce':
            self.criterion = nn.BCEWithLogitsLoss()
        elif criterion_type == 'focal':
            self.criterion = FocalLoss(
                alpha=self.config.get('focal_alpha', 1.0),
                gamma=self.config.get('focal_gamma', 2.0)
            )
        elif criterion_type == 'weighted_bce':
            pos_weight = self.config.get('pos_weight', 1.0)
            self.criterion = nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor(pos_weight, device=self.device)
            )
        else:
            raise ValueError(f"Unknown criterion: {criterion_type}")
        
        logger.info(f"Criterion setup: {criterion_type}")
    
    def setup_data_loaders(self, train_dataset, val_dataset=None, test_dataset=None):
        """
        Setup data loaders.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset (optional)
            test_dataset: Test dataset (optional)
        """
        batch_size = self.config.get('batch_size', 32)
        num_workers = self.config.get('num_workers', 4)
        
        # Training loader
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        # Validation loader
        if val_dataset is not None:
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True if self.device.type == 'cuda' else False
            )
        
        # Test loader
        if test_dataset is not None:
            self.test_loader = DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True if self.device.type == 'cuda' else False
            )
        
        logger.info(f"Data loaders setup: train={len(self.train_loader)}, "
                   f"val={len(self.val_loader) if self.val_loader else 0}, "
                   f"test={len(self.test_loader) if self.test_loader else 0}")
    
    def train_epoch(self) -> Tuple[float, float]:
        """
        Train for one epoch.
        
        Returns:
            Tuple of (average_loss, accuracy)
        """
        if self.model is None or self.train_loader is None:
            raise ValueError("Model and data loader must be setup")
        
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        for batch_idx, (features, labels) in enumerate(self.train_loader):
            features = features.to(self.device)
            labels = labels.to(self.device).float()
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(features).squeeze()
            
            # Handle single sample case
            if outputs.dim() == 0:
                outputs = outputs.unsqueeze(0)
                labels = labels.unsqueeze(0)
            
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.config.get('grad_clip', 0) > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['grad_clip']
                )
            
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            predictions = torch.sigmoid(outputs) > 0.5
            total_correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = total_correct / total_samples
        
        return avg_loss, accuracy
    
    def validate_epoch(self) -> Tuple[float, float, Dict[str, float]]:
        """
        Validate for one epoch.
        
        Returns:
            Tuple of (average_loss, accuracy, metrics_dict)
        """
        if self.model is None or self.val_loader is None:
            return 0.0, 0.0, {}
        
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            for features, labels in self.val_loader:
                features = features.to(self.device)
                labels = labels.to(self.device).float()
                
                outputs = self.model(features).squeeze()
                
                # Handle single sample case
                if outputs.dim() == 0:
                    outputs = outputs.unsqueeze(0)
                    labels = labels.unsqueeze(0)
                
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                
                # Collect predictions and labels
                probabilities = torch.sigmoid(outputs)
                predictions = probabilities > 0.5
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = accuracy_score(all_labels, all_predictions)
        
        # Additional metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average='binary', zero_division=0
        )
        
        try:
            auc = roc_auc_score(all_labels, all_probabilities)
        except ValueError:
            auc = 0.0
        
        metrics = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc
        }
        
        return avg_loss, accuracy, metrics
    
    def train(
        self,
        model: nn.Module,
        train_dataset,
        val_dataset=None,
        progress_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Complete training pipeline.
        
        Args:
            model: Model to train
            train_dataset: Training dataset
            val_dataset: Validation dataset
            progress_callback: Callback for progress updates
            
        Returns:
            Training results dictionary
        """
        # Setup
        self.setup_model(model)
        self.setup_optimizer()
        self.setup_scheduler()
        self.setup_criterion()
        self.setup_data_loaders(train_dataset, val_dataset)
        
        epochs = self.config.get('epochs', 100)
        early_stopping_patience = self.config.get('early_stopping_patience', 10)
        save_best_model = self.config.get('save_best_model', True)
        
        logger.info(f"Starting training for {epochs} epochs")
        start_time = time.time()
        
        for epoch in range(epochs):
            epoch_start = time.time()
            
            # Training
            train_loss, train_acc = self.train_epoch()
            
            # Validation
            val_loss, val_acc, val_metrics = self.validate_epoch()
            
            # Learning rate scheduling
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Update history
            current_lr = self.optimizer.param_groups[0]['lr']
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            self.history['lr'].append(current_lr)
            
            # Best model tracking
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_val_acc = val_acc
                self.patience_counter = 0
                
                if save_best_model:
                    self.best_model_state = self.model.state_dict().copy()
            else:
                self.patience_counter += 1
            
            # Logging
            epoch_time = time.time() - epoch_start
            logger.info(
                f"Epoch {epoch+1}/{epochs}: "
                f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
                f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}, "
                f"lr={current_lr:.6f}, time={epoch_time:.2f}s"
            )
            
            # W&B logging
            if self.use_wandb:
                log_data = {
                    'epoch': epoch + 1,
                    'train_loss': train_loss,
                    'train_acc': train_acc,
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                    'learning_rate': current_lr,
                    'epoch_time': epoch_time
                }
                log_data.update({f"val_{k}": v for k, v in val_metrics.items()})
                wandb.log(log_data)
            
            # Progress callback
            if progress_callback:
                progress_callback(epoch + 1, train_loss, train_acc, current_lr)
            
            # Early stopping
            if early_stopping_patience > 0 and self.patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        total_time = time.time() - start_time
        
        # Load best model
        if save_best_model and self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            logger.info(f"Best model loaded: val_loss={self.best_val_loss:.4f}, val_acc={self.best_val_acc:.4f}")
        
        # Training results
        results = {
            'total_time': total_time,
            'total_epochs': epoch + 1,
            'best_val_loss': self.best_val_loss,
            'best_val_acc': self.best_val_acc,
            'final_train_loss': train_loss,
            'final_train_acc': train_acc,
            'history': self.history.copy()
        }
        
        logger.info(f"Training completed in {total_time:.2f}s")
        return results
    
    def save_model(self, output_path: str, include_config: bool = True):
        """
        Save trained model.
        
        Args:
            output_path: Output file path
            include_config: Whether to include training config
        """
        if self.model is None:
            raise ValueError("No model to save")
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'model_type': self.config.get('model_type', 'Unknown'),
            'config': self.model.config if hasattr(self.model, 'config') else {},
            'accuracy': self.best_val_acc,
            'loss': self.best_val_loss,
            'epoch': len(self.history['train_loss']),
            'timestamp': time.time(),
            'version': '0.1.0'
        }
        
        if include_config:
            checkpoint['training_config'] = self.config
            checkpoint['history'] = self.history
        
        torch.save(checkpoint, output_path)
        logger.info(f"Model saved to {output_path}")
    
    def evaluate(self, test_dataset) -> Dict[str, Any]:
        """
        Evaluate model on test dataset.
        
        Args:
            test_dataset: Test dataset
            
        Returns:
            Evaluation results dictionary
        """
        if self.model is None:
            raise ValueError("No model to evaluate")
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.get('batch_size', 32),
            shuffle=False,
            num_workers=self.config.get('num_workers', 4)
        )
        
        self.model.eval()
        all_predictions = []
        all_labels = []
        all_probabilities = []
        total_loss = 0.0
        inference_times = []
        
        with torch.no_grad():
            for features, labels in test_loader:
                features = features.to(self.device)
                labels = labels.to(self.device).float()
                
                # Measure inference time
                start_time = time.time()
                outputs = self.model(features).squeeze()
                inference_time = time.time() - start_time
                inference_times.append(inference_time)
                
                # Handle single sample case
                if outputs.dim() == 0:
                    outputs = outputs.unsqueeze(0)
                    labels = labels.unsqueeze(0)
                
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                
                # Collect predictions
                probabilities = torch.sigmoid(outputs)
                predictions = probabilities > 0.5
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average='binary', zero_division=0
        )
        
        try:
            auc = roc_auc_score(all_labels, all_probabilities)
        except ValueError:
            auc = 0.0
        
        avg_loss = total_loss / len(test_loader)
        avg_inference_time = np.mean(inference_times) / len(features)  # Per sample
        
        results = {
            'test_loss': avg_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc': auc,
            'avg_inference_time': avg_inference_time,
            'total_samples': len(all_labels)
        }
        
        logger.info(f"Evaluation results: acc={accuracy:.4f}, f1={f1:.4f}, auc={auc:.4f}")
        return results


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance."""
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none'
        )
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()


def hyperparameter_optimization(
    model_class,
    train_dataset,
    val_dataset,
    n_trials: int = 50,
    timeout: int = 3600
) -> Dict[str, Any]:
    """
    Hyperparameter optimization using Optuna.
    
    Args:
        model_class: Model class to optimize
        train_dataset: Training dataset
        val_dataset: Validation dataset
        n_trials: Number of optimization trials
        timeout: Timeout in seconds
        
    Returns:
        Best hyperparameters and results
    """
    if not OPTUNA_AVAILABLE:
        raise ImportError("Optuna not available for hyperparameter optimization")
    
    def objective(trial):
        # Suggest hyperparameters
        config = {
            'model_type': model_class.__name__,
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
            'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64]),
            'hidden_dim': trial.suggest_categorical('hidden_dim', [32, 64, 128]),
            'dropout': trial.suggest_float('dropout', 0.1, 0.5),
            'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True),
            'optimizer': trial.suggest_categorical('optimizer', ['adam', 'adamw']),
            'scheduler': trial.suggest_categorical('scheduler', ['cosine', 'step', None]),
            'epochs': 20  # Reduced for optimization
        }
        
        # Create model
        model = model_class(
            hidden_dim=config['hidden_dim'],
            dropout=config['dropout']
        )
        
        # Train model
        trainer = Trainer(config)
        results = trainer.train(model, train_dataset, val_dataset)
        
        return results['best_val_acc']
    
    # Run optimization
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, timeout=timeout)
    
    return {
        'best_params': study.best_params,
        'best_value': study.best_value,
        'n_trials': len(study.trials)
    } 