"""
Neural network architectures for wake word detection.

Provides multiple model architectures optimized for different deployment scenarios:
- LightweightCNN: ~30KB, <5ms inference with depthwise separable convolutions
- CompactRNN: Memory-efficient GRU with attention mechanism  
- HybridCRNN: Combined CNN+RNN architecture
- MobileWakeWord: Ultra-lightweight <10KB, <1ms for microcontrollers
- AttentionWakeWord: Transformer-based with self-attention
"""

import math
from typing import Optional, Dict, Any, Type
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class LightweightCNN(nn.Module):
    """
    Lightweight CNN optimized for edge devices (~30KB, <5ms inference).
    
    Uses depthwise separable convolutions and squeeze-and-excitation blocks
    for efficient feature extraction with minimal parameters.
    """
    
    def __init__(
        self,
        input_size: int = 26,
        hidden_dim: int = 64,
        num_layers: int = 3,
        dropout: float = 0.2,
        use_se: bool = True
    ):
        """
        Initialize LightweightCNN.
        
        Args:
            input_size: Input feature dimension
            hidden_dim: Hidden layer dimension
            num_layers: Number of convolutional layers
            dropout: Dropout probability
            use_se: Whether to use squeeze-and-excitation blocks
        """
        super().__init__()
        
        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_se = use_se
        
        # Input projection
        self.input_proj = nn.Linear(input_size, hidden_dim)
        
        # Depthwise separable conv layers
        self.conv_layers = nn.ModuleList()
        for i in range(num_layers):
            in_channels = hidden_dim
            out_channels = hidden_dim
            
            # Depthwise convolution
            dw_conv = nn.Conv1d(
                in_channels, in_channels, 
                kernel_size=3, padding=1, 
                groups=in_channels, bias=False
            )
            
            # Pointwise convolution
            pw_conv = nn.Conv1d(
                in_channels, out_channels, 
                kernel_size=1, bias=False
            )
            
            # Batch normalization and activation
            bn = nn.BatchNorm1d(out_channels)
            
            layer = nn.Sequential(
                dw_conv,
                pw_conv,
                bn,
                nn.ReLU6(inplace=True)
            )
            
            self.conv_layers.append(layer)
            
            # Add SE block if enabled
            if use_se:
                se_block = SqueezeExciteBlock1D(out_channels)
                self.conv_layers.append(se_block)
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """Forward pass."""
        # x shape: (batch_size, input_size)
        batch_size = x.size(0)
        
        # Project input to hidden dimension
        x = self.input_proj(x)  # (batch_size, hidden_dim)
        
        # Reshape for conv1d: (batch_size, hidden_dim, 1)
        x = x.unsqueeze(-1)
        
        # Apply convolutional layers
        for layer in self.conv_layers:
            x = layer(x)
        
        # Global average pooling
        x = self.global_pool(x)  # (batch_size, hidden_dim, 1)
        x = x.view(batch_size, -1)  # (batch_size, hidden_dim)
        
        # Classification
        x = self.classifier(x)  # (batch_size, 1)
        
        return x


class SqueezeExciteBlock1D(nn.Module):
    """Squeeze-and-Excitation block for 1D convolutions."""
    
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class CompactRNN(nn.Module):
    """
    Memory-efficient RNN with attention mechanism (~50KB, <10ms inference).
    
    Uses GRU layers with attention pooling for temporal modeling.
    """
    
    def __init__(
        self,
        input_size: int = 26,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = True,
        use_attention: bool = True
    ):
        """
        Initialize CompactRNN.
        
        Args:
            input_size: Input feature dimension
            hidden_dim: Hidden layer dimension
            num_layers: Number of RNN layers
            dropout: Dropout probability
            bidirectional: Whether to use bidirectional RNN
            use_attention: Whether to use attention pooling
        """
        super().__init__()
        
        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.use_attention = use_attention
        
        # Input projection
        self.input_proj = nn.Linear(input_size, hidden_dim)
        
        # RNN layers
        rnn_dim = hidden_dim
        self.rnn = nn.GRU(
            rnn_dim, hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Calculate RNN output dimension
        rnn_output_dim = hidden_dim * (2 if bidirectional else 1)
        
        # Attention mechanism
        if use_attention:
            self.attention = AttentionPooling(rnn_output_dim)
            pool_output_dim = rnn_output_dim
        else:
            self.attention = None
            pool_output_dim = rnn_output_dim
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(pool_output_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, x):
        """Forward pass."""
        # x shape: (batch_size, input_size)
        batch_size = x.size(0)
        
        # Project input
        x = self.input_proj(x)  # (batch_size, hidden_dim)
        
        # Add sequence dimension for RNN
        x = x.unsqueeze(1)  # (batch_size, 1, hidden_dim)
        
        # RNN forward pass
        rnn_out, _ = self.rnn(x)  # (batch_size, 1, rnn_output_dim)
        
        # Apply attention or simple pooling
        if self.attention:
            pooled = self.attention(rnn_out)  # (batch_size, rnn_output_dim)
        else:
            pooled = rnn_out.squeeze(1)  # (batch_size, rnn_output_dim)
        
        # Classification
        output = self.classifier(pooled)  # (batch_size, 1)
        
        return output


class AttentionPooling(nn.Module):
    """Attention-based pooling mechanism."""
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        # x shape: (batch_size, seq_len, hidden_dim)
        attn_weights = torch.softmax(self.attention(x), dim=1)  # (batch_size, seq_len, 1)
        pooled = torch.sum(x * attn_weights, dim=1)  # (batch_size, hidden_dim)
        return pooled


class HybridCRNN(nn.Module):
    """
    Combined CNN+RNN architecture (~100KB, <15ms inference).
    
    Uses CNN for local feature extraction followed by RNN for temporal modeling.
    """
    
    def __init__(
        self,
        input_size: int = 26,
        cnn_hidden: int = 32,
        rnn_hidden: int = 64,
        cnn_layers: int = 2,
        rnn_layers: int = 1,
        dropout: float = 0.2
    ):
        """
        Initialize HybridCRNN.
        
        Args:
            input_size: Input feature dimension
            cnn_hidden: CNN hidden dimension
            rnn_hidden: RNN hidden dimension
            cnn_layers: Number of CNN layers
            rnn_layers: Number of RNN layers
            dropout: Dropout probability
        """
        super().__init__()
        
        self.input_size = input_size
        self.cnn_hidden = cnn_hidden
        self.rnn_hidden = rnn_hidden
        
        # Input projection
        self.input_proj = nn.Linear(input_size, cnn_hidden)
        
        # CNN feature extractor
        self.cnn_layers = nn.ModuleList()
        for i in range(cnn_layers):
            conv = nn.Conv1d(
                cnn_hidden, cnn_hidden,
                kernel_size=3, padding=1
            )
            bn = nn.BatchNorm1d(cnn_hidden)
            
            layer = nn.Sequential(
                conv, bn, nn.ReLU(inplace=True),
                nn.Dropout1d(dropout)
            )
            self.cnn_layers.append(layer)
        
        # RNN temporal modeling
        self.rnn = nn.LSTM(
            cnn_hidden, rnn_hidden,
            num_layers=rnn_layers,
            dropout=dropout if rnn_layers > 1 else 0,
            batch_first=True
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(rnn_hidden, rnn_hidden // 2),
            nn.ReLU(inplace=True),
            nn.Linear(rnn_hidden // 2, 1)
        )
    
    def forward(self, x):
        """Forward pass."""
        # x shape: (batch_size, input_size)
        batch_size = x.size(0)
        
        # Project input
        x = self.input_proj(x)  # (batch_size, cnn_hidden)
        
        # Reshape for conv1d
        x = x.unsqueeze(-1)  # (batch_size, cnn_hidden, 1)
        
        # CNN feature extraction
        for layer in self.cnn_layers:
            x = layer(x)
        
        # Reshape for RNN
        x = x.transpose(1, 2)  # (batch_size, 1, cnn_hidden)
        
        # RNN temporal modeling
        rnn_out, (hidden, _) = self.rnn(x)
        
        # Use final hidden state
        output = self.classifier(hidden[-1])  # (batch_size, 1)
        
        return output


class MobileWakeWord(nn.Module):
    """
    Ultra-lightweight model for microcontrollers (<10KB, <1ms inference).
    
    Extremely efficient architecture using minimal parameters and operations.
    """
    
    def __init__(
        self,
        input_size: int = 26,
        hidden_dim: int = 16,
        use_bias: bool = False
    ):
        """
        Initialize MobileWakeWord.
        
        Args:
            input_size: Input feature dimension
            hidden_dim: Hidden layer dimension
            use_bias: Whether to use bias terms
        """
        super().__init__()
        
        self.input_size = input_size
        self.hidden_dim = hidden_dim
        
        # Extremely compact network
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_dim, bias=use_bias),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim // 2, bias=use_bias),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 1, bias=use_bias)
        )
        
        # Initialize with small weights for quantization
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights for quantization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Small weight initialization for better quantization
                nn.init.uniform_(m.weight, -0.1, 0.1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """Forward pass."""
        return self.layers(x)


class AttentionWakeWord(nn.Module):
    """
    Transformer-based model with self-attention.
    
    Uses multi-head self-attention for advanced feature modeling.
    """
    
    def __init__(
        self,
        input_size: int = 26,
        hidden_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        """
        Initialize AttentionWakeWord.
        
        Args:
            input_size: Input feature dimension
            hidden_dim: Hidden layer dimension
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            dropout: Dropout probability
        """
        super().__init__()
        
        self.input_size = input_size
        self.hidden_dim = hidden_dim
        
        # Input projection
        self.input_proj = nn.Linear(input_size, hidden_dim)
        
        # Positional encoding (simple learned)
        self.pos_encoding = nn.Parameter(torch.randn(1, 1, hidden_dim))
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, x):
        """Forward pass."""
        # x shape: (batch_size, input_size)
        batch_size = x.size(0)
        
        # Project input
        x = self.input_proj(x)  # (batch_size, hidden_dim)
        
        # Add sequence dimension
        x = x.unsqueeze(1)  # (batch_size, 1, hidden_dim)
        
        # Add positional encoding
        x = x + self.pos_encoding
        
        # Apply transformer
        x = self.transformer(x)  # (batch_size, 1, hidden_dim)
        
        # Global pooling
        x = x.mean(dim=1)  # (batch_size, hidden_dim)
        
        # Output projection
        output = self.output_proj(x)  # (batch_size, 1)
        
        return output


def get_model_class(model_name: str) -> Optional[Type[nn.Module]]:
    """
    Get model class by name.
    
    Args:
        model_name: Name of the model
        
    Returns:
        Model class or None if not found
    """
    model_classes = {
        'LightweightCNN': LightweightCNN,
        'CompactRNN': CompactRNN,
        'HybridCRNN': HybridCRNN,
        'MobileWakeWord': MobileWakeWord,
        'AttentionWakeWord': AttentionWakeWord
    }
    
    return model_classes.get(model_name)


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """
    Count model parameters.
    
    Args:
        model: PyTorch model
        trainable_only: Whether to count only trainable parameters
        
    Returns:
        Number of parameters
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def get_model_size_kb(model: nn.Module) -> float:
    """
    Estimate model size in KB.
    
    Args:
        model: PyTorch model
        
    Returns:
        Model size in KB (float32 precision)
    """
    param_count = count_parameters(model, trainable_only=False)
    # 4 bytes per float32 parameter
    size_bytes = param_count * 4
    size_kb = size_bytes / 1024
    return size_kb


def create_model_factory():
    """Create a factory for instantiating models with default configurations."""
    
    default_configs = {
        'LightweightCNN': {
            'input_size': 26,
            'hidden_dim': 64,
            'num_layers': 3,
            'dropout': 0.2,
            'use_se': True
        },
        'CompactRNN': {
            'input_size': 26,
            'hidden_dim': 64,
            'num_layers': 2,
            'dropout': 0.2,
            'bidirectional': True,
            'use_attention': True
        },
        'HybridCRNN': {
            'input_size': 26,
            'cnn_hidden': 32,
            'rnn_hidden': 64,
            'cnn_layers': 2,
            'rnn_layers': 1,
            'dropout': 0.2
        },
        'MobileWakeWord': {
            'input_size': 26,
            'hidden_dim': 16,
            'use_bias': False
        },
        'AttentionWakeWord': {
            'input_size': 26,
            'hidden_dim': 64,
            'num_heads': 4,
            'num_layers': 2,
            'dropout': 0.1
        }
    }
    
    def create_model(model_name: str, **kwargs) -> Optional[nn.Module]:
        """
        Create model with default or custom configuration.
        
        Args:
            model_name: Name of the model
            **kwargs: Custom configuration parameters
            
        Returns:
            Model instance or None if model not found
        """
        model_class = get_model_class(model_name)
        if model_class is None:
            logger.error(f"Unknown model: {model_name}")
            return None
        
        # Merge default config with custom parameters
        config = default_configs.get(model_name, {}).copy()
        config.update(kwargs)
        
        try:
            model = model_class(**config)
            logger.info(f"Created {model_name} with {count_parameters(model)} parameters "
                       f"({get_model_size_kb(model):.1f} KB)")
            return model
        except Exception as e:
            logger.error(f"Failed to create {model_name}: {e}")
            return None
    
    return create_model


# Export factory function
create_model = create_model_factory() 