"""
Model optimization and quantization utilities.

Provides ModelQuantizer for post-training quantization and ModelOptimizer
for pruning, knowledge distillation, and other optimization techniques.
"""

import os
import time
import copy
from typing import Optional, Dict, Any, Tuple, List, Union
import logging

import torch
import torch.nn as nn
import torch.quantization as quant
import numpy as np
from torch.ao.quantization import get_default_qconfig_mapping
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx

logger = logging.getLogger(__name__)


class ModelQuantizer:
    """
    Model quantization utility for compressing models for edge deployment.
    
    Supports post-training quantization (PTQ) and quantization-aware training (QAT)
    with INT8, INT16, and dynamic quantization modes.
    """
    
    def __init__(
        self,
        backend: str = 'fbgemm',
        quantization_type: str = 'dynamic'
    ):
        """
        Initialize model quantizer.
        
        Args:
            backend: Quantization backend ('fbgemm', 'qnnpack')
            quantization_type: Type of quantization ('dynamic', 'static', 'qat')
        """
        self.backend = backend
        self.quantization_type = quantization_type
        
        # Set quantization backend
        if backend == 'fbgemm':
            torch.backends.quantized.engine = 'fbgemm'
        elif backend == 'qnnpack':
            torch.backends.quantized.engine = 'qnnpack'
        else:
            logger.warning(f"Unknown backend {backend}, using default")
        
        logger.info(f"ModelQuantizer initialized: {backend} backend, {quantization_type} mode")
    
    def quantize_dynamic(
        self,
        model: nn.Module,
        dtype: torch.dtype = torch.qint8
    ) -> nn.Module:
        """
        Apply dynamic quantization to model.
        
        Args:
            model: Model to quantize
            dtype: Quantization data type
            
        Returns:
            Quantized model
        """
        try:
            model_copy = copy.deepcopy(model)
            model_copy.eval()
            
            # Apply dynamic quantization
            quantized_model = torch.quantization.quantize_dynamic(
                model_copy,
                {nn.Linear, nn.LSTM, nn.GRU, nn.RNN},
                dtype=dtype
            )
            
            logger.info(f"Dynamic quantization applied with {dtype}")
            return quantized_model
            
        except Exception as e:
            logger.error(f"Dynamic quantization failed: {e}")
            return model
    
    def quantize_static(
        self,
        model: nn.Module,
        calibration_loader: torch.utils.data.DataLoader,
        dtype: torch.dtype = torch.qint8
    ) -> nn.Module:
        """
        Apply static quantization to model.
        
        Args:
            model: Model to quantize
            calibration_loader: Data loader for calibration
            dtype: Quantization data type
            
        Returns:
            Quantized model
        """
        try:
            model_copy = copy.deepcopy(model)
            model_copy.eval()
            
            # Set quantization config
            model_copy.qconfig = torch.quantization.get_default_qconfig(self.backend)
            
            # Prepare model for quantization
            prepared_model = torch.quantization.prepare(model_copy)
            
            # Calibration
            logger.info("Running calibration...")
            with torch.no_grad():
                for batch_idx, (data, _) in enumerate(calibration_loader):
                    if batch_idx >= 100:  # Limit calibration samples
                        break
                    prepared_model(data)
            
            # Convert to quantized model
            quantized_model = torch.quantization.convert(prepared_model)
            
            logger.info(f"Static quantization applied with {dtype}")
            return quantized_model
            
        except Exception as e:
            logger.error(f"Static quantization failed: {e}")
            return model
    
    def prepare_qat_model(
        self,
        model: nn.Module,
        dtype: torch.dtype = torch.qint8
    ) -> nn.Module:
        """
        Prepare model for quantization-aware training.
        
        Args:
            model: Model to prepare
            dtype: Quantization data type
            
        Returns:
            QAT-prepared model
        """
        try:
            model_copy = copy.deepcopy(model)
            model_copy.train()
            
            # Set QAT config
            model_copy.qconfig = torch.quantization.get_default_qat_qconfig(self.backend)
            
            # Prepare for QAT
            prepared_model = torch.quantization.prepare_qat(model_copy)
            
            logger.info("Model prepared for quantization-aware training")
            return prepared_model
            
        except Exception as e:
            logger.error(f"QAT preparation failed: {e}")
            return model
    
    def convert_qat_model(self, qat_model: nn.Module) -> nn.Module:
        """
        Convert QAT model to quantized model.
        
        Args:
            qat_model: QAT-trained model
            
        Returns:
            Quantized model
        """
        try:
            qat_model.eval()
            quantized_model = torch.quantization.convert(qat_model)
            
            logger.info("QAT model converted to quantized model")
            return quantized_model
            
        except Exception as e:
            logger.error(f"QAT conversion failed: {e}")
            return qat_model
    
    def benchmark_quantization(
        self,
        original_model: nn.Module,
        quantized_model: nn.Module,
        test_data: torch.Tensor,
        num_runs: int = 100
    ) -> Dict[str, Any]:
        """
        Benchmark quantization performance.
        
        Args:
            original_model: Original model
            quantized_model: Quantized model
            test_data: Test data tensor
            num_runs: Number of benchmark runs
            
        Returns:
            Benchmark results dictionary
        """
        results = {}
        
        # Model sizes
        original_size = self._get_model_size(original_model)
        quantized_size = self._get_model_size(quantized_model)
        
        results['original_size_mb'] = original_size
        results['quantized_size_mb'] = quantized_size
        results['compression_ratio'] = original_size / quantized_size
        
        # Inference time comparison
        original_time = self._benchmark_inference(original_model, test_data, num_runs)
        quantized_time = self._benchmark_inference(quantized_model, test_data, num_runs)
        
        results['original_inference_ms'] = original_time * 1000
        results['quantized_inference_ms'] = quantized_time * 1000
        results['speedup'] = original_time / quantized_time
        
        # Accuracy comparison (if same output)
        try:
            with torch.no_grad():
                original_output = original_model(test_data)
                quantized_output = quantized_model(test_data)
                
                mse = torch.mean((original_output - quantized_output) ** 2).item()
                results['output_mse'] = mse
                
        except Exception as e:
            logger.warning(f"Could not compare outputs: {e}")
            results['output_mse'] = None
        
        return results
    
    def _get_model_size(self, model: nn.Module) -> float:
        """Get model size in MB."""
        param_size = 0
        buffer_size = 0
        
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        size_mb = (param_size + buffer_size) / 1024 / 1024
        return size_mb
    
    def _benchmark_inference(
        self,
        model: nn.Module,
        test_data: torch.Tensor,
        num_runs: int
    ) -> float:
        """Benchmark inference time."""
        model.eval()
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model(test_data)
        
        # Benchmark
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(num_runs):
                _ = model(test_data)
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end_time = time.time()
        
        avg_time = (end_time - start_time) / num_runs
        return avg_time
    
    def export_onnx(
        self,
        model: nn.Module,
        example_input: torch.Tensor,
        output_path: str,
        opset_version: int = 11
    ) -> bool:
        """
        Export model to ONNX format.
        
        Args:
            model: Model to export
            example_input: Example input tensor
            output_path: Output ONNX file path
            opset_version: ONNX opset version
            
        Returns:
            True if export successful, False otherwise
        """
        try:
            model.eval()
            
            torch.onnx.export(
                model,
                example_input,
                output_path,
                export_params=True,
                opset_version=opset_version,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                }
            )
            
            logger.info(f"Model exported to ONNX: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"ONNX export failed: {e}")
            return False


class ModelOptimizer:
    """
    Model optimization utility for pruning, knowledge distillation, and other techniques.
    
    Provides structured and unstructured pruning, knowledge distillation,
    and progressive training strategies.
    """
    
    def __init__(self):
        """Initialize model optimizer."""
        self.pruning_history = []
        logger.info("ModelOptimizer initialized")
    
    def prune_unstructured(
        self,
        model: nn.Module,
        pruning_ratio: float = 0.2,
        layer_types: Tuple = (nn.Linear, nn.Conv1d, nn.Conv2d)
    ) -> nn.Module:
        """
        Apply unstructured pruning to model.
        
        Args:
            model: Model to prune
            pruning_ratio: Fraction of weights to prune
            layer_types: Layer types to prune
            
        Returns:
            Pruned model
        """
        try:
            import torch.nn.utils.prune as prune
            
            # Collect layers to prune
            parameters_to_prune = []
            for module in model.modules():
                if isinstance(module, layer_types):
                    parameters_to_prune.append((module, 'weight'))
            
            if not parameters_to_prune:
                logger.warning("No layers found for pruning")
                return model
            
            # Apply global unstructured pruning
            prune.global_unstructured(
                parameters_to_prune,
                pruning_method=prune.L1Unstructured,
                amount=pruning_ratio
            )
            
            # Make pruning permanent
            for module, param_name in parameters_to_prune:
                prune.remove(module, param_name)
            
            self.pruning_history.append({
                'type': 'unstructured',
                'ratio': pruning_ratio,
                'timestamp': time.time()
            })
            
            logger.info(f"Unstructured pruning applied: {pruning_ratio:.1%} of weights removed")
            return model
            
        except ImportError:
            logger.error("torch.nn.utils.prune not available")
            return model
        except Exception as e:
            logger.error(f"Unstructured pruning failed: {e}")
            return model
    
    def prune_structured(
        self,
        model: nn.Module,
        pruning_ratio: float = 0.2,
        dim: int = 0
    ) -> nn.Module:
        """
        Apply structured pruning to model.
        
        Args:
            model: Model to prune
            pruning_ratio: Fraction of channels/neurons to prune
            dim: Dimension to prune along (0 for output, 1 for input)
            
        Returns:
            Pruned model
        """
        try:
            import torch.nn.utils.prune as prune
            
            for module in model.modules():
                if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                    prune.ln_structured(
                        module, 
                        name='weight',
                        amount=pruning_ratio,
                        n=2,
                        dim=dim
                    )
                    prune.remove(module, 'weight')
            
            self.pruning_history.append({
                'type': 'structured',
                'ratio': pruning_ratio,
                'dim': dim,
                'timestamp': time.time()
            })
            
            logger.info(f"Structured pruning applied: {pruning_ratio:.1%} pruning ratio")
            return model
            
        except ImportError:
            logger.error("torch.nn.utils.prune not available")
            return model
        except Exception as e:
            logger.error(f"Structured pruning failed: {e}")
            return model
    
    def knowledge_distillation_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        true_labels: torch.Tensor,
        temperature: float = 3.0,
        alpha: float = 0.7
    ) -> torch.Tensor:
        """
        Compute knowledge distillation loss.
        
        Args:
            student_logits: Student model outputs
            teacher_logits: Teacher model outputs
            true_labels: Ground truth labels
            temperature: Distillation temperature
            alpha: Weight for distillation loss
            
        Returns:
            Combined loss tensor
        """
        # Distillation loss
        soft_teacher = torch.softmax(teacher_logits / temperature, dim=-1)
        soft_student = torch.log_softmax(student_logits / temperature, dim=-1)
        distillation_loss = -torch.sum(soft_teacher * soft_student, dim=-1).mean()
        distillation_loss *= (temperature ** 2)
        
        # Task loss
        task_loss = nn.functional.binary_cross_entropy_with_logits(
            student_logits, true_labels.float()
        )
        
        # Combined loss
        total_loss = alpha * distillation_loss + (1 - alpha) * task_loss
        
        return total_loss
    
    def progressive_shrinking(
        self,
        model: nn.Module,
        target_size_ratio: float = 0.5,
        num_steps: int = 5
    ) -> List[nn.Module]:
        """
        Apply progressive model shrinking.
        
        Args:
            model: Model to shrink
            target_size_ratio: Target size as ratio of original
            num_steps: Number of shrinking steps
            
        Returns:
            List of progressively smaller models
        """
        models = [model]
        current_ratio = 1.0
        step_ratio = (1.0 - target_size_ratio) / num_steps
        
        for step in range(num_steps):
            current_ratio -= step_ratio
            pruning_ratio = 1.0 - current_ratio
            
            # Create a copy and prune it
            step_model = copy.deepcopy(models[-1])
            step_model = self.prune_unstructured(step_model, pruning_ratio)
            models.append(step_model)
            
            logger.info(f"Progressive shrinking step {step + 1}: "
                       f"{pruning_ratio:.1%} total pruning")
        
        return models
    
    def calculate_model_efficiency(
        self,
        model: nn.Module,
        input_size: Tuple[int, ...] = (1, 26),
        device: str = 'cpu'
    ) -> Dict[str, Any]:
        """
        Calculate model efficiency metrics.
        
        Args:
            model: Model to analyze
            input_size: Input tensor size
            device: Device for computation
            
        Returns:
            Efficiency metrics dictionary
        """
        model = model.to(device)
        model.eval()
        
        # Parameter count
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Model size
        model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 / 1024
        
        # FLOPs estimation (simplified)
        flops = self._estimate_flops(model, input_size, device)
        
        # Memory usage
        dummy_input = torch.randn(input_size).to(device)
        
        if device == 'cuda' and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            _ = model(dummy_input)
            peak_memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
        else:
            peak_memory_mb = 0.0
        
        # Inference time
        inference_time = self._measure_inference_time(model, dummy_input)
        
        efficiency_metrics = {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': model_size_mb,
            'estimated_flops': flops,
            'peak_memory_mb': peak_memory_mb,
            'inference_time_ms': inference_time * 1000,
            'efficiency_score': self._calculate_efficiency_score(
                total_params, model_size_mb, inference_time
            )
        }
        
        return efficiency_metrics
    
    def _estimate_flops(
        self,
        model: nn.Module,
        input_size: Tuple[int, ...],
        device: str
    ) -> int:
        """Estimate FLOPs for model (simplified calculation)."""
        flops = 0
        dummy_input = torch.randn(input_size).to(device)
        
        def flop_count_hook(module, input, output):
            nonlocal flops
            if isinstance(module, nn.Linear):
                flops += module.in_features * module.out_features
            elif isinstance(module, (nn.Conv1d, nn.Conv2d)):
                output_elements = output.numel()
                kernel_flops = module.kernel_size[0] if hasattr(module, 'kernel_size') else 1
                flops += output_elements * kernel_flops * module.in_channels
        
        # Register hooks
        hooks = []
        for module in model.modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                hooks.append(module.register_forward_hook(flop_count_hook))
        
        # Forward pass
        with torch.no_grad():
            _ = model(dummy_input)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        return flops
    
    def _measure_inference_time(
        self,
        model: nn.Module,
        dummy_input: torch.Tensor,
        num_runs: int = 100
    ) -> float:
        """Measure average inference time."""
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model(dummy_input)
        
        # Measure
        torch.cuda.synchronize() if dummy_input.is_cuda else None
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(num_runs):
                _ = model(dummy_input)
        
        torch.cuda.synchronize() if dummy_input.is_cuda else None
        end_time = time.time()
        
        return (end_time - start_time) / num_runs
    
    def _calculate_efficiency_score(
        self,
        params: int,
        size_mb: float,
        inference_time: float
    ) -> float:
        """Calculate overall efficiency score (higher is better)."""
        # Normalize metrics (lower values are better, so invert)
        param_score = 1.0 / (1.0 + params / 1000000)  # Normalize by 1M params
        size_score = 1.0 / (1.0 + size_mb)  # Normalize by MB
        speed_score = 1.0 / (1.0 + inference_time * 1000)  # Normalize by ms
        
        # Weighted average
        efficiency_score = 0.4 * param_score + 0.3 * size_score + 0.3 * speed_score
        
        return efficiency_score
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get summary of applied optimizations."""
        return {
            'pruning_history': self.pruning_history,
            'total_optimizations': len(self.pruning_history),
            'last_optimization': self.pruning_history[-1] if self.pruning_history else None
        } 