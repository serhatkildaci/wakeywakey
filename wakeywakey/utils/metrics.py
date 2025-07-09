"""
Performance metrics utilities for wake word detection.

Provides comprehensive metrics calculation and analysis tools
for evaluating wake word detection models.
"""

import time
from typing import Dict, Any, List, Tuple, Optional
import logging

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, 
    roc_auc_score, confusion_matrix, classification_report
)

logger = logging.getLogger(__name__)


class WakeWordMetrics:
    """
    Comprehensive metrics calculator for wake word detection.
    
    Tracks predictions, calculates various performance metrics,
    and provides detailed analysis for model evaluation.
    """
    
    def __init__(self):
        """Initialize metrics calculator."""
        self.reset()
        
    def reset(self):
        """Reset all metrics and stored predictions."""
        self.predictions = []
        self.labels = []
        self.probabilities = []
        self.inference_times = []
        self.timestamps = []
        
    def add_prediction(
        self,
        prediction: int,
        label: int,
        probability: float,
        inference_time: float = 0.0
    ):
        """
        Add a single prediction.
        
        Args:
            prediction: Predicted class (0 or 1)
            label: True label (0 or 1)
            probability: Prediction probability
            inference_time: Inference time in seconds
        """
        self.predictions.append(prediction)
        self.labels.append(label)
        self.probabilities.append(probability)
        self.inference_times.append(inference_time)
        self.timestamps.append(time.time())
    
    def add_batch_predictions(
        self,
        predictions: List[int],
        labels: List[int],
        probabilities: List[float],
        inference_times: Optional[List[float]] = None
    ):
        """
        Add batch of predictions.
        
        Args:
            predictions: List of predicted classes
            labels: List of true labels
            probabilities: List of prediction probabilities
            inference_times: List of inference times (optional)
        """
        if len(predictions) != len(labels) or len(predictions) != len(probabilities):
            raise ValueError("All input lists must have the same length")
        
        self.predictions.extend(predictions)
        self.labels.extend(labels)
        self.probabilities.extend(probabilities)
        
        if inference_times:
            self.inference_times.extend(inference_times)
        else:
            self.inference_times.extend([0.0] * len(predictions))
        
        current_time = time.time()
        self.timestamps.extend([current_time] * len(predictions))
    
    def calculate_basic_metrics(self) -> Dict[str, float]:
        """
        Calculate basic classification metrics.
        
        Returns:
            Dictionary of basic metrics
        """
        if not self.predictions:
            return {}
        
        # Basic metrics
        accuracy = accuracy_score(self.labels, self.predictions)
        
        # Precision, recall, F1
        precision, recall, f1, _ = precision_recall_fscore_support(
            self.labels, self.predictions, average='binary', zero_division=0
        )
        
        # AUC if probabilities available
        try:
            auc = roc_auc_score(self.labels, self.probabilities)
        except ValueError:
            auc = 0.0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc': auc
        }
    
    def calculate_confusion_matrix(self) -> Dict[str, int]:
        """
        Calculate confusion matrix components.
        
        Returns:
            Dictionary with TP, TN, FP, FN counts
        """
        if not self.predictions:
            return {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0}
        
        cm = confusion_matrix(self.labels, self.predictions)
        
        # Handle binary classification
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
        else:
            # Single class case
            if len(set(self.labels)) == 1:
                if self.labels[0] == 1:  # All positive
                    tp = sum(self.predictions)
                    fn = len(self.predictions) - tp
                    tn = fp = 0
                else:  # All negative
                    tn = len(self.predictions) - sum(self.predictions)
                    fp = sum(self.predictions)
                    tp = fn = 0
            else:
                tp = tn = fp = fn = 0
        
        return {
            'tp': int(tp),
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn)
        }
    
    def calculate_rates(self) -> Dict[str, float]:
        """
        Calculate detection rates.
        
        Returns:
            Dictionary with various rate metrics
        """
        cm = self.calculate_confusion_matrix()
        tp, tn, fp, fn = cm['tp'], cm['tn'], cm['fp'], cm['fn']
        
        # Calculate rates
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # True Positive Rate (Sensitivity)
        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # True Negative Rate (Specificity)
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0  # False Positive Rate
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0  # False Negative Rate
        
        return {
            'true_positive_rate': tpr,
            'true_negative_rate': tnr,
            'false_positive_rate': fpr,
            'false_negative_rate': fnr,
            'sensitivity': tpr,
            'specificity': tnr
        }
    
    def calculate_performance_metrics(self) -> Dict[str, float]:
        """
        Calculate performance-related metrics.
        
        Returns:
            Dictionary with performance metrics
        """
        if not self.inference_times:
            return {}
        
        inference_times = [t for t in self.inference_times if t > 0]
        
        if not inference_times:
            return {}
        
        return {
            'avg_inference_time': np.mean(inference_times),
            'min_inference_time': np.min(inference_times),
            'max_inference_time': np.max(inference_times),
            'std_inference_time': np.std(inference_times),
            'median_inference_time': np.median(inference_times),
            'total_predictions': len(self.predictions),
            'predictions_per_second': len(self.predictions) / np.sum(inference_times) if np.sum(inference_times) > 0 else 0
        }
    
    def calculate_threshold_metrics(self, thresholds: List[float] = None) -> Dict[str, Any]:
        """
        Calculate metrics at different thresholds.
        
        Args:
            thresholds: List of thresholds to evaluate
            
        Returns:
            Dictionary with threshold analysis
        """
        if not self.probabilities:
            return {}
        
        if thresholds is None:
            thresholds = np.arange(0.1, 1.0, 0.1).tolist()
        
        threshold_results = {}
        
        for threshold in thresholds:
            # Convert probabilities to predictions at this threshold
            thresh_predictions = [1 if p >= threshold else 0 for p in self.probabilities]
            
            # Calculate metrics
            accuracy = accuracy_score(self.labels, thresh_predictions)
            precision, recall, f1, _ = precision_recall_fscore_support(
                self.labels, thresh_predictions, average='binary', zero_division=0
            )
            
            threshold_results[f"threshold_{threshold:.1f}"] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            }
        
        return threshold_results
    
    def find_optimal_threshold(self, metric: str = 'f1_score') -> Tuple[float, float]:
        """
        Find optimal threshold for given metric.
        
        Args:
            metric: Metric to optimize ('f1_score', 'accuracy', 'precision', 'recall')
            
        Returns:
            Tuple of (optimal_threshold, best_metric_value)
        """
        if not self.probabilities:
            return 0.5, 0.0
        
        thresholds = np.arange(0.05, 0.96, 0.05)
        best_threshold = 0.5
        best_value = 0.0
        
        for threshold in thresholds:
            thresh_predictions = [1 if p >= threshold else 0 for p in self.probabilities]
            
            if metric == 'accuracy':
                value = accuracy_score(self.labels, thresh_predictions)
            elif metric == 'f1_score':
                _, _, value, _ = precision_recall_fscore_support(
                    self.labels, thresh_predictions, average='binary', zero_division=0
                )
            elif metric == 'precision':
                value, _, _, _ = precision_recall_fscore_support(
                    self.labels, thresh_predictions, average='binary', zero_division=0
                )
            elif metric == 'recall':
                _, value, _, _ = precision_recall_fscore_support(
                    self.labels, thresh_predictions, average='binary', zero_division=0
                )
            else:
                raise ValueError(f"Unknown metric: {metric}")
            
            if value > best_value:
                best_value = value
                best_threshold = threshold
        
        return best_threshold, best_value
    
    def get_classification_report(self) -> str:
        """
        Get detailed classification report.
        
        Returns:
            Classification report string
        """
        if not self.predictions:
            return "No predictions available"
        
        return classification_report(
            self.labels, self.predictions,
            target_names=['Not Wake Word', 'Wake Word'],
            zero_division=0
        )
    
    def calculate_all_metrics(self) -> Dict[str, Any]:
        """
        Calculate all available metrics.
        
        Returns:
            Comprehensive metrics dictionary
        """
        if not self.predictions:
            return {'error': 'No predictions available'}
        
        metrics = {}
        
        # Basic metrics
        metrics.update(self.calculate_basic_metrics())
        
        # Confusion matrix
        metrics['confusion_matrix'] = self.calculate_confusion_matrix()
        
        # Rates
        metrics.update(self.calculate_rates())
        
        # Performance metrics
        performance = self.calculate_performance_metrics()
        if performance:
            metrics['performance'] = performance
        
        # Threshold analysis
        threshold_metrics = self.calculate_threshold_metrics()
        if threshold_metrics:
            metrics['threshold_analysis'] = threshold_metrics
        
        # Optimal threshold
        opt_threshold, opt_value = self.find_optimal_threshold()
        metrics['optimal_threshold'] = {
            'threshold': opt_threshold,
            'f1_score': opt_value
        }
        
        # Data summary
        metrics['data_summary'] = {
            'total_samples': len(self.predictions),
            'positive_samples': sum(self.labels),
            'negative_samples': len(self.labels) - sum(self.labels),
            'positive_ratio': sum(self.labels) / len(self.labels) if self.labels else 0,
            'prediction_positive_ratio': sum(self.predictions) / len(self.predictions) if self.predictions else 0
        }
        
        return metrics
    
    def get_summary_string(self) -> str:
        """
        Get human-readable summary string.
        
        Returns:
            Formatted summary string
        """
        if not self.predictions:
            return "No predictions available for summary"
        
        metrics = self.calculate_all_metrics()
        
        summary = []
        summary.append("=== Wake Word Detection Metrics Summary ===")
        summary.append(f"Total Samples: {metrics['data_summary']['total_samples']}")
        summary.append(f"Positive Samples: {metrics['data_summary']['positive_samples']}")
        summary.append(f"Negative Samples: {metrics['data_summary']['negative_samples']}")
        summary.append("")
        
        summary.append("Performance Metrics:")
        summary.append(f"  Accuracy:  {metrics['accuracy']:.4f}")
        summary.append(f"  Precision: {metrics['precision']:.4f}")
        summary.append(f"  Recall:    {metrics['recall']:.4f}")
        summary.append(f"  F1-Score:  {metrics['f1_score']:.4f}")
        summary.append(f"  AUC:       {metrics['auc']:.4f}")
        summary.append("")
        
        cm = metrics['confusion_matrix']
        summary.append("Confusion Matrix:")
        summary.append(f"  True Positives:  {cm['tp']}")
        summary.append(f"  True Negatives:  {cm['tn']}")
        summary.append(f"  False Positives: {cm['fp']}")
        summary.append(f"  False Negatives: {cm['fn']}")
        summary.append("")
        
        if 'performance' in metrics:
            perf = metrics['performance']
            summary.append("Performance:")
            summary.append(f"  Avg Inference Time: {perf['avg_inference_time']*1000:.2f} ms")
            summary.append(f"  Predictions/sec:    {perf['predictions_per_second']:.1f}")
        
        summary.append("")
        summary.append(f"Optimal Threshold: {metrics['optimal_threshold']['threshold']:.2f} "
                      f"(F1: {metrics['optimal_threshold']['f1_score']:.4f})")
        
        return "\n".join(summary)


def calculate_metrics(
    predictions: List[int],
    labels: List[int],
    probabilities: Optional[List[float]] = None,
    inference_times: Optional[List[float]] = None
) -> Dict[str, Any]:
    """
    Convenience function to calculate metrics from lists.
    
    Args:
        predictions: List of predicted classes
        labels: List of true labels
        probabilities: List of prediction probabilities (optional)
        inference_times: List of inference times (optional)
        
    Returns:
        Comprehensive metrics dictionary
    """
    metrics_calc = WakeWordMetrics()
    
    if probabilities is None:
        probabilities = [float(p) for p in predictions]
    
    if inference_times is None:
        inference_times = [0.0] * len(predictions)
    
    metrics_calc.add_batch_predictions(
        predictions, labels, probabilities, inference_times
    )
    
    return metrics_calc.calculate_all_metrics()


def calculate_detection_latency(
    timestamps: List[float],
    wake_word_times: List[float],
    detection_times: List[float],
    max_latency: float = 2.0
) -> Dict[str, float]:
    """
    Calculate wake word detection latency metrics.
    
    Args:
        timestamps: Audio timestamps
        wake_word_times: Ground truth wake word occurrence times
        detection_times: Detection timestamps
        max_latency: Maximum acceptable latency in seconds
        
    Returns:
        Latency metrics dictionary
    """
    latencies = []
    missed_detections = 0
    false_alarms = 0
    
    for wake_time in wake_word_times:
        # Find closest detection within max_latency
        valid_detections = [
            d for d in detection_times 
            if wake_time <= d <= wake_time + max_latency
        ]
        
        if valid_detections:
            # Found valid detection, calculate latency
            closest_detection = min(valid_detections, key=lambda x: abs(x - wake_time))
            latency = closest_detection - wake_time
            latencies.append(latency)
        else:
            # Missed detection
            missed_detections += 1
    
    # Count false alarms (detections not near any wake word)
    for det_time in detection_times:
        near_wake_word = any(
            abs(det_time - wake_time) <= max_latency 
            for wake_time in wake_word_times
        )
        if not near_wake_word:
            false_alarms += 1
    
    if latencies:
        return {
            'avg_latency': np.mean(latencies),
            'min_latency': np.min(latencies),
            'max_latency': np.max(latencies),
            'std_latency': np.std(latencies),
            'median_latency': np.median(latencies),
            'detection_rate': len(latencies) / len(wake_word_times),
            'missed_detections': missed_detections,
            'false_alarms': false_alarms,
            'false_alarm_rate': false_alarms / len(detection_times) if detection_times else 0
        }
    else:
        return {
            'avg_latency': float('inf'),
            'detection_rate': 0.0,
            'missed_detections': len(wake_word_times),
            'false_alarms': len(detection_times),
            'false_alarm_rate': 1.0 if detection_times else 0.0
        } 