"""
WakeyWakey CLI - Terminal interface for wake word detection.

A comprehensive command-line interface for training, testing, and deploying
wake word detection models. Features terminal-style output and technical
monitoring capabilities.
"""

import sys
import os
import time
import json
import argparse
import threading
from pathlib import Path
from typing import Optional, Dict, Any, List
import logging

# Terminal styling and formatting
try:
    import colorama
    from colorama import Fore, Back, Style
    colorama.init()
    COLORS_AVAILABLE = True
except ImportError:
    COLORS_AVAILABLE = False
    # Fallback color codes
    class Fore:
        RED = ""
        GREEN = ""
        YELLOW = ""
        BLUE = ""
        MAGENTA = ""
        CYAN = ""
        WHITE = ""
        RESET = ""
    
    class Style:
        BRIGHT = ""
        DIM = ""
        RESET_ALL = ""

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from wakeywakey.core import WakeWordDetector, AudioProcessor
from wakeywakey.training import Trainer, WakeWordDataset
from wakeywakey.models import LightweightCNN, CompactRNN, HybridCRNN, MobileWakeWord

# Configure logging with custom formatter
class TerminalFormatter(logging.Formatter):
    """Custom formatter for terminal-style logging."""
    
    def format(self, record):
        timestamp = self.formatTime(record, '%H:%M:%S')
        level_colors = {
            'DEBUG': Fore.CYAN,
            'INFO': Fore.GREEN,
            'WARNING': Fore.YELLOW,
            'ERROR': Fore.RED,
            'CRITICAL': Fore.MAGENTA + Style.BRIGHT
        }
        
        level_color = level_colors.get(record.levelname, Fore.WHITE)
        
        return (f"{Fore.WHITE}[{timestamp}] "
               f"{level_color}[{record.levelname:^8}] "
               f"{Fore.WHITE}{record.getMessage()}{Style.RESET_ALL}")


class WakeyWakeyCLI:
    """
    WakeyWakey Command Line Interface.
    
    Terminal-style interface for wake word detection with real-time monitoring,
    training capabilities, and deployment tools.
    """
    
    def __init__(self):
        self.detector: Optional[WakeWordDetector] = None
        self.is_monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.stats_display_interval = 1.0
        
        # Setup logging
        self.setup_logging()
        
        # Terminal configuration
        self.terminal_width = os.get_terminal_size().columns if hasattr(os, 'get_terminal_size') else 80
        
    def setup_logging(self):
        """Setup terminal-style logging."""
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        
        # Remove existing handlers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # Add custom terminal handler
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(TerminalFormatter())
        logger.addHandler(handler)
    
    def print_banner(self):
        """Print ASCII banner with system info."""
        banner = f"""
{Fore.CYAN}{Style.BRIGHT}
██╗    ██╗ █████╗ ██╗  ██╗███████╗██╗   ██╗    ██╗    ██╗ █████╗ ██╗  ██╗███████╗██╗   ██╗
██║    ██║██╔══██╗██║ ██╔╝██╔════╝╚██╗ ██╔╝    ██║    ██║██╔══██╗██║ ██╔╝██╔════╝╚██╗ ██╔╝
██║ █╗ ██║███████║█████╔╝ █████╗   ╚████╔╝     ██║ █╗ ██║███████║█████╔╝ █████╗   ╚████╔╝ 
██║███╗██║██╔══██║██╔═██╗ ██╔══╝    ╚██╔╝      ██║███╗██║██╔══██║██╔═██╗ ██╔══╝    ╚██╔╝  
╚███╔███╔╝██║  ██║██║  ██╗███████╗   ██║       ╚███╔███╔╝██║  ██║██║  ██╗███████╗   ██║   
 ╚══╝╚══╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝   ╚═╝        ╚══╝╚══╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝   ╚═╝   
{Style.RESET_ALL}
{Fore.WHITE}{Style.DIM}                    Lightweight Wake Word Detection System
                           Optimized for Microcontrollers to Raspberry Pi{Style.RESET_ALL}

"""
        print(banner)
        
        # System information
        try:
            import torch
            device = "CUDA" if torch.cuda.is_available() else "CPU"
            if torch.cuda.is_available():
                gpu_info = torch.cuda.get_device_name(0)
                device += f" ({gpu_info})"
        except ImportError:
            device = "CPU (PyTorch not available)"
        
        print(f"{Fore.CYAN}[SYSTEM]{Style.RESET_ALL} Device: {device}")
        print(f"{Fore.CYAN}[SYSTEM]{Style.RESET_ALL} Terminal: {self.terminal_width} cols")
        print("─" * self.terminal_width)
        print()
    
    def print_section_header(self, title: str, subtitle: str = ""):
        """Print styled section header."""
        print(f"\n{Fore.YELLOW}{Style.BRIGHT}▶ {title.upper()}{Style.RESET_ALL}")
        if subtitle:
            print(f"{Fore.WHITE}{Style.DIM}  {subtitle}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}{'─' * (len(title) + 2)}{Style.RESET_ALL}")
    
    def print_progress_bar(self, current: int, total: int, prefix: str = "", width: int = 40):
        """Print terminal progress bar."""
        percentage = current / total
        filled_width = int(width * percentage)
        bar = "█" * filled_width + "░" * (width - filled_width)
        
        print(f"\r{prefix} [{Fore.GREEN}{bar}{Style.RESET_ALL}] "
              f"{percentage:.1%} ({current}/{total})", end="", flush=True)
    
    def list_models(self, model_dir: str):
        """List available models."""
        self.print_section_header("Available Models", f"Scanning {model_dir}")
        
        from wakeywakey.core.detector import ModelLoader
        models = ModelLoader.list_models(model_dir)
        
        if not models:
            print(f"{Fore.YELLOW}⚠ No models found in {model_dir}{Style.RESET_ALL}")
            return
        
        # Table header
        print(f"\n{Fore.CYAN}{Style.BRIGHT}{'NAME':<20} {'TYPE':<15} {'SIZE':<8} {'ACCURACY':<10} {'CREATED':<20}{Style.RESET_ALL}")
        print("─" * 73)
        
        for model in models:
            created = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(model['created']))
            accuracy = f"{model['accuracy']:.1%}" if isinstance(model['accuracy'], float) else str(model['accuracy'])
            
            print(f"{Fore.WHITE}{model['name']:<20} "
                  f"{Fore.CYAN}{model['type']:<15} "
                  f"{Fore.YELLOW}{model['size_mb']:.1f}MB{'':<3} "
                  f"{Fore.GREEN}{accuracy:<10} "
                  f"{Fore.WHITE}{Style.DIM}{created}{Style.RESET_ALL}")
    
    def train_model(self, args):
        """Train a wake word detection model."""
        self.print_section_header("Model Training", f"Architecture: {args.model_type}")
        
        try:
            # Setup training configuration
            config = {
                'model_type': args.model_type,
                'epochs': args.epochs,
                'batch_size': args.batch_size,
                'learning_rate': args.learning_rate,
                'data_dir': args.data_dir,
                'output_dir': args.output_dir
            }
            
            print(f"{Fore.CYAN}[CONFIG]{Style.RESET_ALL} Training configuration:")
            for key, value in config.items():
                print(f"  {key}: {value}")
            
            # Initialize trainer
            trainer = Trainer(config)
            
            # Load dataset
            print(f"\n{Fore.CYAN}[DATA]{Style.RESET_ALL} Loading dataset...")
            dataset = WakeWordDataset(args.data_dir)
            
            print(f"{Fore.GREEN}[DATA]{Style.RESET_ALL} Dataset loaded:")
            print(f"  Positive samples: {len(dataset.positive_samples)}")
            print(f"  Negative samples: {len(dataset.negative_samples)}")
            
            # Start training
            print(f"\n{Fore.CYAN}[TRAIN]{Style.RESET_ALL} Starting training...")
            
            def progress_callback(epoch, loss, accuracy, lr):
                print(f"\r{Fore.CYAN}[EPOCH {epoch:3d}]{Style.RESET_ALL} "
                      f"Loss: {Fore.RED}{loss:.4f}{Style.RESET_ALL} "
                      f"Acc: {Fore.GREEN}{accuracy:.3f}{Style.RESET_ALL} "
                      f"LR: {Fore.YELLOW}{lr:.2e}{Style.RESET_ALL}", end="", flush=True)
            
            results = trainer.train(dataset, progress_callback=progress_callback)
            
            print(f"\n\n{Fore.GREEN}[SUCCESS]{Style.RESET_ALL} Training completed!")
            print(f"Best accuracy: {Fore.GREEN}{results['best_accuracy']:.3f}{Style.RESET_ALL}")
            print(f"Model saved to: {Fore.CYAN}{results['model_path']}{Style.RESET_ALL}")
            
        except Exception as e:
            print(f"\n{Fore.RED}[ERROR]{Style.RESET_ALL} Training failed: {e}")
            return False
        
        return True
    
    def test_model(self, args):
        """Test a model on audio files."""
        self.print_section_header("Model Testing", f"Model: {args.model}")
        
        try:
            # Load detector
            detector = WakeWordDetector(
                model_path=args.model,
                model_type=args.model_type,
                threshold=args.threshold
            )
            
            print(f"{Fore.CYAN}[MODEL]{Style.RESET_ALL} Model loaded successfully")
            
            # Test on files
            test_files = []
            if os.path.isfile(args.input):
                test_files = [args.input]
            elif os.path.isdir(args.input):
                for ext in ['*.wav', '*.mp3', '*.flac']:
                    test_files.extend(Path(args.input).glob(ext))
            
            if not test_files:
                print(f"{Fore.YELLOW}⚠ No audio files found{Style.RESET_ALL}")
                return
            
            print(f"\n{Fore.CYAN}[TEST]{Style.RESET_ALL} Testing {len(test_files)} files...")
            
            results = []
            for i, file_path in enumerate(test_files):
                self.print_progress_bar(i, len(test_files), "Processing")
                
                result = detector.test_detection(str(file_path))
                results.append(result)
            
            print()  # New line after progress bar
            
            # Display results
            print(f"\n{Fore.CYAN}{Style.BRIGHT}{'FILE':<30} {'CONFIDENCE':<12} {'DETECTED':<10} {'TIME(ms)':<10}{Style.RESET_ALL}")
            print("─" * 62)
            
            total_detected = 0
            for result in results:
                if 'error' in result:
                    continue
                
                filename = Path(result['file']).name[:28]
                confidence = result['confidence']
                detected = result['detected']
                proc_time = result['processing_time_ms']
                
                if detected:
                    total_detected += 1
                
                conf_color = Fore.GREEN if confidence > args.threshold else Fore.RED
                det_color = Fore.GREEN if detected else Fore.RED
                
                print(f"{Fore.WHITE}{filename:<30} "
                      f"{conf_color}{confidence:.3f}{Style.RESET_ALL}       "
                      f"{det_color}{'YES' if detected else 'NO':<10}{Style.RESET_ALL} "
                      f"{Fore.YELLOW}{proc_time:.1f}{Style.RESET_ALL}")
            
            # Summary
            print(f"\n{Fore.CYAN}[SUMMARY]{Style.RESET_ALL}")
            print(f"  Total files: {len(results)}")
            print(f"  Detected: {Fore.GREEN}{total_detected}{Style.RESET_ALL}")
            print(f"  Detection rate: {Fore.YELLOW}{total_detected/len(results):.1%}{Style.RESET_ALL}")
            
        except Exception as e:
            print(f"{Fore.RED}[ERROR]{Style.RESET_ALL} Testing failed: {e}")
            return False
        
        return True
    
    def start_realtime_detection(self, args):
        """Start real-time wake word detection."""
        self.print_section_header("Real-time Detection", f"Model: {Path(args.model).name}")
        
        try:
            # Initialize detector
            self.detector = WakeWordDetector(
                model_path=args.model,
                model_type=args.model_type,
                threshold=args.threshold
            )
            
            print(f"{Fore.CYAN}[MODEL]{Style.RESET_ALL} Model loaded successfully")
            print(f"{Fore.CYAN}[AUDIO]{Style.RESET_ALL} Initializing microphone...")
            
            # Detection callback
            def on_detection(info):
                timestamp = time.strftime('%H:%M:%S', time.localtime(info['timestamp']))
                confidence = info['confidence']
                proc_time = info['processing_time'] * 1000
                
                print(f"\n{Fore.GREEN}{Style.BRIGHT}🔊 WAKE WORD DETECTED!{Style.RESET_ALL}")
                print(f"{Fore.WHITE}[{timestamp}] Confidence: {Fore.GREEN}{confidence:.3f}{Style.RESET_ALL} "
                      f"({proc_time:.1f}ms)")
                print(f"{Fore.CYAN}[MONITOR]{Style.RESET_ALL} ", end="", flush=True)
            
            # Debug callback for monitoring
            def on_debug(info):
                if args.verbose:
                    conf = info['confidence']
                    smooth_conf = info['smoothed_confidence']
                    proc_time = info['processing_time'] * 1000
                    
                    conf_bar = "█" * int(conf * 20)
                    conf_bar += "░" * (20 - len(conf_bar))
                    
                    print(f"\r{Fore.CYAN}[MONITOR]{Style.RESET_ALL} "
                          f"[{Fore.GREEN if conf > args.threshold else Fore.RED}{conf_bar}{Style.RESET_ALL}] "
                          f"Raw: {conf:.3f} Smooth: {smooth_conf:.3f} "
                          f"({proc_time:.1f}ms)", end="", flush=True)
            
            # Start monitoring thread
            self.is_monitoring = True
            self.monitor_thread = threading.Thread(target=self.monitor_stats, daemon=True)
            self.monitor_thread.start()
            
            # Start detection
            success = self.detector.start_detection(
                detection_callback=on_detection,
                debug_callback=on_debug if args.verbose else None
            )
            
            if not success:
                print(f"{Fore.RED}[ERROR]{Style.RESET_ALL} Failed to start detection")
                return False
            
            print(f"{Fore.GREEN}[READY]{Style.RESET_ALL} Listening for wake word...")
            print(f"{Fore.WHITE}{Style.DIM}Press Ctrl+C to stop{Style.RESET_ALL}")
            
            if not args.verbose:
                print(f"{Fore.CYAN}[MONITOR]{Style.RESET_ALL} ", end="", flush=True)
            
            # Keep alive
            try:
                while True:
                    time.sleep(0.1)
            except KeyboardInterrupt:
                print(f"\n\n{Fore.YELLOW}[STOP]{Style.RESET_ALL} Stopping detection...")
                self.stop_detection()
                
        except Exception as e:
            print(f"{Fore.RED}[ERROR]{Style.RESET_ALL} Detection failed: {e}")
            return False
        
        return True
    
    def monitor_stats(self):
        """Monitor and display detection statistics."""
        while self.is_monitoring and self.detector:
            time.sleep(self.stats_display_interval)
            
            if not self.detector.is_detecting:
                continue
            
            stats = self.detector.get_stats()
            
            # Only update every few seconds for non-verbose mode
            if stats['runtime_seconds'] % 5 < self.stats_display_interval:
                detections = stats['total_detections']
                runtime = stats['runtime_seconds'] / 60  # Convert to minutes
                avg_time = stats['average_processing_time_ms']
                
                status = f"Detections: {detections} | Runtime: {runtime:.1f}m | Avg: {avg_time:.1f}ms"
                
                # Only print if not in verbose mode (to avoid interfering with debug output)
                if not hasattr(self, '_verbose_mode') or not self._verbose_mode:
                    print(f"\r{Fore.CYAN}[MONITOR]{Style.RESET_ALL} {status}", end="", flush=True)
    
    def stop_detection(self):
        """Stop detection and cleanup."""
        self.is_monitoring = False
        
        if self.detector:
            self.detector.stop_detection()
            
            # Show final stats
            stats = self.detector.get_stats()
            print(f"\n{Fore.CYAN}[STATS]{Style.RESET_ALL} Final Statistics:")
            print(f"  Total detections: {stats['total_detections']}")
            print(f"  Runtime: {stats['runtime_seconds']:.1f}s")
            print(f"  Avg processing time: {stats['average_processing_time_ms']:.1f}ms")
            print(f"  Detections/min: {stats['detections_per_minute']:.1f}")
    
    def list_devices(self):
        """List available audio devices."""
        self.print_section_header("Audio Devices")
        
        try:
            processor = AudioProcessor()
            devices = processor.get_audio_devices()
            
            print(f"{Fore.CYAN}{Style.BRIGHT}{'ID':<4} {'NAME':<40} {'CHANNELS':<10} {'TYPE':<10}{Style.RESET_ALL}")
            print("─" * 64)
            
            for device in devices:
                device_type = "Input" if device['max_input_channels'] > 0 else "Output"
                channels = device['max_input_channels'] if device_type == "Input" else device['max_output_channels']
                
                print(f"{Fore.YELLOW}{device['index']:<4} "
                      f"{Fore.WHITE}{device['name'][:38]:<40} "
                      f"{Fore.CYAN}{channels:<10} "
                      f"{Fore.GREEN if device_type == 'Input' else Fore.RED}{device_type}{Style.RESET_ALL}")
                
        except Exception as e:
            print(f"{Fore.RED}[ERROR]{Style.RESET_ALL} Failed to list devices: {e}")


def create_parser():
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description="WakeyWakey - Lightweight Wake Word Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  wakeywakey train --data-dir ./data --model-type LightweightCNN --epochs 50
  wakeywakey test --model ./models/best_model.pth --input ./test_audio/
  wakeywakey detect --model ./models/best_model.pth --threshold 0.7
  wakeywakey list-models --model-dir ./models/
  wakeywakey list-devices
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train a wake word detection model')
    train_parser.add_argument('--data-dir', required=True, help='Directory containing training data')
    train_parser.add_argument('--model-type', default='LightweightCNN',
                             choices=['LightweightCNN', 'CompactRNN', 'HybridCRNN', 'MobileWakeWord'],
                             help='Model architecture to use')
    train_parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    train_parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training')
    train_parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')
    train_parser.add_argument('--output-dir', default='./models', help='Output directory for trained models')
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Test a model on audio files')
    test_parser.add_argument('--model', required=True, help='Path to trained model')
    test_parser.add_argument('--input', required=True, help='Audio file or directory to test')
    test_parser.add_argument('--model-type', default='LightweightCNN',
                            choices=['LightweightCNN', 'CompactRNN', 'HybridCRNN', 'MobileWakeWord'],
                            help='Model architecture type')
    test_parser.add_argument('--threshold', type=float, default=0.7, help='Detection threshold')
    
    # Detect command
    detect_parser = subparsers.add_parser('detect', help='Real-time wake word detection')
    detect_parser.add_argument('--model', required=True, help='Path to trained model')
    detect_parser.add_argument('--model-type', default='LightweightCNN',
                              choices=['LightweightCNN', 'CompactRNN', 'HybridCRNN', 'MobileWakeWord'],
                              help='Model architecture type')
    detect_parser.add_argument('--threshold', type=float, default=0.7, help='Detection threshold')
    detect_parser.add_argument('--sensitivity', choices=['low', 'medium', 'high', 'very_high'],
                              help='Detection sensitivity preset')
    detect_parser.add_argument('--verbose', action='store_true', help='Show detailed monitoring info')
    detect_parser.add_argument('--device-id', type=int, help='Audio input device ID')
    
    # List models command
    list_models_parser = subparsers.add_parser('list-models', help='List available models')
    list_models_parser.add_argument('--model-dir', default='./models', help='Directory containing models')
    
    # List devices command
    subparsers.add_parser('list-devices', help='List available audio devices')
    
    return parser


def main():
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Initialize CLI
    cli = WakeyWakeyCLI()
    cli.print_banner()
    
    try:
        if args.command == 'train':
            success = cli.train_model(args)
        elif args.command == 'test':
            success = cli.test_model(args)
        elif args.command == 'detect':
            cli._verbose_mode = args.verbose  # Store for monitoring
            if args.sensitivity:
                # Convert sensitivity to threshold
                sensitivity_map = {'low': 0.9, 'medium': 0.7, 'high': 0.5, 'very_high': 0.3}
                args.threshold = sensitivity_map[args.sensitivity]
            success = cli.start_realtime_detection(args)
        elif args.command == 'list-models':
            cli.list_models(args.model_dir)
            success = True
        elif args.command == 'list-devices':
            cli.list_devices()
            success = True
        else:
            print(f"{Fore.RED}[ERROR]{Style.RESET_ALL} Unknown command: {args.command}")
            success = False
        
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}[INTERRUPTED]{Style.RESET_ALL} Operation cancelled by user")
        sys.exit(130)
    except Exception as e:
        print(f"{Fore.RED}[FATAL]{Style.RESET_ALL} Unexpected error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main() 