#!/usr/bin/env python3
"""
Evaluation script for Nepali student model.
Provides comprehensive evaluation across multiple Nepali language tasks.
"""

import os
import sys
import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Any

import torch
import yaml

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from model.student_architecture import NepaliStudentModel, NepaliStudentConfig
from evaluation.benchmarks import NepaliEvaluator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_model_from_checkpoint(checkpoint_path: str, device: torch.device):
    """Load model from checkpoint or saved model directory."""
    
    checkpoint_path = Path(checkpoint_path)
    
    if checkpoint_path.is_file() and checkpoint_path.suffix == '.pt':
        # Load from checkpoint file
        logger.info(f"📂 Loading from checkpoint file: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Extract model config from checkpoint
        if 'config' in checkpoint:
            model_config_dict = checkpoint['config'].get('model', {}).get('student_model', {})
        else:
            logger.error("❌ No config found in checkpoint")
            return None
        
        # Create model
        model_config = NepaliStudentConfig(**model_config_dict)
        model = NepaliStudentModel(model_config)
        
        # Load state dict
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load additional info
        training_info = {
            'global_step': checkpoint.get('global_step', 0),
            'epochs_trained': checkpoint.get('epochs_trained', 0),
            'best_metric': checkpoint.get('best_metric', 0),
        }
        
        logger.info(f"✅ Loaded checkpoint from step {training_info['global_step']}")
        
    elif checkpoint_path.is_dir():
        # Load from saved model directory
        logger.info(f"📁 Loading from model directory: {checkpoint_path}")
        
        config_file = checkpoint_path / "config.json"
        model_file = checkpoint_path / "pytorch_model.bin"
        training_info_file = checkpoint_path / "training_info.json"
        
        if not config_file.exists():
            logger.error(f"❌ Model config not found: {config_file}")
            return None
        
        if not model_file.exists():
            logger.error(f"❌ Model weights not found: {model_file}")
            return None
        
        # Load config
        with open(config_file, 'r') as f:
            model_config_dict = json.load(f)
        
        model_config = NepaliStudentConfig(**model_config_dict)
        model = NepaliStudentModel(model_config)
        
        # Load weights
        model.load_state_dict(torch.load(model_file, map_location=device))
        
        # Load training info if available
        training_info = {}
        if training_info_file.exists():
            with open(training_info_file, 'r') as f:
                training_info = json.load(f)
        
        logger.info("✅ Model loaded successfully from directory")
        
    else:
        logger.error(f"❌ Invalid checkpoint path: {checkpoint_path}")
        return None
    
    model.to(device)
    model.eval()
    
    return model, training_info


def setup_device(device_name: str = "auto") -> torch.device:
    """Setup optimal device for evaluation."""
    
    if device_name == "auto":
        if torch.backends.mps.is_available():
            device = torch.device("mps")
            logger.info("✅ Using Metal Performance Shaders (MPS)")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"✅ Using CUDA: {torch.cuda.get_device_name()}")
        else:
            device = torch.device("cpu")
            logger.info("⚠️ Using CPU")
    else:
        device = torch.device(device_name)
        logger.info(f"🔧 Using specified device: {device}")
    
    return device


def load_eval_config(config_path: str) -> Dict[str, Any]:
    """Load evaluation configuration."""
    
    if not Path(config_path).exists():
        logger.warning(f"⚠️ Eval config not found: {config_path}")
        logger.info("📝 Using default evaluation configuration")
        
        # Default configuration
        return {
            'benchmarks': [
                {'name': 'nepali_qa', 'metric': 'exact_match', 'weight': 0.3},
                {'name': 'nepali_translation', 'metric': 'bleu', 'weight': 0.3},
                {'name': 'nepali_summarization', 'metric': 'rouge_l', 'weight': 0.2},
                {'name': 'nepali_sentiment', 'metric': 'f1', 'weight': 0.2}
            ],
            'thresholds': {
                'minimum_performance': 0.8,
                'speed_improvement': 10,
                'model_size_mb': 800
            }
        }
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info(f"✅ Loaded evaluation config from {config_path}")
    return config


def print_model_info(model: NepaliStudentModel, training_info: Dict[str, Any]):
    """Print comprehensive model information."""
    
    memory_footprint = model.get_memory_footprint()
    
    print("\n" + "="*60)
    print("🧠 MODEL INFORMATION")
    print("="*60)
    print(f"Model Type: {model.__class__.__name__}")
    print(f"Architecture: {model.config.num_hidden_layers} layers × {model.config.num_attention_heads} heads")
    print(f"Hidden Size: {model.config.hidden_size}")
    print(f"Vocab Size: {model.config.vocab_size:,}")
    print(f"Max Length: {model.config.max_position_embeddings:,}")
    print(f"Total Parameters: {memory_footprint['total_parameters']:,}")
    print(f"Model Size: {memory_footprint['memory_mb']:.1f} MB")
    print(f"Model Size: {memory_footprint['memory_gb']:.3f} GB")
    
    # Training information
    if training_info:
        print("-" * 60)
        print("🎓 TRAINING INFORMATION")
        print("-" * 60)
        if 'training_steps' in training_info:
            print(f"Training Steps: {training_info['training_steps']:,}")
        if 'epochs_trained' in training_info:
            print(f"Epochs Trained: {training_info['epochs_trained']}")
        if 'best_metric' in training_info:
            print(f"Best Metric: {training_info['best_metric']:.4f}")
    
    print("="*60)


def print_evaluation_results(results: Dict[str, Any], config: Dict[str, Any]):
    """Print detailed evaluation results."""
    
    print("\n" + "="*60)
    print("📊 EVALUATION RESULTS")
    print("="*60)
    
    # Overall performance
    if 'overall_score' in results:
        print(f"Overall Performance: {results['overall_score']:.4f}")
        print("-" * 60)
    
    # Task-specific results
    benchmarks = config.get('benchmarks', [])
    
    if benchmarks:
        print("TASK-SPECIFIC PERFORMANCE:")
        for benchmark in benchmarks:
            task_name = benchmark['name']
            metric_key = f"{task_name}_score"
            weight = benchmark.get('weight', 1.0)
            
            if metric_key in results:
                score = results[metric_key]
                print(f"  {task_name.replace('_', ' ').title():.<35} {score:.4f} (weight: {weight})")
        
        print("-" * 60)
    
    # Performance metrics
    performance_metrics = [
        ('model_size_mb', 'Model Size (MB)'),
        ('total_parameters', 'Total Parameters'),
        ('inference_speed_tokens_per_second', 'Inference Speed (tokens/sec)'),
    ]
    
    print("PERFORMANCE METRICS:")
    for metric_key, metric_name in performance_metrics:
        if metric_key in results:
            value = results[metric_key]
            if metric_key == 'total_parameters':
                print(f"  {metric_name:.<35} {value:,.0f}")
            elif isinstance(value, float):
                print(f"  {metric_name:.<35} {value:.2f}")
            else:
                print(f"  {metric_name:.<35} {value}")
    
    print("="*60)
    
    # Requirement checks
    print("\n🎯 REQUIREMENT ANALYSIS")
    print("="*60)
    
    thresholds = config.get('thresholds', {})
    
    checks = [
        ('Performance Retention', results.get('overall_score', 0), 
         thresholds.get('minimum_performance', 0.8), '≥'),
        ('Model Size (MB)', results.get('model_size_mb', float('inf')), 
         thresholds.get('model_size_mb', 800), '≤'),
    ]
    
    # Speed improvement check (if baseline available)
    baseline_speed = 50  # Assumed baseline tokens/sec
    current_speed = results.get('inference_speed_tokens_per_second', 0)
    if current_speed > 0:
        speed_improvement = current_speed / baseline_speed
        checks.append(('Speed Improvement', speed_improvement, 
                      thresholds.get('speed_improvement', 10), '≥'))
    
    all_passed = True
    
    for check_name, actual, threshold, operator in checks:
        if operator == '≥':
            passed = actual >= threshold
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"{status} {check_name}: {actual:.2f} {operator} {threshold}")
        else:  # operator == '≤'
            passed = actual <= threshold
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"{status} {check_name}: {actual:.2f} {operator} {threshold}")
        
        if not passed:
            all_passed = False
    
    print("-" * 60)
    if all_passed:
        print("🎉 ALL REQUIREMENTS MET!")
        print("✅ Model is ready for production deployment")
    else:
        print("⚠️ SOME REQUIREMENTS NOT MET")
        print("💡 Consider additional training or optimization")
    
    print("="*60)


def save_results_detailed(results: Dict[str, Any], output_path: str, model_info: Dict[str, Any]):
    """Save detailed evaluation results with metadata."""
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Prepare comprehensive results
    detailed_results = {
        'timestamp': str(torch.utils.data.get_worker_info().id if torch.utils.data.get_worker_info() else 'main'),
        'model_info': model_info,
        'evaluation_results': results,
        'summary': {
            'overall_performance': results.get('overall_score', 0),
            'model_size_mb': results.get('model_size_mb', 0),
            'inference_speed': results.get('inference_speed_tokens_per_second', 0),
            'total_parameters': results.get('total_parameters', 0),
        }
    }
    
    # Add task breakdown
    task_results = {}
    for key, value in results.items():
        if key.endswith('_score') and key != 'overall_score':
            task_name = key.replace('_score', '')
            task_results[task_name] = value
    
    if task_results:
        detailed_results['task_breakdown'] = task_results
    
    # Save to JSON
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(detailed_results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"📄 Detailed results saved to: {output_path}")


def main():
    """Main evaluation function."""
    
    parser = argparse.ArgumentParser(description="Evaluate Nepali Student Model")
    
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to model checkpoint or saved model directory")
    parser.add_argument("--config", type=str, default="config/eval_config.yaml",
                       help="Path to evaluation configuration")
    parser.add_argument("--output", type=str, default="outputs/metrics/evaluation_results.json",
                       help="Output path for results")
    parser.add_argument("--device", type=str, default="auto",
                       choices=["auto", "mps", "cuda", "cpu"],
                       help="Device to use for evaluation")
    parser.add_argument("--batch-size", type=int, default=16,
                       help="Batch size for evaluation")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose output")
    parser.add_argument("--save-detailed", action="store_true",
                       help="Save detailed results with metadata")
    parser.add_argument("--quick", action="store_true",
                       help="Run quick evaluation with fewer samples")
    
    args = parser.parse_args()
    
    # Setup logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Setup device
    device = setup_device(args.device)
    logger.info(f"🔧 Using device: {device}")
    
    # Load model
    logger.info(f"📂 Loading model from: {args.checkpoint}")
    
    try:
        result = load_model_from_checkpoint(args.checkpoint, device)
        if result is None:
            sys.exit(1)
        
        model, training_info = result
        
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        sys.exit(1)
    
    # Print model information
    print_model_info(model, training_info)
    
    # Load evaluation config
    eval_config = load_eval_config(args.config)
    
    # Modify config for quick evaluation
    if args.quick:
        eval_config['evaluation'] = eval_config.get('evaluation', {})
        eval_config['evaluation']['max_samples'] = 100
        logger
