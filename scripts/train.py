#!/usr/bin/env python3
"""
Main training script for Nepali knowledge distillation.
Supports staged training and resuming from checkpoints.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import yaml
from transformers import set_seed
from datasets import Dataset, DatasetDict, load_from_disk

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data.collector import NepaliDataCollector
from data.preprocessor import NepaliPreprocessor  
from data.teacher_inference import TeacherInferenceEngine
from model.student_architecture import NepaliStudentModel, NepaliStudentConfig
from model.tokenizer import NepaliTokenizer
from training.trainer import DistillationTrainer, TrainingValidator
from evaluation.benchmarks import NepaliEvaluator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)


def load_configs(config_paths: Dict[str, str]) -> Dict[str, Any]:
    """Load and merge configuration files."""
    
    merged_config = {}
    
    for config_name, config_path in config_paths.items():
        if Path(config_path).exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                merged_config[config_name] = config
            logger.info(f"✅ Loaded {config_name} from {config_path}")
        else:
            logger.error(f"❌ Config file not found: {config_path}")
            sys.exit(1)
    
    return merged_config


def setup_device(device_name: str = "auto") -> torch.device:
    """Setup optimal device for training."""
    
    if device_name == "auto":
        if torch.backends.mps.is_available():
            device = torch.device("mps")
            logger.info("✅ Using Metal Performance Shaders (MPS)")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"✅ Using CUDA: {torch.cuda.get_device_name()}")
        else:
            device = torch.device("cpu")
            logger.info("⚠️ Using CPU (training will be slow)")
    else:
        device = torch.device(device_name)
        logger.info(f"🔧 Using specified device: {device}")
    
    # Validate device setup
    is_valid, errors = TrainingValidator.validate_device_setup(device)
    if not is_valid:
        for error in errors:
            logger.error(f"❌ Device validation error: {error}")
        sys.exit(1)
    
    return device


def stage_data_preparation(configs: Dict[str, Any], force_rebuild: bool = False) -> DatasetDict:
    """Stage 1: Data collection and preprocessing."""
    
    logger.info("🗂️ Starting data preparation stage...")
    
    # Check if processed data already exists
    dataset_path = Path("data/datasets/nepali_distillation")
    if dataset_path.exists() and not force_rebuild:
        logger.info("📁 Loading existing processed dataset...")
        try:
            dataset = load_from_disk(str(dataset_path))
            logger.info(f"✅ Loaded existing dataset: {len(dataset['train'])} train, {len(dataset.get('validation', []))} validation samples")
            return dataset
        except Exception as e:
            logger.warning(f"⚠️ Failed to load existing dataset: {e}")
            logger.info("🔄 Rebuilding dataset...")
    
    data_config = configs.get('training', {}).get('data', {})
    
    # Step 1: Collect raw data
    logger.info("📥 Collecting raw Nepali data...")
    collector = NepaliDataCollector(data_config)
    raw_data = collector.collect_all_sources()
    
    if not raw_data or sum(len(texts) for texts in raw_data.values()) == 0:
        logger.error("❌ No data collected. Please check your data sources.")
        sys.exit(1)
    
    # Step 2: Preprocess and filter data
    logger.info("🧹 Preprocessing and filtering data...")
    preprocessor = NepaliPreprocessor(data_config)
    processed_dataset = preprocessor.process(raw_data)
    
    # Step 3: Train tokenizer if not exists
    tokenizer_path = Path("models/tokenizer")
    if not tokenizer_path.exists() or force_rebuild:
        logger.info("🔤 Training Nepali tokenizer...")
        tokenizer = NepaliTokenizer(vocab_size=configs.get('model', {}).get('tokenizer', {}).get('vocab_size', 32000))
        
        # Get all texts for tokenizer training
        all_texts = []
        for split_dataset in processed_dataset.values():
            all_texts.extend(split_dataset['text'])
        
        tokenizer.train_tokenizer(all_texts, str(tokenizer_path))
        logger.info(f"✅ Tokenizer trained and saved to {tokenizer_path}")
    else:
        logger.info(f"📁 Using existing tokenizer from {tokenizer_path}")
    
    # Step 4: Generate teacher outputs (if teacher model is available)
    teacher_config = configs.get('model', {}).get('teacher_model', {})
    if teacher_config and teacher_config.get('backend'):
        logger.info("🎓 Generating teacher outputs...")
        teacher_engine = TeacherInferenceEngine(teacher_config)
        
        try:
            dataset_with_teacher = teacher_engine.generate_training_data(processed_dataset)
            processed_dataset = dataset_with_teacher
            logger.info("✅ Teacher outputs generated successfully")
        except Exception as e:
            logger.warning(f"⚠️ Failed to generate teacher outputs: {e}")
            logger.info("📝 Proceeding with self-supervised training")
    
    # Step 5: Save processed dataset
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    processed_dataset.save_to_disk(str(dataset_path))
    
    logger.info(f"✅ Data preparation complete! Dataset saved to {dataset_path}")
    logger.info(f"📊 Final dataset: {len(processed_dataset['train'])} train, {len(processed_dataset.get('test', []))} validation samples")
    
    return processed_dataset


def stage_model_training(configs: Dict[str, Any], dataset: DatasetDict, device: torch.device, resume_checkpoint: Optional[str] = None):
    """Stage 2: Knowledge distillation training."""
    
    logger.info("🎓 Starting model training stage...")
    
    # Validate training config
    is_valid, errors = TrainingValidator.validate_config(configs.get('training', {}))
    if not is_valid:
        for error in errors:
            logger.error(f"❌ Config validation error: {error}")
        sys.exit(1)
    
    # Load tokenizer
    tokenizer_path = Path("models/tokenizer")
    if not tokenizer_path.exists():
        logger.error("❌ Tokenizer not found. Please run data preparation first.")
        sys.exit(1)
    
    tokenizer = NepaliTokenizer.from_pretrained(str(tokenizer_path))
    logger.info("✅ Tokenizer loaded successfully")
    
    # Initialize student model
    model_config = NepaliStudentConfig(**configs.get('model', {}).get('student_model', {}))
    model = NepaliStudentModel(model_config)
    
    logger.info(f"🧠 Student model initialized: {model.count_parameters():,} parameters")
    logger.info(f"💾 Model memory footprint: {model.get_memory_footprint()}")
    
    # Prepare datasets
    train_dataset = dataset['train']
    eval_dataset = dataset.get('test', dataset.get('validation', train_dataset.select(range(min(1000, len(train_dataset))))))
    
    # Initialize trainer
    trainer = DistillationTrainer(
        model=model,
        config=configs,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        device=device
    )
    
    # Resume from checkpoint if specified
    if resume_checkpoint:
        trainer.resume_from_checkpoint(resume_checkpoint)
    
    # Start training
    trainer.train()
    
    logger.info("✅ Model training completed!")
    return trainer


def stage_evaluation(configs: Dict[str, Any], model_path: str) -> Dict[str, Any]:
    """Stage 3: Comprehensive model evaluation."""
    
    logger.info("📊 Starting evaluation stage...")
    
    # Load model for evaluation
    if not Path(model_path).exists():
        logger.error(f"❌ Model not found: {model_path}")
        sys.exit(1)
    
    # Load model
    config_file = Path(model_path) / "config.json"
    model_file = Path(model_path) / "pytorch_model.bin"
    
    if not (config_file.exists() and model_file.exists()):
        logger.error(f"❌ Model files not found in {model_path}")
        sys.exit(1)
    
    # Load model configuration
    import json
    with open(config_file, 'r') as f:
        model_config_dict = json.load(f)
    
    model_config = NepaliStudentConfig(**model_config_dict)
    model = NepaliStudentModel(model_config)
    model.load_state_dict(torch.load(model_file, map_location='cpu'))
    
    logger.info("✅ Model loaded for evaluation")
    
    # Initialize evaluator
    eval_config = configs.get('evaluation', {})
    evaluator = NepaliEvaluator(eval_config)
    
    # Run evaluation
    results = evaluator.evaluate_model(model)
    
    # Save results
    results_dir = Path("outputs/metrics")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    results_file = results_dir / "evaluation_results.json"
    evaluator.save_results(results, str(results_file))
    
    logger.info("✅ Evaluation completed!")
    
    return results


def print_results_summary(results: Dict[str, Any]):
    """Print a formatted summary of evaluation results."""
    
    print("\n" + "="*60)
    print("📊 FINAL EVALUATION RESULTS")
    print("="*60)
    
    # Key metrics
    key_metrics = [
        ('overall_score', 'Overall Performance'),
        ('model_size_mb', 'Model Size (MB)'),
        ('total_parameters', 'Total Parameters'),
        ('inference_speed_tokens_per_second', 'Inference Speed (tokens/sec)')
    ]
    
    for metric_key, metric_name in key_metrics:
        if metric_key in results:
            value = results[metric_key]
            if isinstance(value, float):
                if metric_key == 'total_parameters':
                    print(f"{metric_name:.<40} {value:,.0f}")
                else:
                    print(f"{metric_name:.<40} {value:.4f}")
            else:
                print(f"{metric_name:.<40} {value}")
    
    print("-" * 60)
    
    # Task-specific scores
    task_metrics = [(k, v) for k, v in results.items() if k.endswith('_score') and k != 'overall_score']
    
    if task_metrics:
        print("Task-specific Performance:")
        for metric_key, value in task_metrics:
            task_name = metric_key.replace('_score', '').replace('_', ' ').title()
            print(f"  {task_name:.<35} {value:.4f}")
    
    print("="*60)
    
    # Check requirements
    print("\n🎯 REQUIREMENT CHECKS:")
    
    checks = [
        ('Performance Retention', results.get('overall_score', 0), 0.8, '≥'),
        ('Model Size (MB)', results.get('model_size_mb', float('inf')), 800, '≤'),
        ('Speed Improvement', results.get('inference_speed_tokens_per_second', 0) / 50, 10, '≥')  # Assuming baseline 50 tok/s
    ]
    
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
    
    print("\n" + ("🎉 ALL REQUIREMENTS MET!" if all_passed else "⚠️ SOME REQUIREMENTS NOT MET"))
    print("="*60)


def main():
    """Main training pipeline."""
    
    parser = argparse.ArgumentParser(description="Nepali Knowledge Distillation Training Pipeline")
    
    # Main arguments
    parser.add_argument("--stage", type=str, 
                       choices=["data-prep", "distillation", "evaluation", "all"],
                       default="all",
                       help="Training stage to run")
    
    # Configuration files
    parser.add_argument("--config", type=str, 
                       default="config/training_config.yaml",
                       help="Path to main training configuration")
    parser.add_argument("--model-config", type=str,
                       default="config/model_config.yaml", 
                       help="Path to model configuration")
    parser.add_argument("--eval-config", type=str,
                       default="config/eval_config.yaml",
                       help="Path to evaluation configuration")
    
    # Training options
    parser.add_argument("--resume", type=str, 
                       help="Resume from checkpoint path")
    parser.add_argument("--device", type=str, default="auto",
                       choices=["auto", "mps", "cuda", "cpu"],
                       help="Device to use for training")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    
    # Data options
    parser.add_argument("--force-rebuild", action="store_true",
                       help="Force rebuild of processed data")
    parser.add_argument("--data-dir", type=str, default="data",
                       help="Data directory path")
    
    # Output options
    parser.add_argument("--output-dir", type=str, default="models/student",
                       help="Output directory for model and checkpoints")
    parser.add_argument("--run-name", type=str, 
                       help="Custom run name for experiment tracking")
    
    # Debugging options
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug mode with verbose logging")
    parser.add_argument("--dry-run", action="store_true",
                       help="Dry run mode - validate setup without training")
    
    args = parser.parse_args()
    
    # Setup logging level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.info("🐛 Debug mode enabled")
    
    # Set random seed for reproducibility
    set_seed(args.seed)
    logger.info(f"🌱 Random seed set to: {args.seed}")
    
    # Load configurations
    config_paths = {
        'training': args.config,
        'model': args.model_config,
        'evaluation': args.eval_config
    }
    
    try:
        configs = load_configs(config_paths)
    except Exception as e:
        logger.error(f"❌ Failed to load configurations: {e}")
        sys.exit(1)
    
    # Override output directory if specified
    if args.output_dir != "models/student":
        configs['training']['checkpointing']['output_dir'] = args.output_dir
    
    # Setup device
    device = setup_device(args.device)
    
    # Add device to hardware config
    if 'hardware' not in configs['model']:
        configs['model']['hardware'] = {}
    configs['model']['hardware']['device'] = str(device)
    
    # Set run name for experiment tracking
    if args.run_name:
        if 'training' not in configs:
            configs['training'] = {}
        configs['training']['run_name'] = args.run_name
    
    # Validate configurations
    logger.info("🔍 Validating configuration...")
    
    # Print configuration summary
    print("\n" + "="*50)
    print("⚙️ TRAINING CONFIGURATION SUMMARY")
    print("="*50)
    print(f"Stage: {args.stage}")
    print(f"Device: {device}")
    print(f"Seed: {args.seed}")
    print(f"Output Dir: {configs['training']['checkpointing']['output_dir']}")
    
    if 'student_model' in configs['model']:
        student_config = configs['model']['student_model']
        print(f"Model Size: ~{student_config.get('hidden_size', 768) * student_config.get('num_hidden_layers', 12) / 1000:.0f}M parameters")
        print(f"Vocab Size: {student_config.get('vocab_size', 32000)}")
    
    if 'training' in configs:
        training_config = configs['training']
        print(f"Epochs: {training_config.get('num_epochs', 3)}")
        print(f"Batch Size: {training_config['data'].get('batch_size', 8)}")
        print(f"Learning Rate: {training_config.get('learning_rate', 5e-4)}")
    
    print("="*50 + "\n")
    
    # Dry run mode - just validate setup
    if args.dry_run:
        logger.info("🧪 Dry run mode - validating setup...")
        
        # Test device functionality
        try:
            test_tensor = torch.randn(2, 2).to(device)
            result = test_tensor @ test_tensor
            logger.info("✅ Device functionality test passed")
        except Exception as e:
            logger.error(f"❌ Device test failed: {e}")
            sys.exit(1)
        
        logger.info("✅ Dry run completed successfully - ready for training!")
        return
    
    # Execute training pipeline
    dataset = None
    model_path = None
    results = None
    
    try:
        # Stage 1: Data Preparation
        if args.stage in ["data-prep", "all"]:
            dataset = stage_data_preparation(configs, force_rebuild=args.force_rebuild)
        
        # Stage 2: Model Training
        if args.stage in ["distillation", "all"]:
            if dataset is None:
                # Load existing dataset
                dataset_path = Path("data/datasets/nepali_distillation")
                if not dataset_path.exists():
                    logger.error("❌ No processed dataset found. Run data preparation first.")
                    sys.exit(1)
                dataset = load_from_disk(str(dataset_path))
            
            trainer = stage_model_training(configs, dataset, device, args.resume)
            model_path = str(trainer.output_dir / "best_model")
        
        # Stage 3: Evaluation
        if args.stage in ["evaluation", "all"]:
            if model_path is None:
                # Use default model path
                model_path = str(Path(configs['training']['checkpointing']['output_dir']) / "best_model")
            
            if not Path(model_path).exists():
                logger.error(f"❌ Model not found for evaluation: {model_path}")
                logger.info("💡 Train a model first or specify correct model path")
                sys.exit(1)
            
            results = stage_evaluation(configs, model_path)
    
    except KeyboardInterrupt:
        logger.info("⚠️ Training interrupted by user")
        sys.exit(130)
    
    except Exception as e:
        logger.error(f"❌ Training failed with error: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)
    
    # Print final results
    if results:
        print_results_summary(results)
    
    # Final success message
    print("\n🎉 Training pipeline completed successfully!")
    
    if model_path:
        print(f"📁 Trained model available at: {model_path}")
    
    if results:
        print(f"📊 Evaluation results saved to: outputs/metrics/evaluation_results.json")
    
    print("\n🚀 Next steps:")
    print("  • Deploy API server: python scripts/deploy.py --mode api")
    print("  • Export for mobile: python scripts/deploy.py --mode mobile")
    print("  • Run additional evaluation: python scripts/evaluate.py --checkpoint [model_path]")


if __name__ == "__main__":
    main()
