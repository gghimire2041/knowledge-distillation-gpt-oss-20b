#!/usr/bin/env python3
"""
Deployment script for Nepali student model.
Supports API server deployment and mobile export.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import json

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from deployment.api_server import run_server
from deployment.mobile_export import MobileExporter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def validate_model_path(model_path: str) -> bool:
    """Validate that model path contains required files."""
    
    model_path = Path(model_path)
    required_files = ["config.json", "pytorch_model.bin"]
    
    if not model_path.exists():
        logger.error(f"❌ Model directory not found: {model_path}")
        return False
    
    missing_files = []
    for file_name in required_files:
        if not (model_path / file_name).exists():
            missing_files.append(file_name)
    
    if missing_files:
        logger.error(f"❌ Missing required files in {model_path}: {missing_files}")
        return False
    
    # Check tokenizer files
    tokenizer_files = ["tokenizer.model", "tokenizer_config.json"]
    tokenizer_found = any((model_path / f).exists() for f in tokenizer_files)
    
    if not tokenizer_found:
        logger.warning(f"⚠️ No tokenizer files found in {model_path}")
        logger.warning("   Model may not work without tokenizer")
    
    logger.info(f"✅ Model validation passed: {model_path}")
    return True


def deploy_api_server(args):
    """Deploy API server for model inference."""
    
    logger.info("🚀 Starting Nepali Model API Server")
    logger.info(f"📁 Model path: {args.model_path}")
    logger.info(f"🌐 Server will run on http://{args.host}:{args.port}")
    
    # Validate model
    if not validate_model_path(args.model_path):
        sys.exit(1)
    
    # Set environment variables for the API server
    os.environ['NEPALI_MODEL_PATH'] = args.model_path
    os.environ['NEPALI_API_HOST'] = args.host
    os.environ['NEPALI_API_PORT'] = str(args.port)
    
    try:
        # Start the server
        run_server(host=args.host, port=args.port)
    except KeyboardInterrupt:
        logger.info("⚠️ Server stopped by user")
    except Exception as e:
        logger.error(f"❌ Server error: {e}")
        sys.exit(1)


def export_mobile(args):
    """Export model for mobile deployment."""
    
    logger.info("📱 Starting mobile export process")
    logger.info(f"📁 Model path: {args.model_path}")
    
    # Validate model
    if not validate_model_path(args.model_path):
        sys.exit(1)
    
    try:
        # Initialize exporter
        exporter = MobileExporter(args.model_path)
        
        # Export based on requested formats
        exports = {}
        
        if 'all' in args.formats or 'onnx' in args.formats:
            logger.info("📦 Exporting to ONNX format...")
            onnx_path = exporter.export_onnx()
            exports['onnx'] = onnx_path
            logger.info(f"✅ ONNX export complete: {onnx_path}")
        
        if 'all' in args.formats or 'torchscript' in args.formats:
            logger.info("📦 Exporting to TorchScript format...")
            ts_path = exporter.export_torchscript()
            exports['torchscript'] = ts_path
            logger.info(f"✅ TorchScript export complete: {ts_path}")
        
        if 'all' in args.formats or 'coreml' in args.formats:
            logger.info("📦 Exporting to Core ML format...")
            try:
                coreml_path = exporter.export_coreml()
                exports['coreml'] = coreml_path
                logger.info(f"✅ Core ML export complete: {coreml_path}")
            except ImportError:
                logger.warning("⚠️ Core ML export skipped (coremltools not installed)")
                logger.warning("   Install with: pip install coremltools")
        
        # Create deployment package if multiple formats exported
        if len(exports) > 1 or args.package:
            logger.info("📦 Creating deployment package...")
            package_path = exporter._create_deployment_package(exports)
            logger.info(f"✅ Deployment package created: {package_path}")
            exports['package'] = package_path
        
        # Print summary
        print("\n" + "="*50)
        print("📱 MOBILE EXPORT SUMMARY")
        print("="*50)
        
        for format_name, file_path in exports.items():
            file_size = Path(file_path).stat().st_size / (1024*1024) if Path(file_path).exists() else 0
            print(f"{format_name.upper():.<20} {file_path}")
            print(f"{'Size':.<20} {file_size:.1f} MB")
            print("-" * 50)
        
        print(f"\n🎉 Mobile export completed successfully!")
        print(f"📁 All exports saved to: outputs/exports/")
        
        # Usage instructions
        print(f"\n📖 USAGE INSTRUCTIONS:")
        print(f"  • ONNX: Use with ONNX Runtime on any platform")
        print(f"  • TorchScript: Use with PyTorch Mobile (Android/iOS)")
        print(f"  • Core ML: Use with Core ML framework (iOS/macOS)")
        
        if 'package' in exports:
            print(f"  • Package: Complete deployment bundle with docs")
        
    except Exception as e:
        logger.error(f"❌ Mobile export failed: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


def benchmark_model(args):
    """Benchmark model performance."""
    
    logger.info("⚡ Starting model benchmark")
    logger.info(f"📁 Model path: {args.model_path}")
    
    # Validate model
    if not validate_model_path(args.model_path):
        sys.exit(1)
    
    try:
        # Import benchmark utilities
        from evaluation.benchmarks import NepaliEvaluator
        import torch
        import time
        
        # Load model
        sys.path.append(str(Path(__file__).parent.parent / "src"))
        from model.student_architecture import NepaliStudentModel, NepaliStudentConfig
        
        config_file = Path(args.model_path) / "config.json"
        model_file = Path(args.model_path) / "pytorch_model.bin"
        
        with open(config_file, 'r') as f:
            model_config_dict = json.load(f)
        
        model_config = NepaliStudentConfig(**model_config_dict)
        model = NepaliStudentModel(model_config)
        model.load_state_dict(torch.load(model_file, map_location='cpu'))
        model.eval()
        
        logger.info("✅ Model loaded successfully")
        
        # Benchmark inference speed
        logger.info("🔥 Benchmarking inference speed...")
        
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        model.to(device)
        
        # Create dummy input
        batch_size = args.batch_size
        seq_length = args.sequence_length
        vocab_size = model.config.vocab_size
        
        dummy_input = torch.randint(0, vocab_size, (batch_size, seq_length)).to(device)
        
        # Warmup
        logger.info("🔥 Warming up...")
        with torch.no_grad():
            for _ in range(10):
                _ = model(dummy_input)
        
        # Actual benchmark
        logger.info(f"📊 Running benchmark ({args.num_runs} runs)...")
        times = []
        
        with torch.no_grad():
            for i in range(args.num_runs):
                start_time = time.time()
                outputs = model(dummy_input)
                torch.mps.synchronize() if device.type == 'mps' else None
                end_time = time.time()
                
                times.append(end_time - start_time)
                
                if (i + 1) % 10 == 0:
                    logger.info(f"  Completed {i + 1}/{args.num_runs} runs...")
        
        # Calculate statistics
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        
        total_tokens = batch_size * seq_length
        tokens_per_second = total_tokens / avg_time
        
        # Print results
        print("\n" + "="*50)
        print("⚡ PERFORMANCE BENCHMARK RESULTS")
        print("="*50)
        print(f"Model: {Path(args.model_path).name}")
        print(f"Device: {device}")
        print(f"Parameters: {model.count_parameters():,}")
        print("-" * 50)
        print(f"Batch Size: {batch_size}")
        print(f"Sequence Length: {seq_length}")
        print(f"Total Tokens per Run: {total_tokens:,}")
        print(f"Number of Runs: {args.num_runs}")
        print("-" * 50)
        print(f"Average Time: {avg_time*1000:.2f} ms")
        print(f"Min Time: {min_time*1000:.2f} ms")
        print(f"Max Time: {max_time*1000:.2f} ms")
        print(f"Tokens/Second: {tokens_per_second:.0f}")
        print(f"Throughput: {tokens_per_second/1000:.1f}K tokens/sec")
        print("="*50)
        
        # Memory usage
        if device.type == 'mps':
            import psutil
            memory_info = psutil.virtual_memory()
            print(f"Memory Used: {memory_info.used / (1024**3):.1f} GB")
        elif device.type == 'cuda':
            memory_used = torch.cuda.memory_allocated(device) / (1024**3)
            print(f"GPU Memory Used: {memory_used:.1f} GB")
        
        logger.info("✅ Benchmark completed successfully")
        
    except Exception as e:
        logger.error(f"❌ Benchmark failed: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


def main():
    """Main deployment function."""
    
    parser = argparse.ArgumentParser(description="Deploy Nepali Student Model")
    
    # Common arguments
    parser.add_argument("--model-path", type=str, default="models/student/best_model",
                       help="Path to trained model directory")
    parser.add_argument("--debug", action="store_true", 
                       help="Enable debug mode")
    
    # Subcommands
    subparsers = parser.add_subparsers(dest="mode", help="Deployment mode")
    
    # API Server subcommand
    api_parser = subparsers.add_parser("api", help="Deploy API server")
    api_parser.add_argument("--host", type=str, default="127.0.0.1",
                           help="API server host")
    api_parser.add_argument("--port", type=int, default=8000,
                           help="API server port")
    api_parser.add_argument("--workers", type=int, default=1,
                           help="Number of worker processes")
    api_parser.add_argument("--reload", action="store_true",
                           help="Enable auto-reload for development")
    
    # Mobile Export subcommand
    mobile_parser = subparsers.add_parser("mobile", help="Export for mobile")
    mobile_parser.add_argument("--formats", nargs="+", 
                              choices=["onnx", "torchscript", "coreml", "all"],
                              default=["all"],
                              help="Export formats")
    mobile_parser.add_argument("--package", action="store_true",
                              help="Create deployment package")
    mobile_parser.add_argument("--optimize", action="store_true",
                              help="Apply additional optimizations")
    
    # Benchmark subcommand
    benchmark_parser = subparsers.add_parser("benchmark", help="Benchmark model performance")
    benchmark_parser.add_argument("--batch-size", type=int, default=1,
                                 help="Batch size for benchmarking")
    benchmark_parser.add_argument("--sequence-length", type=int, default=512,
                                 help="Sequence length for benchmarking") 
    benchmark_parser.add_argument("--num-runs", type=int, default=100,
                                 help="Number of benchmark runs")
    
    args = parser.parse_args()
    
    # Check if mode is specified
    if not args.mode:
        parser.print_help()
        sys.exit(1)
    
    # Set debug logging
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Execute based on mode
    try:
        if args.mode == "api":
            deploy_api_server(args)
        elif args.mode == "mobile":
            export_mobile(args)
        elif args.mode == "benchmark":
            benchmark_model(args)
        else:
            logger.error(f"❌ Unknown mode: {args.mode}")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("⚠️ Deployment interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"❌ Deployment failed: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
