#!/usr/bin/env python3
"""
Enhanced Fine-tuning Script for Large Language Models

This script provides a comprehensive fine-tuning framework with:
- Hardware-aware optimization
- Robust model loading with fallbacks
- Enhanced dataset validation
- Comprehensive logging and monitoring
- Model evaluation capabilities
"""

import argparse
import yaml
import sys
import os
from pathlib import Path
from typing import Dict, Any

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from dotenv import load_dotenv

# Import our enhanced modules
from src.config.model_config import (
    ModelConfig, TrainingConfig, LoRAConfig, DataConfig, 
    HardwareConfig, EvaluationConfig, LoggingConfig
)
from src.config.hardware_config import HardwareOptimizer
from src.data.dataset_loader import DatasetLoader
from src.models.model_loader import ModelLoader
from src.models.lora_config import LoRAConfigManager
from src.training.trainer import EnhancedTrainer
from src.utils.logging import TrainingLogger
from src.utils.validation import ModelEvaluator


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Enhanced LLM Fine-tuning Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic fine-tuning with default config
  python finetune.py
  
  # Use custom config file
  python finetune.py --config custom_config.yaml
  
  # Override output directory
  python finetune.py --output-dir ./my-custom-model
  
  # Dry run to validate configuration
  python finetune.py --dry-run
  
  # Resume from checkpoint
  python finetune.py --resume-from ./policy-chatbot-v1/checkpoint-50
  
  # Disable evaluation
  python finetune.py --no-evaluation
        """
    )
    
    parser.add_argument(
        "--config", 
        default="config.yaml", 
        help="Path to configuration file (default: config.yaml)"
    )
    parser.add_argument(
        "--output-dir", 
        help="Override output directory from config"
    )
    parser.add_argument(
        "--dry-run", 
        action="store_true", 
        help="Validate configuration without training"
    )
    parser.add_argument(
        "--resume-from", 
        help="Resume training from checkpoint path"
    )
    parser.add_argument(
        "--no-evaluation", 
        action="store_true", 
        help="Skip model evaluation after training"
    )
    parser.add_argument(
        "--verbose", "-v", 
        action="store_true", 
        help="Enable verbose logging"
    )
    
    return parser.parse_args()


def load_configuration(config_path: str) -> Dict[str, Any]:
    """Load and validate configuration from YAML file"""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Validate required sections
        required_sections = ['model', 'training', 'lora', 'data']
        missing_sections = [section for section in required_sections if section not in config]
        
        if missing_sections:
            raise ValueError(f"Missing required configuration sections: {missing_sections}")
        
        return config
        
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML configuration: {e}")


def validate_environment():
    """Validate environment and dependencies"""
    # Load environment variables
    load_dotenv()
    
    # Check for HuggingFace token
    hf_token = os.getenv("HUGGINGFACE_TOKEN")
    if not hf_token:
        print("⚠️  Warning: HUGGINGFACE_TOKEN not found in environment.")
        print("   Some models may require authentication.")
        print("   Set it in .env file: HUGGINGFACE_TOKEN=your_token_here")
    
    return hf_token


def create_config_objects(config: Dict[str, Any], args) -> tuple:
    """Create configuration objects from loaded config"""
    
    # Model configuration
    model_config = config['model'].copy()
    if args.output_dir:
        model_config['output_dir'] = args.output_dir
    
    # Create config objects
    model_cfg = ModelConfig(**model_config)
    training_cfg = TrainingConfig(**config['training'])
    lora_cfg = LoRAConfig(**config['lora'])
    data_cfg = DataConfig(**config['data'])
    hardware_cfg = HardwareConfig(**config.get('hardware', {}))
    evaluation_cfg = EvaluationConfig(**config.get('evaluation', {}))
    logging_cfg = LoggingConfig(**config.get('logging', {}))
    
    # Override evaluation config if requested
    if args.no_evaluation:
        evaluation_cfg.run_evaluation = False
    
    # Set verbose logging if requested
    if args.verbose:
        logging_cfg.level = "DEBUG"
    
    return (model_cfg, training_cfg, lora_cfg, data_cfg, 
            hardware_cfg, evaluation_cfg, logging_cfg)


def main():
    """Main fine-tuning execution function"""
    try:
        # Parse arguments
        args = parse_arguments()
        
        # Load configuration
        print("📋 Loading configuration...")
        config = load_configuration(args.config)
        
        # Create configuration objects
        (model_cfg, training_cfg, lora_cfg, data_cfg, 
         hardware_cfg, evaluation_cfg, logging_cfg) = create_config_objects(config, args)
        
        # Validate environment
        validate_environment()
        
        # Initialize logging
        logger = TrainingLogger(
            log_dir=str(Path(model_cfg.output_dir) / "logs"),
            config=logging_cfg.__dict__
        )
        
        # Log configuration start
        logger.log_training_start({
            'model': model_cfg.__dict__,
            'training': training_cfg.__dict__,
            'lora': lora_cfg.__dict__,
            'data': data_cfg.__dict__,
            'hardware': hardware_cfg.__dict__,
            'evaluation': evaluation_cfg.__dict__
        })
        
        # Initialize hardware optimizer
        hw_optimizer = HardwareOptimizer()
        hw_optimizer.log_hardware_info()
        
        # Load and validate dataset
        print("📊 Loading and validating dataset...")
        dataset_loader = DatasetLoader(data_cfg.__dict__)
        dataset = dataset_loader.load_and_validate(data_cfg.dataset_path)
        
        # Log dataset information
        dataset_loader.log_dataset_info(dataset)
        dataset_stats = dataset_loader.get_dataset_statistics(dataset)
        logger.log_dataset_info(dataset_stats)
        
        # Check if this is a dry run
        if args.dry_run:
            print("✅ Dry run completed successfully!")
            print("   Configuration is valid and dataset loaded correctly.")
            print("   Run without --dry-run to start training.")
            return
        
        # Load model with fallback strategies
        print("🤖 Loading model...")
        model_loader = ModelLoader(model_cfg.__dict__, hardware_cfg.__dict__)
        model, tokenizer = model_loader.load_model_with_fallback()
        
        # Configure model for training
        model = model_loader.configure_model_for_training(model)
        
        # Get model information
        model_info = model_loader.get_model_info(model)
        logger.log_model_info(model_info)
        
        # Apply LoRA adaptation
        print("🔧 Applying LoRA adaptation...")
        lora_manager = LoRAConfigManager(lora_cfg.__dict__, model_cfg.__dict__)
        model = lora_manager.apply_lora_to_model(model)
        
        # Create enhanced trainer
        print("🎯 Initializing trainer...")
        trainer = EnhancedTrainer(
            model=model,
            tokenizer=tokenizer,
            dataset=dataset,
            training_config=training_cfg.__dict__,
            hardware_config=hardware_cfg.__dict__,
            output_dir=model_cfg.output_dir,
            max_seq_length=model_cfg.max_seq_length
        )
        
        # Resume from checkpoint if specified
        if args.resume_from:
            print(f"📂 Resuming from checkpoint: {args.resume_from}")
            trainer.resume_from_checkpoint(args.resume_from)
        else:
            # Start training
            print("🚀 Starting training...")
            trainer.train()
        
        # Log training completion
        logger.log_training_complete({
            "final_model_path": str(Path(model_cfg.output_dir) / "final"),
            "dataset_size": len(dataset),
            "training_completed": True
        })
        
        # Run evaluation if enabled
        if evaluation_cfg.run_evaluation:
            print("🔍 Running model evaluation...")
            try:
                # Load the trained model for evaluation
                from peft import PeftModel
                from transformers import AutoModelForCausalLM
                
                # Load base model and adapter
                final_model_path = Path(model_cfg.output_dir) / "final"
                
                base_model = AutoModelForCausalLM.from_pretrained(
                    model_cfg.base_model,
                    torch_dtype=model.dtype,
                    device_map="auto",
                    trust_remote_code=model_cfg.trust_remote_code
                )
                
                eval_model = PeftModel.from_pretrained(base_model, str(final_model_path))
                eval_model = eval_model.merge_and_unload()
                eval_model.eval()
                
                # Run evaluation
                evaluator = ModelEvaluator(
                    model=eval_model,
                    tokenizer=tokenizer,
                    test_questions=evaluation_cfg.test_questions,
                    template_format=data_cfg.template_format
                )
                
                evaluation_results = evaluator.evaluate_model(
                    save_results=True,
                    results_dir=str(Path(model_cfg.output_dir) / "evaluation")
                )
                
                print("✅ Evaluation completed successfully!")
                
            except Exception as e:
                print(f"⚠️ Evaluation failed: {e}")
                logger.log_warning("Model evaluation failed", {"error": str(e)})
        
        print("\n" + "="*80)
        print("🎉 FINE-TUNING COMPLETED SUCCESSFULLY!")
        print("="*80)
        print(f"📁 Model saved to: {model_cfg.output_dir}/final")
        print(f"📊 Logs saved to: {model_cfg.output_dir}/logs")
        if evaluation_cfg.run_evaluation:
            print(f"📈 Evaluation results: {model_cfg.output_dir}/evaluation")
        print("\nNext steps:")
        print("1. Test your model using chat.py")
        print("2. Review training logs and evaluation results")
        print("3. Consider further fine-tuning if needed")
        print("="*80)
        
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()