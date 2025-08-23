import os
import torch
import logging
from typing import Dict, Any, Optional
from pathlib import Path

from transformers import TrainingArguments
from trl import SFTTrainer
from datasets import Dataset

from .callbacks import EnhancedTrainingCallback
from ..config.hardware_config import HardwareOptimizer

logger = logging.getLogger(__name__)


class EnhancedTrainer:
    """Enhanced trainer with comprehensive monitoring and optimization"""
    
    def __init__(self, model, tokenizer, dataset: Dataset, 
                 training_config: Dict[str, Any], hardware_config: Dict[str, Any],
                 output_dir: str, max_seq_length: int = 512):
        
        self.model = model
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.training_config = training_config
        self.hardware_config = hardware_config
        self.output_dir = Path(output_dir)
        self.max_seq_length = max_seq_length
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize hardware optimizer
        self.hw_optimizer = HardwareOptimizer()
        
        # Setup training arguments
        self.training_args = self._create_training_arguments()
        
        # Setup callbacks
        self.callbacks = self._setup_callbacks()
        
        # Initialize trainer
        self.trainer = self._create_trainer()
        
    def _create_training_arguments(self) -> TrainingArguments:
        """Create optimized training arguments"""
        logger.info("Creating training arguments...")
        
        # Get hardware-optimized config
        base_config = {
            "output_dir": str(self.output_dir),
            "num_train_epochs": self.training_config.get('num_epochs', 3),
            "learning_rate": self.training_config.get('learning_rate', 2e-4),
            "warmup_ratio": self.training_config.get('warmup_ratio', 0.03),
            "save_steps": self.training_config.get('save_steps', 25),
            "logging_steps": self.training_config.get('logging_steps', 5),
            "weight_decay": self.training_config.get('weight_decay', 0.001),
            "max_grad_norm": self.training_config.get('max_grad_norm', 0.3),
            "lr_scheduler_type": self.training_config.get('lr_scheduler_type', 'cosine'),
            "save_total_limit": self.training_config.get('save_total_limit', 3),
            "remove_unused_columns": self.training_config.get('remove_unused_columns', True),
            "gradient_checkpointing": self.training_config.get('gradient_checkpointing', True),
            "max_steps": -1,
            "group_by_length": True,
            "report_to": "none",  # Disable wandb/tensorboard by default
            "logging_dir": str(self.output_dir / "logs"),
            "save_strategy": "steps",
            "evaluation_strategy": "no",  # Can be enabled if validation set is available
        }
        
        # Apply hardware optimizations
        optimized_config = self.hw_optimizer.get_hardware_optimized_training_args(base_config)
        
        # Log configuration
        self._log_training_config(optimized_config)
        
        return TrainingArguments(**optimized_config)
    
    def _log_training_config(self, config: Dict[str, Any]):
        """Log training configuration"""
        logger.info("Training Configuration:")
        logger.info(f"  Epochs: {config.get('num_train_epochs')}")
        logger.info(f"  Learning Rate: {config.get('learning_rate')}")
        logger.info(f"  Batch Size: {config.get('per_device_train_batch_size')}")
        logger.info(f"  Gradient Accumulation: {config.get('gradient_accumulation_steps')}")
        logger.info(f"  Effective Batch Size: {config.get('per_device_train_batch_size', 1) * config.get('gradient_accumulation_steps', 1)}")
        logger.info(f"  Optimizer: {config.get('optim')}")
        logger.info(f"  Mixed Precision: FP16={config.get('fp16')}, BF16={config.get('bf16')}")
        logger.info(f"  Gradient Checkpointing: {config.get('gradient_checkpointing')}")
        logger.info(f"  Save Steps: {config.get('save_steps')}")
        logger.info(f"  Logging Steps: {config.get('logging_steps')}")
    
    def _setup_callbacks(self) -> list:
        """Setup training callbacks"""
        callbacks = []
        
        # Add enhanced callback for monitoring
        enhanced_callback = EnhancedTrainingCallback(
            log_dir=str(self.output_dir / "logs"),
            save_metrics=True
        )
        callbacks.append(enhanced_callback)
        
        return callbacks
    
    def _create_trainer(self) -> SFTTrainer:
        """Create the SFT trainer"""
        logger.info("Initializing SFT Trainer...")
        
        trainer = SFTTrainer(
            model=self.model,
            train_dataset=self.dataset,
            dataset_text_field="text",
            max_seq_length=self.max_seq_length,
            tokenizer=self.tokenizer,
            args=self.training_args,
            packing=False,  # Disable packing for better control
            callbacks=self.callbacks
        )
        
        return trainer
    
    def train(self) -> None:
        """Execute the training process"""
        try:
            logger.info("🚀 Starting training process...")
            
            # Log training start info
            self._log_training_start()
            
            # Start training
            train_result = self.trainer.train()
            
            # Log training completion
            self._log_training_completion(train_result)
            
            # Save the final model
            self._save_final_model()
            
            logger.info("✅ Training completed successfully!")
            
        except Exception as e:
            logger.error(f"❌ Training failed: {e}")
            raise
    
    def _log_training_start(self):
        """Log training start information"""
        logger.info("=" * 80)
        logger.info("🎯 TRAINING INFORMATION")
        logger.info("=" * 80)
        logger.info(f"Dataset size: {len(self.dataset)} examples")
        logger.info(f"Number of epochs: {self.training_args.num_train_epochs}")
        logger.info(f"Steps per epoch: {len(self.dataset) // (self.training_args.per_device_train_batch_size * self.training_args.gradient_accumulation_steps)}")
        logger.info(f"Total training steps: {self.trainer.get_train_dataloader().__len__()}")
        logger.info(f"Output directory: {self.output_dir}")
        
        # Log model info
        if hasattr(self.model, 'print_trainable_parameters'):
            logger.info("Model parameters:")
            self.model.print_trainable_parameters()
        
        logger.info("=" * 80)
    
    def _log_training_completion(self, train_result):
        """Log training completion information"""
        logger.info("=" * 80)
        logger.info("🎉 TRAINING COMPLETED")
        logger.info("=" * 80)
        
        if train_result and hasattr(train_result, 'training_loss'):
            logger.info(f"Final training loss: {train_result.training_loss:.4f}")
        
        if train_result and hasattr(train_result, 'metrics'):
            logger.info("Final metrics:")
            for key, value in train_result.metrics.items():
                logger.info(f"  {key}: {value}")
        
        logger.info("=" * 80)
    
    def _save_final_model(self):
        """Save the final trained model"""
        final_model_path = self.output_dir / "final"
        
        logger.info(f"💾 Saving final model to: {final_model_path}")
        
        # Save the model adapter
        self.trainer.model.save_pretrained(final_model_path)
        
        # Save the tokenizer
        self.tokenizer.save_pretrained(final_model_path)
        
        # Save training configuration
        config_path = final_model_path / "training_config.json"
        import json
        with open(config_path, 'w') as f:
            json.dump({
                "training_config": self.training_config,
                "hardware_config": self.hardware_config,
                "training_args": self.training_args.to_dict(),
                "dataset_size": len(self.dataset),
                "max_seq_length": self.max_seq_length
            }, f, indent=2)
        
        logger.info("✅ Model saved successfully!")
    
    def resume_from_checkpoint(self, checkpoint_path: str):
        """Resume training from a checkpoint"""
        logger.info(f"📂 Resuming training from checkpoint: {checkpoint_path}")
        
        try:
            self.trainer.train(resume_from_checkpoint=checkpoint_path)
            logger.info("✅ Successfully resumed training from checkpoint")
        except Exception as e:
            logger.error(f"❌ Failed to resume from checkpoint: {e}")
            raise
    
    def get_training_state(self) -> Dict[str, Any]:
        """Get current training state information"""
        if not hasattr(self.trainer, 'state'):
            return {"status": "not_started"}
        
        state = self.trainer.state
        return {
            "epoch": state.epoch,
            "global_step": state.global_step,
            "max_steps": state.max_steps,
            "num_train_epochs": state.num_train_epochs,
            "log_history": state.log_history[-5:] if state.log_history else [],  # Last 5 logs
            "best_metric": getattr(state, 'best_metric', None),
            "is_local_process_zero": state.is_local_process_zero,
            "is_world_process_zero": state.is_world_process_zero
        }
    
    def save_checkpoint(self, checkpoint_name: Optional[str] = None):
        """Save a training checkpoint manually"""
        if checkpoint_name:
            checkpoint_path = self.output_dir / checkpoint_name
        else:
            checkpoint_path = self.output_dir / f"checkpoint-{self.trainer.state.global_step}"
        
        logger.info(f"💾 Saving checkpoint to: {checkpoint_path}")
        
        try:
            self.trainer.save_model(str(checkpoint_path))
            logger.info("✅ Checkpoint saved successfully!")
        except Exception as e:
            logger.error(f"❌ Failed to save checkpoint: {e}")
            raise
    
    def evaluate_during_training(self, eval_dataset: Optional[Dataset] = None):
        """Run evaluation during training (if evaluation dataset is available)"""
        if eval_dataset is not None:
            logger.info("📊 Running evaluation...")
            try:
                eval_results = self.trainer.evaluate(eval_dataset=eval_dataset)
                logger.info(f"Evaluation results: {eval_results}")
                return eval_results
            except Exception as e:
                logger.warning(f"⚠️ Evaluation failed: {e}")
                return None
        else:
            logger.info("ℹ️ No evaluation dataset provided, skipping evaluation")
            return None
