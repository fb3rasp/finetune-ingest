import json
import time
import torch
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
from transformers import TrainerCallback, TrainerState, TrainerControl
from transformers.trainer_utils import TrainOutput

logger = logging.getLogger(__name__)


class EnhancedTrainingCallback(TrainerCallback):
    """Enhanced training callback with comprehensive monitoring and logging"""
    
    def __init__(self, log_dir: str, save_metrics: bool = True):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.save_metrics = save_metrics
        
        # Training metrics storage
        self.training_metrics = []
        self.epoch_metrics = []
        self.start_time = None
        self.last_log_time = None
        self.best_loss = float('inf')
        
        # Performance tracking
        self.step_times = []
        self.gpu_memory_usage = []
        
        logger.info(f"Enhanced training callback initialized. Logs will be saved to: {self.log_dir}")
    
    def on_train_begin(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        """Called at the beginning of training"""
        self.start_time = time.time()
        self.last_log_time = self.start_time
        
        logger.info("🚀 Training started")
        logger.info(f"Total steps planned: {state.max_steps}")
        logger.info(f"Total epochs planned: {state.num_train_epochs}")
        
        # Log initial hardware state
        self._log_hardware_state("Training Start")
    
    def on_train_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        """Called at the end of training"""
        end_time = time.time()
        total_duration = end_time - self.start_time if self.start_time else 0
        
        logger.info("🎉 Training completed!")
        logger.info(f"Total training time: {self._format_duration(total_duration)}")
        logger.info(f"Final step: {state.global_step}")
        logger.info(f"Final epoch: {state.epoch:.2f}")
        
        # Log final hardware state
        self._log_hardware_state("Training End")
        
        # Save final metrics
        if self.save_metrics:
            self._save_training_summary(state, total_duration)
    
    def on_epoch_begin(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        """Called at the beginning of each epoch"""
        logger.info(f"📅 Starting epoch {int(state.epoch) + 1}/{state.num_train_epochs}")
        
        # Reset epoch-specific metrics
        self.epoch_start_time = time.time()
        self.epoch_step_times = []
    
    def on_epoch_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        """Called at the end of each epoch"""
        epoch_duration = time.time() - getattr(self, 'epoch_start_time', time.time())
        
        # Calculate epoch metrics
        epoch_metrics = {
            "epoch": int(state.epoch),
            "duration": epoch_duration,
            "global_step": state.global_step,
            "learning_rate": self._get_current_learning_rate(state),
            "timestamp": time.time()
        }
        
        # Add loss if available
        if state.log_history:
            recent_logs = [log for log in state.log_history if 'train_loss' in log]
            if recent_logs:
                epoch_metrics["train_loss"] = recent_logs[-1]['train_loss']
        
        self.epoch_metrics.append(epoch_metrics)
        
        logger.info(f"✅ Completed epoch {int(state.epoch)} in {self._format_duration(epoch_duration)}")
        if 'train_loss' in epoch_metrics:
            logger.info(f"   Loss: {epoch_metrics['train_loss']:.4f}")
    
    def on_step_begin(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        """Called at the beginning of each training step"""
        self.step_start_time = time.time()
    
    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        """Called at the end of each training step"""
        step_duration = time.time() - getattr(self, 'step_start_time', time.time())
        self.step_times.append(step_duration)
        
        # Keep only recent step times (last 100 steps)
        if len(self.step_times) > 100:
            self.step_times = self.step_times[-100:]
    
    def on_log(self, args, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        """Called when logging occurs"""
        if logs is None:
            return
        
        current_time = time.time()
        
        # Enhance logs with additional metrics
        enhanced_logs = dict(logs)
        enhanced_logs.update({
            "timestamp": current_time,
            "global_step": state.global_step,
            "epoch": state.epoch,
            "learning_rate": self._get_current_learning_rate(state)
        })
        
        # Add hardware metrics
        if torch.cuda.is_available():
            gpu_metrics = self._get_gpu_metrics()
            enhanced_logs.update(gpu_metrics)
        
        # Add performance metrics
        if self.step_times:
            avg_step_time = sum(self.step_times[-10:]) / len(self.step_times[-10:])  # Last 10 steps
            enhanced_logs["avg_step_time"] = avg_step_time
            
            # Calculate steps per second
            if avg_step_time > 0:
                enhanced_logs["steps_per_second"] = 1.0 / avg_step_time
        
        # Add time since last log
        if self.last_log_time:
            enhanced_logs["time_since_last_log"] = current_time - self.last_log_time
        
        self.last_log_time = current_time
        
        # Store metrics
        self.training_metrics.append(enhanced_logs)
        
        # Update best loss
        if 'train_loss' in logs:
            if logs['train_loss'] < self.best_loss:
                self.best_loss = logs['train_loss']
                enhanced_logs["is_best_loss"] = True
                logger.info(f"🎯 New best loss: {self.best_loss:.4f}")
        
        # Log detailed progress every N steps
        if state.global_step % (args.logging_steps * 5) == 0:
            self._log_detailed_progress(enhanced_logs, state)
    
    def on_save(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        """Called when a checkpoint is saved"""
        logger.info(f"💾 Checkpoint saved at step {state.global_step}")
        
        # Save current metrics when checkpoint is saved
        if self.save_metrics:
            self._save_current_metrics(state.global_step)
    
    def on_evaluate(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        """Called when evaluation occurs"""
        logger.info(f"📊 Evaluation at step {state.global_step}")
    
    def _get_current_learning_rate(self, state: TrainerState) -> float:
        """Extract current learning rate from logs"""
        if state.log_history:
            for log_entry in reversed(state.log_history):
                if 'learning_rate' in log_entry:
                    return log_entry['learning_rate']
        return 0.0
    
    def _get_gpu_metrics(self) -> Dict[str, float]:
        """Get current GPU metrics"""
        metrics = {}
        
        try:
            if torch.cuda.is_available():
                device = torch.cuda.current_device()
                
                # Memory metrics
                allocated = torch.cuda.memory_allocated(device)
                reserved = torch.cuda.memory_reserved(device)
                max_allocated = torch.cuda.max_memory_allocated(device)
                
                metrics.update({
                    "gpu_memory_allocated_mb": allocated / (1024**2),
                    "gpu_memory_reserved_mb": reserved / (1024**2),
                    "gpu_memory_max_allocated_mb": max_allocated / (1024**2)
                })
                
                # GPU utilization (if nvidia-ml-py is available)
                try:
                    import pynvml
                    pynvml.nvmlInit()
                    handle = pynvml.nvmlDeviceGetHandleByIndex(device)
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    metrics["gpu_utilization_percent"] = util.gpu
                except ImportError:
                    pass  # nvidia-ml-py not available
                except Exception:
                    pass  # Other errors in getting GPU utilization
                
        except Exception as e:
            logger.debug(f"Error getting GPU metrics: {e}")
        
        return metrics
    
    def _log_hardware_state(self, event: str):
        """Log current hardware state"""
        logger.info(f"🖥️ Hardware State - {event}:")
        
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            allocated = torch.cuda.memory_allocated(device) / (1024**3)  # GB
            reserved = torch.cuda.memory_reserved(device) / (1024**3)    # GB
            total = torch.cuda.get_device_properties(device).total_memory / (1024**3)  # GB
            
            logger.info(f"   GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved, {total:.2f}GB total")
            logger.info(f"   GPU Utilization: {(allocated/total)*100:.1f}%")
        else:
            logger.info("   Using CPU (CUDA not available)")
    
    def _log_detailed_progress(self, logs: Dict[str, Any], state: TrainerState):
        """Log detailed progress information"""
        logger.info("📈 Detailed Progress:")
        
        # Training progress
        progress_percent = (state.global_step / state.max_steps) * 100 if state.max_steps > 0 else 0
        logger.info(f"   Progress: {progress_percent:.1f}% ({state.global_step}/{state.max_steps} steps)")
        
        # Loss information
        if 'train_loss' in logs:
            logger.info(f"   Current Loss: {logs['train_loss']:.4f} (Best: {self.best_loss:.4f})")
        
        # Performance metrics
        if 'steps_per_second' in logs:
            logger.info(f"   Performance: {logs['steps_per_second']:.2f} steps/sec")
        
        # GPU metrics
        if 'gpu_memory_allocated_mb' in logs:
            mem_mb = logs['gpu_memory_allocated_mb']
            logger.info(f"   GPU Memory: {mem_mb:.0f} MB allocated")
        
        # Time estimation
        if state.max_steps > 0 and self.start_time:
            elapsed = time.time() - self.start_time
            steps_remaining = state.max_steps - state.global_step
            if state.global_step > 0:
                time_per_step = elapsed / state.global_step
                eta = steps_remaining * time_per_step
                logger.info(f"   ETA: {self._format_duration(eta)}")
    
    def _format_duration(self, seconds: float) -> str:
        """Format duration in human-readable format"""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f}m"
        else:
            hours = seconds / 3600
            return f"{hours:.1f}h"
    
    def _save_current_metrics(self, step: int):
        """Save current training metrics to file"""
        try:
            metrics_file = self.log_dir / f"training_metrics_step_{step}.json"
            
            metrics_data = {
                "step": step,
                "timestamp": time.time(),
                "recent_metrics": self.training_metrics[-10:] if self.training_metrics else [],
                "epoch_metrics": self.epoch_metrics,
                "best_loss": self.best_loss,
                "total_steps_completed": len(self.training_metrics)
            }
            
            with open(metrics_file, 'w') as f:
                json.dump(metrics_data, f, indent=2)
                
        except Exception as e:
            logger.warning(f"Failed to save current metrics: {e}")
    
    def _save_training_summary(self, state: TrainerState, total_duration: float):
        """Save comprehensive training summary"""
        try:
            summary_file = self.log_dir / "training_summary.json"
            
            # Calculate performance statistics
            if self.step_times:
                avg_step_time = sum(self.step_times) / len(self.step_times)
                min_step_time = min(self.step_times)
                max_step_time = max(self.step_times)
            else:
                avg_step_time = min_step_time = max_step_time = 0
            
            summary = {
                "training_completed": True,
                "total_duration_seconds": total_duration,
                "total_duration_formatted": self._format_duration(total_duration),
                "final_step": state.global_step,
                "final_epoch": state.epoch,
                "best_loss": self.best_loss,
                "total_training_steps": len(self.training_metrics),
                "performance_stats": {
                    "avg_step_time": avg_step_time,
                    "min_step_time": min_step_time,
                    "max_step_time": max_step_time,
                    "steps_per_second": 1.0 / avg_step_time if avg_step_time > 0 else 0
                },
                "epoch_summary": self.epoch_metrics,
                "final_logs": state.log_history[-5:] if state.log_history else []
            }
            
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
            
            # Also save detailed metrics
            detailed_metrics_file = self.log_dir / "detailed_training_metrics.json"
            with open(detailed_metrics_file, 'w') as f:
                json.dump(self.training_metrics, f, indent=2)
            
            logger.info(f"📊 Training summary saved to: {summary_file}")
            logger.info(f"📊 Detailed metrics saved to: {detailed_metrics_file}")
            
        except Exception as e:
            logger.error(f"Failed to save training summary: {e}")
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get a summary of current training metrics"""
        if not self.training_metrics:
            return {"status": "no_metrics_available"}
        
        recent_metrics = self.training_metrics[-1]
        
        summary = {
            "current_step": recent_metrics.get("global_step", 0),
            "current_epoch": recent_metrics.get("epoch", 0),
            "current_loss": recent_metrics.get("train_loss"),
            "best_loss": self.best_loss,
            "learning_rate": recent_metrics.get("learning_rate"),
            "steps_per_second": recent_metrics.get("steps_per_second"),
            "gpu_memory_mb": recent_metrics.get("gpu_memory_allocated_mb"),
            "total_metrics_logged": len(self.training_metrics)
        }
        
        return summary
