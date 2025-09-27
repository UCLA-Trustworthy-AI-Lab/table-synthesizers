"""
Metrics Manager for Table Synthesizer Training
==============================================

This module provides comprehensive logging and tracking of training metrics
for different synthesis algorithms including TVAE, CTGAN, PATECTGAN, etc.

Usage:
    from stg.metrics_manager import MetricsManager, MetricsLogger

    # Initialize metrics manager
    metrics = MetricsManager()

    # Log training metrics
    metrics.log_epoch(algorithm='CTGAN', epoch=1, generator_loss=0.5, discriminator_loss=0.3)

    # Get progress summary
    summary = metrics.get_progress_summary()
"""

import time
import threading
import json
from typing import Dict, Any, Optional, List
from datetime import datetime
import logging


class MetricsLogger:
    """Individual logger for a specific training session"""

    def __init__(self, algorithm: str, dataset_name: str, session_id: str):
        self.algorithm = algorithm
        self.dataset_name = dataset_name
        self.session_id = session_id
        self.start_time = time.time()
        self.metrics_history = []
        self.current_metrics = {
            'algorithm': algorithm,
            'dataset': dataset_name,
            'session_id': session_id,
            'status': 'initializing',
            'epoch': 0,
            'total_epochs': 0,
            'elapsed_time': 0,
            'training_samples': 0,
            'learning_rate': None,
            'batch_size': None,
            # Algorithm-specific metrics
            'loss': None,
            'generator_loss': None,
            'discriminator_loss': None,
            'reconstruction_loss': None,
            'kl_divergence': None,
            'validation_loss': None,
            # Performance metrics
            'samples_per_second': None,
            'memory_usage_mb': None,
            'gpu_utilization': None,
            # Progress tracking
            'stage': 'initializing',
            'progress_percent': 0.0,
            'eta_seconds': None,
        }

    def update_metrics(self, **kwargs):
        """Update current metrics"""
        self.current_metrics.update(kwargs)
        self.current_metrics['elapsed_time'] = time.time() - self.start_time

        # Calculate progress percentage
        if self.current_metrics['total_epochs'] > 0:
            self.current_metrics['progress_percent'] = (
                self.current_metrics['epoch'] / self.current_metrics['total_epochs'] * 100
            )

        # Store in history
        metrics_snapshot = self.current_metrics.copy()
        metrics_snapshot['timestamp'] = datetime.now().isoformat()
        self.metrics_history.append(metrics_snapshot)

    def log_epoch(self, epoch: int, **metrics):
        """Log metrics for a specific epoch"""
        self.update_metrics(epoch=epoch, stage=f'epoch_{epoch}', **metrics)
        self._print_epoch_progress()

    def log_stage(self, stage: str, **metrics):
        """Log a training stage (e.g., 'data_loading', 'training', 'sampling')"""
        self.update_metrics(stage=stage, **metrics)
        self._print_stage_progress()

    def _print_epoch_progress(self):
        """Print formatted epoch progress"""
        m = self.current_metrics

        # Build progress string
        progress_parts = []
        progress_parts.append(f"📊 {m['algorithm']} | {m['dataset'][:30]}")
        progress_parts.append(f"Epoch {m['epoch']}/{m['total_epochs']}")

        if m['progress_percent'] > 0:
            progress_bar = self._create_progress_bar(m['progress_percent'])
            progress_parts.append(f"{progress_bar} {m['progress_percent']:.1f}%")

        progress_parts.append(f"⏱️ {m['elapsed_time']:.1f}s")

        # Add algorithm-specific metrics
        if m['loss'] is not None:
            progress_parts.append(f"Loss: {m['loss']:.4f}")

        if m['generator_loss'] is not None and m['discriminator_loss'] is not None:
            progress_parts.append(f"G_Loss: {m['generator_loss']:.4f}")
            progress_parts.append(f"D_Loss: {m['discriminator_loss']:.4f}")

        if m['reconstruction_loss'] is not None:
            progress_parts.append(f"Recon: {m['reconstruction_loss']:.4f}")

        if m['kl_divergence'] is not None:
            progress_parts.append(f"KL: {m['kl_divergence']:.4f}")

        if m['learning_rate'] is not None:
            progress_parts.append(f"LR: {m['learning_rate']}")

        print(" | ".join(progress_parts))

    def _print_stage_progress(self):
        """Print formatted stage progress"""
        m = self.current_metrics
        stage_emoji = {
            'initializing': '🔄',
            'data_loading': '📥',
            'preprocessing': '⚙️',
            'training': '🎯',
            'validation': '✅',
            'sampling': '🎲',
            'completed': '✅',
            'failed': '❌'
        }

        emoji = stage_emoji.get(m['stage'], '📊')
        print(f"{emoji} {m['algorithm']} | {m['dataset'][:30]} | {m['stage'].replace('_', ' ').title()} | ⏱️ {m['elapsed_time']:.1f}s")

    def _create_progress_bar(self, percent: float, width: int = 20) -> str:
        """Create a text progress bar"""
        filled = int(width * percent / 100)
        bar = '█' * filled + '░' * (width - filled)
        return f"[{bar}]"

    def get_summary(self) -> Dict[str, Any]:
        """Get current metrics summary"""
        return self.current_metrics.copy()

    def get_history(self) -> List[Dict[str, Any]]:
        """Get full metrics history"""
        return self.metrics_history.copy()


class MetricsManager:
    """Global metrics manager for all training sessions"""

    def __init__(self):
        self.loggers: Dict[str, MetricsLogger] = {}
        self.lock = threading.Lock()
        self.global_start_time = time.time()

    def create_logger(self, algorithm: str, dataset_name: str, session_id: Optional[str] = None) -> MetricsLogger:
        """Create a new metrics logger for a training session"""
        if session_id is None:
            session_id = f"{algorithm}_{dataset_name}_{int(time.time())}_{threading.current_thread().ident}"

        with self.lock:
            logger = MetricsLogger(algorithm, dataset_name, session_id)
            self.loggers[session_id] = logger
            logger.log_stage('initializing')
            return logger

    def get_logger(self, session_id: str) -> Optional[MetricsLogger]:
        """Get an existing logger by session ID"""
        with self.lock:
            return self.loggers.get(session_id)

    def remove_logger(self, session_id: str):
        """Remove a logger (typically after training completion)"""
        with self.lock:
            if session_id in self.loggers:
                del self.loggers[session_id]

    def get_all_active_sessions(self) -> Dict[str, Dict[str, Any]]:
        """Get summaries of all active training sessions"""
        with self.lock:
            return {
                session_id: logger.get_summary()
                for session_id, logger in self.loggers.items()
            }

    def print_global_summary(self):
        """Print a summary of all active training sessions"""
        summaries = self.get_all_active_sessions()

        if not summaries:
            print("📊 No active training sessions")
            return

        print(f"\n📊 TRAINING PROGRESS SUMMARY ({len(summaries)} active sessions)")
        print("=" * 80)

        for session_id, summary in summaries.items():
            status_emoji = {
                'initializing': '🔄',
                'training': '🎯',
                'sampling': '🎲',
                'completed': '✅',
                'failed': '❌'
            }

            emoji = status_emoji.get(summary['status'], '📊')
            print(f"{emoji} {summary['algorithm']:12} | {summary['dataset'][:25]:25} | "
                  f"Epoch {summary['epoch']:3}/{summary['total_epochs']:3} | "
                  f"{summary['progress_percent']:5.1f}% | "
                  f"{summary['elapsed_time']:6.1f}s")

        print("=" * 80)

    def export_metrics(self, session_id: str, filepath: str):
        """Export metrics history to JSON file"""
        logger = self.get_logger(session_id)
        if logger:
            with open(filepath, 'w') as f:
                json.dump({
                    'session_info': {
                        'algorithm': logger.algorithm,
                        'dataset': logger.dataset_name,
                        'session_id': logger.session_id,
                        'start_time': logger.start_time
                    },
                    'metrics_history': logger.get_history(),
                    'final_summary': logger.get_summary()
                }, f, indent=2)


# Global instance for easy access
_global_metrics_manager = MetricsManager()


def get_metrics_manager() -> MetricsManager:
    """Get the global metrics manager instance"""
    return _global_metrics_manager


def create_algorithm_logger(algorithm: str, dataset_name: str, **params) -> MetricsLogger:
    """Convenience function to create a new algorithm logger"""
    manager = get_metrics_manager()
    logger = manager.create_logger(algorithm, dataset_name)

    # Initialize with algorithm parameters
    logger.update_metrics(
        total_epochs=params.get('epochs', 1),
        learning_rate=params.get('learning_rate', params.get('lr', None)),
        batch_size=params.get('batch_size', None),
        training_samples=params.get('training_samples', 0),
        status='initialized'
    )

    return logger


# Algorithm-specific logging helpers
class TVAELogger:
    """Specialized logger for TVAE algorithm"""

    @staticmethod
    def log_epoch(logger: MetricsLogger, epoch: int, reconstruction_loss: float,
                  kl_divergence: float, **kwargs):
        logger.log_epoch(
            epoch=epoch,
            reconstruction_loss=reconstruction_loss,
            kl_divergence=kl_divergence,
            loss=reconstruction_loss + kl_divergence,
            **kwargs
        )


class CTGANLogger:
    """Specialized logger for CTGAN algorithm"""

    @staticmethod
    def log_epoch(logger: MetricsLogger, epoch: int, generator_loss: float,
                  discriminator_loss: float, **kwargs):
        logger.log_epoch(
            epoch=epoch,
            generator_loss=generator_loss,
            discriminator_loss=discriminator_loss,
            loss=(generator_loss + discriminator_loss) / 2,
            **kwargs
        )


class PATECTGANLogger:
    """Specialized logger for PATECTGAN algorithm"""

    @staticmethod
    def log_epoch(logger: MetricsLogger, epoch: int, generator_loss: float,
                  discriminator_loss: float, privacy_loss: float = None, **kwargs):
        total_loss = generator_loss + discriminator_loss
        if privacy_loss is not None:
            total_loss += privacy_loss

        logger.log_epoch(
            epoch=epoch,
            generator_loss=generator_loss,
            discriminator_loss=discriminator_loss,
            privacy_loss=privacy_loss,
            loss=total_loss,
            **kwargs
        )


class SMOTELogger:
    """Specialized logger for SMOTE algorithm"""

    @staticmethod
    def log_stage(logger: MetricsLogger, stage: str, **kwargs):
        logger.log_stage(stage, **kwargs)