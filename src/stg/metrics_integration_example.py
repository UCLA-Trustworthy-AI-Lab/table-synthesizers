"""
Example Integration of Metrics Manager with Table Synthesizers
==============================================================

This file demonstrates how to integrate the metrics_manager with existing
synthesizer algorithms to provide detailed training progress logging.
"""

import time
import pandas as pd
from typing import Dict, Any

from .metrics_manager import (
    create_algorithm_logger,
    get_metrics_manager,
    TVAELogger,
    CTGANLogger,
    PATECTGANLogger,
    SMOTELogger
)
from .tableSynthesizer import TableSynthesizer


class MetricsEnabledSynthesizer:
    """Wrapper around TableSynthesizer with integrated metrics logging"""

    def __init__(self, algorithm: str, params: Dict[str, Any], dataset_name: str = "unknown"):
        self.algorithm = algorithm
        self.params = params
        self.dataset_name = dataset_name
        self.synthesizer = TableSynthesizer(algorithm, params)
        self.logger = None
        self.metrics_manager = get_metrics_manager()

    def fit(self, data: pd.DataFrame, verbose: bool = True):
        """Fit the synthesizer with detailed metrics logging"""

        # Create metrics logger
        self.logger = create_algorithm_logger(
            algorithm=self.algorithm,
            dataset_name=self.dataset_name,
            epochs=self.params.get('epochs', 1),
            learning_rate=self.params.get('learning_rate', self.params.get('lr', 'auto')),
            batch_size=self.params.get('batch_size', 256),
            training_samples=len(data)
        )

        try:
            # Log data loading stage
            self.logger.log_stage('data_loading',
                                training_samples=len(data),
                                features=data.shape[1])

            # Log preprocessing stage
            self.logger.log_stage('preprocessing')

            # Log training started
            self.logger.log_stage('training', status='training')

            # Wrap the original fit method with progress tracking
            if self.algorithm.upper() == 'TVAE':
                self._fit_tvae_with_logging(data)
            elif self.algorithm.upper() == 'CTGAN':
                self._fit_ctgan_with_logging(data)
            elif self.algorithm.upper() == 'PATECTGAN':
                self._fit_patectgan_with_logging(data)
            elif self.algorithm.upper() == 'SMOTE':
                self._fit_smote_with_logging(data)
            else:
                self._fit_generic_with_logging(data)

            # Log completion
            self.logger.log_stage('completed', status='completed')

            if verbose:
                print(f"✅ {self.algorithm} training completed for {self.dataset_name}")

        except Exception as e:
            self.logger.log_stage('failed', status='failed', error=str(e))
            if verbose:
                print(f"❌ {self.algorithm} training failed for {self.dataset_name}: {e}")
            raise

    def _fit_tvae_with_logging(self, data: pd.DataFrame):
        """Fit TVAE with specific logging"""
        epochs = self.params.get('epochs', 1)

        # Simulate epoch-by-epoch training with progress logging
        for epoch in range(epochs):
            epoch_start = time.time()

            # In a real implementation, you would hook into the actual training loop
            # For now, we simulate the training process
            time.sleep(0.1)  # Simulate training time

            # Simulate metrics (in real implementation, these would come from the actual model)
            reconstruction_loss = 0.5 * (1 - epoch / epochs) + 0.1
            kl_divergence = 0.3 * (1 - epoch / epochs) + 0.05

            TVAELogger.log_epoch(
                self.logger,
                epoch=epoch + 1,
                reconstruction_loss=reconstruction_loss,
                kl_divergence=kl_divergence
            )

        # Actual fitting (this would be modified to include callbacks in real implementation)
        self.synthesizer.fit(data)

    def _fit_ctgan_with_logging(self, data: pd.DataFrame):
        """Fit CTGAN with specific logging"""
        epochs = self.params.get('epochs', 1)

        for epoch in range(epochs):
            epoch_start = time.time()

            # Simulate training
            time.sleep(0.1)

            # Simulate GAN losses
            generator_loss = 0.8 * (1 - epoch / epochs) + 0.2
            discriminator_loss = 0.6 * (1 - epoch / epochs) + 0.15

            CTGANLogger.log_epoch(
                self.logger,
                epoch=epoch + 1,
                generator_loss=generator_loss,
                discriminator_loss=discriminator_loss
            )

        # Actual fitting
        self.synthesizer.fit(data)

    def _fit_patectgan_with_logging(self, data: pd.DataFrame):
        """Fit PATECTGAN with specific logging"""
        epochs = self.params.get('epochs', 1)

        for epoch in range(epochs):
            epoch_start = time.time()

            # Simulate training
            time.sleep(0.1)

            # Simulate PATE-CTGAN losses
            generator_loss = 0.7 * (1 - epoch / epochs) + 0.18
            discriminator_loss = 0.55 * (1 - epoch / epochs) + 0.12
            privacy_loss = 0.1 * (1 - epoch / epochs) + 0.02

            PATECTGANLogger.log_epoch(
                self.logger,
                epoch=epoch + 1,
                generator_loss=generator_loss,
                discriminator_loss=discriminator_loss,
                privacy_loss=privacy_loss
            )

        # Actual fitting
        self.synthesizer.fit(data)

    def _fit_smote_with_logging(self, data: pd.DataFrame):
        """Fit SMOTE with specific logging"""
        SMOTELogger.log_stage(self.logger, 'nearest_neighbors_calculation')
        time.sleep(0.05)

        SMOTELogger.log_stage(self.logger, 'synthetic_sample_generation')
        time.sleep(0.05)

        # Actual fitting
        self.synthesizer.fit(data)

    def _fit_generic_with_logging(self, data: pd.DataFrame):
        """Generic fitting with basic logging"""
        epochs = self.params.get('epochs', 1)

        for epoch in range(epochs):
            time.sleep(0.05)
            self.logger.log_epoch(epoch=epoch + 1, loss=0.5 * (1 - epoch / epochs))

        # Actual fitting
        self.synthesizer.fit(data)

    def sample(self, n_samples: int, return_dataframe: bool = True):
        """Sample with logging"""
        if self.logger:
            self.logger.log_stage('sampling', samples_requested=n_samples)

        start_time = time.time()
        result = self.synthesizer.sample(n_samples, return_dataframe=return_dataframe)
        sample_time = time.time() - start_time

        if self.logger:
            self.logger.update_metrics(
                samples_generated=len(result) if hasattr(result, '__len__') else n_samples,
                sample_time=sample_time,
                samples_per_second=n_samples / sample_time if sample_time > 0 else 0
            )

        return result

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get training metrics summary"""
        if self.logger:
            return self.logger.get_summary()
        return {}

    def export_metrics(self, filepath: str):
        """Export training metrics to file"""
        if self.logger:
            self.metrics_manager.export_metrics(self.logger.session_id, filepath)


# Convenience function for parallel synthesis with metrics
def parallel_synthesis_with_metrics(task_data: Dict[str, Any]) -> Dict[str, Any]:
    """Enhanced parallel synthesis task with detailed metrics logging"""

    dataset_name = task_data['dataset_name']
    data = task_data['data']
    algorithm_name = task_data['algorithm']
    params = task_data['params']
    n_synth = task_data['n_synth']

    try:
        # Create metrics-enabled synthesizer
        synthesizer = MetricsEnabledSynthesizer(
            algorithm=algorithm_name,
            params=params,
            dataset_name=dataset_name
        )

        # Fit with metrics logging
        start_time = time.time()
        synthesizer.fit(data, verbose=True)

        # Sample with metrics logging
        synthetic_data = synthesizer.sample(n_synth, return_dataframe=True)
        synthesis_time = time.time() - start_time

        # Get final metrics
        metrics_summary = synthesizer.get_metrics_summary()

        return {
            'dataset': dataset_name,
            'algorithm': algorithm_name,
            'synthetic_data': synthetic_data,
            'synthesis_time': synthesis_time,
            'training_samples': len(data),
            'samples_generated': len(synthetic_data),
            'success': True,
            'error': None,
            'metrics_summary': metrics_summary
        }

    except Exception as e:
        return {
            'dataset': dataset_name,
            'algorithm': algorithm_name,
            'synthetic_data': None,
            'synthesis_time': 0,
            'training_samples': len(data) if 'data' in locals() else 0,
            'samples_generated': 0,
            'success': False,
            'error': str(e),
            'metrics_summary': {}
        }


# Example usage function
def demo_metrics_integration():
    """Demonstrate the metrics integration"""
    import numpy as np

    # Create sample data
    data = pd.DataFrame({
        'feature1': np.random.normal(0, 1, 1000),
        'feature2': np.random.exponential(2, 1000),
        'feature3': np.random.choice(['A', 'B', 'C'], 1000)
    })

    # Test different algorithms with metrics
    algorithms = ['TVAE', 'CTGAN', 'SMOTE']

    for algorithm in algorithms:
        print(f"\n🚀 Testing {algorithm} with metrics logging...")

        params = {'epochs': 3, 'batch_size': 256}
        synthesizer = MetricsEnabledSynthesizer(
            algorithm=algorithm,
            params=params,
            dataset_name='demo_dataset'
        )

        # Fit and sample
        synthesizer.fit(data)
        synthetic_data = synthesizer.sample(100)

        # Show metrics summary
        metrics = synthesizer.get_metrics_summary()
        print(f"📊 Final metrics: {metrics}")

    # Show global summary
    manager = get_metrics_manager()
    manager.print_global_summary()


if __name__ == "__main__":
    demo_metrics_integration()