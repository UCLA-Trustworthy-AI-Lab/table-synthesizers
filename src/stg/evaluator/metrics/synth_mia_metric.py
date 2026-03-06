import pandas as pd
import numpy as np
import sys
import os
from typing import Dict, Any, List

from ..interfaces.metric_interface import MetricInterface, PlotTypes
from ..utils.data_preprocessor import DataPreprocessor

# Add Synth-MIA to path
synth_mia_path = os.path.join(os.path.dirname(__file__), 'Synth-MIA')
if synth_mia_path not in sys.path:
    sys.path.append(synth_mia_path)

# Import Synth-MIA components
try:
    from synth_mia import (
        BaseAttacker, AttackEvaluator, 
        DCR, Classifier, LOGAN, DOMIAS, DPI, GenLRA, LocalNeighborhood, MC
    )
except ImportError as e:
    print(f"Warning: Could not import Synth-MIA components: {e}")
    # Fallback - define dummy classes to prevent import errors
    class BaseAttacker:
        pass
    class AttackEvaluator:
        pass
    DCR = Classifier = LOGAN = DOMIAS = DPI = GenLRA = LocalNeighborhood = MC = BaseAttacker

class SynthMIA(MetricInterface):
    """
    Membership Inference Attack evaluation using the Synth-MIA library.
    
    This metric evaluates privacy of synthetic data by running various membership
    inference attacks to test if attackers can determine whether specific records
    were in the original training data.
    """
    
    def __init__(self, train_data, synth_data, holdout_data, column_name_to_datatype, config=None, *args, **kwargs):
        super().__init__(train_data, synth_data, holdout_data, column_name_to_datatype, config, *args, **kwargs)
        
        self.plot_type = PlotTypes.PRECISION_RECALL
        self.isTable = True
        self.data_preprocessor = DataPreprocessor()
        
        # Available attackers - using all default attackers from Synth-MIA
        self.available_attackers = {
            'DCR': DCR,
            'Classifier': Classifier,
            'LOGAN': LOGAN,
            'DOMIAS': DOMIAS,
            'DPI': DPI,
            'GenLRA': GenLRA,
            'LocalNeighborhood': LocalNeighborhood,
            'MC': MC
        }
        
        # Configure which attackers to run (default to all)
        self.attackers_to_run = config.get('mia_attackers', list(self.available_attackers.keys())) if config else list(self.available_attackers.keys())
        self.mia_config = config.get('mia_config', {}) if config else {}
        
        # Calculate results
        self.table_params['table'] = self.calculate_table_params()

    def calculate_table_params(self):
        """
        Calculate membership inference attack results.
        
        Returns:
            dict: Results for each attacker with various metrics
        """
        try:
            # Preprocess data for Synth-MIA (convert to numpy)
            mem_data, non_mem_data, ref_data, synth_data = self._prepare_data()
            
            results = {}
            
            for attacker_name in self.attackers_to_run:
                if attacker_name not in self.available_attackers:
                    print(f"Warning: Attacker {attacker_name} not available, skipping")
                    continue
                    
                try:
                    # Initialize attacker
                    attacker_class = self.available_attackers[attacker_name]
                    attacker_params = self.mia_config.get(attacker_name.lower(), {})
                    attacker = attacker_class(**attacker_params)
                    
                    # Run attack
                    y_true, scores = attacker.attack(mem_data, non_mem_data, synth_data, ref_data)
                    
                    # Evaluate attack
                    attack_results = attacker.eval(
                        y_true, scores, 
                        metrics=['roc', 'classification', 'privacy', 'epsilon'],
                        **self.mia_config.get('eval_params', {})
                    )
                    
                    # Convert to DataFrame for consistent output format
                    results[attacker_name] = pd.DataFrame([attack_results])
                    
                except Exception as e:
                    print(f"Error running {attacker_name} attack: {e}")
                    # Store error info
                    results[attacker_name] = pd.DataFrame([{'error': str(e)}])
            
            return results
            
        except Exception as e:
            print(f"Error in SynthMIA calculation: {e}")
            return {'error': pd.DataFrame([{'error': str(e)}])}

    def calculate_value(self):
        """
        Calculate a single summary value for the metric.
        For MIA, we use the average AUC across all attackers as the summary.
        """
        if not hasattr(self, 'table_params') or 'table' not in self.table_params:
            return None
            
        results = self.table_params['table']
        if 'error' in results:
            return None
            
        auc_scores = []
        for attacker_name, result_df in results.items():
            if 'auc_roc' in result_df.columns:
                auc_scores.append(result_df['auc_roc'].iloc[0])
        
        return np.mean(auc_scores) if auc_scores else None

    def _prepare_data(self):
        """
        Prepare data for Synth-MIA attackers using provided holdout as the source
        of non-member (non_mem) and reference (ref) datasets.

        Config options (all optional):
        - mia_non_mem_index: list of indices (from holdout_data) to use as non-member
        - mia_ref_index: list of indices (from holdout_data) to use as reference
          If only one of the two is provided, the other is taken as the complement.
        - mia_non_mem_prop: float in (0,1). Fraction of holdout to assign to non_mem.
        - mia_split_seed: int. Random seed used when splitting by proportion.
        - mia_shuffle: bool. Whether to shuffle before splitting by proportion. Defaults True.

        Returns:
            tuple: (mem_data, non_mem_data, ref_data, synth_data) as numpy arrays
        """
        metadata = self.config.get('meta', self.column_name_to_datatype) if self.config else self.column_name_to_datatype

        # Fit preprocessing on train+synth to ensure consistent feature space
        train_processed, synth_processed = self.data_preprocessor.transform_real_and_synth(
            self.train_data, self.synth_data, metadata
        )

        # Split holdout into non_mem/ref
        non_mem_df, ref_df = self._split_holdout(self.holdout_data)

        # Transform holdout splits using the fitted encoders/scalers
        non_mem_processed = self.data_preprocessor.transform_test(non_mem_df, metadata)
        ref_processed = self.data_preprocessor.transform_test(ref_df, metadata)

        # Convert to numpy arrays (required by Synth-MIA)
        mem_data = train_processed.values.astype(np.float32)
        synth_data = synth_processed.values.astype(np.float32)
        non_mem_data = non_mem_processed.values.astype(np.float32)
        ref_data = ref_processed.values.astype(np.float32)

        return mem_data, non_mem_data, ref_data, synth_data

    def _split_holdout(self, holdout_df: pd.DataFrame):
        """Split holdout into non_mem and ref according to config or defaults.

        Defaults to a 50/50 split with shuffling using 'holdout_seed' if present.
        """
        if holdout_df is None or len(holdout_df) == 0:
            raise ValueError("Holdout data is required and must be non-empty for SynthMIA")

        idx = list(holdout_df.index)
        n = len(idx)

        # If explicit indices are provided
        non_mem_index = set(self.config.get('mia_non_mem_index', [])) if self.config else set()
        ref_index = set(self.config.get('mia_ref_index', [])) if self.config else set()

        if non_mem_index or ref_index:
            if not non_mem_index:
                non_mem_index = set(idx) - ref_index
            if not ref_index:
                ref_index = set(idx) - non_mem_index
            # Validate
            if not non_mem_index or not ref_index:
                raise ValueError("mia_non_mem_index and mia_ref_index must define non-empty disjoint splits")
            if non_mem_index & ref_index:
                raise ValueError("mia_non_mem_index and mia_ref_index must be disjoint")
            if not non_mem_index.issubset(set(idx)) or not ref_index.issubset(set(idx)):
                raise ValueError("Provided mia_* indices must be within holdout indices")
            return holdout_df.loc[sorted(non_mem_index)], holdout_df.loc[sorted(ref_index)]

        # Proportion-based split
        prop = None
        if self.config and 'mia_non_mem_prop' in self.config:
            try:
                prop = float(self.config['mia_non_mem_prop'])
            except Exception:
                prop = None
        if prop is None or not (0 < prop < 1):
            prop = 0.5

        seed = None
        if self.config:
            seed = self.config.get('mia_split_seed', self.config.get('holdout_seed', None))
        shuffle = True if (not self.config or self.config.get('mia_shuffle', True)) else False

        rng = np.random.RandomState(seed) if seed is not None else np.random
        order = idx.copy()
        if shuffle:
            rng.shuffle(order)

        k = int(round(prop * n))
        # Ensure both sides non-empty if possible
        if n >= 2:
            k = max(1, min(n - 1, k))
        non_mem_idx = order[:k]
        ref_idx = order[k:]

        return holdout_df.loc[non_mem_idx], holdout_df.loc[ref_idx]

    def get_plot_params(self):
        """
        Get parameters for plotting ROC/PR curves.
        
        Returns:
            dict: Plot parameters for visualization
        """
        if not hasattr(self, 'table_params') or 'table' not in self.table_params:
            return {}
            
        results = self.table_params['table']
        if 'error' in results:
            return {}
        
        # Extract data for plotting (could be used by precision_recall_plotter)
        plot_data = {}
        for attacker_name, result_df in results.items():
            if 'auc_roc' in result_df.columns:
                plot_data[attacker_name] = {
                    'auc': result_df['auc_roc'].iloc[0],
                    'precision': result_df.get('precision', [None]).iloc[0],
                    'recall': result_df.get('recall', [None]).iloc[0]
                }
        
        return {'attack_results': plot_data}
