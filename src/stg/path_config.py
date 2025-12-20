"""
Centralized path configuration for all table synthesizer models.

This module provides a single source of truth for all output directories,
ensuring consistency across all models.
"""

from pathlib import Path

# Base directories
PROJECT_ROOT = Path(__file__).parent.parent.parent  # Points to repo root
DATA_ROOT = PROJECT_ROOT / 'data'

# Model-specific directories
SMOTE_DIR = DATA_ROOT / 'SMOTE'
TABSYN_DIR = DATA_ROOT / 'TabSyn'
TABSYN_SYNTHETIC_DIR = TABSYN_DIR / 'synthetic'
TABSYN_IMPUTE_DIR = TABSYN_DIR / 'impute'
TABSYN_CKPT_DIR = TABSYN_DIR / 'ckpt'

# Ensure directories exist
def ensure_dirs():
    """Create all necessary directories if they don't exist."""
    for dir_path in [SMOTE_DIR, TABSYN_SYNTHETIC_DIR, TABSYN_IMPUTE_DIR, TABSYN_CKPT_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)

# Create directories on import
ensure_dirs()
