"""
ConfigManager - Unified Configuration Management for Table Synthesizers
========================================================================

Provides centralized configuration handling with support for:
- Model-specific configurations
- Configuration profiles (default, quick, production)
- Configuration templates
- Environment variable overrides

Usage:
    from stg.config_manager import ConfigManager

    # Initialize
    cm = ConfigManager()

    # Load configuration with profile
    config = cm.load_config('TVAE', profile='production')

    # Apply template
    config = cm.load_config_with_template('TVAE', 'quick_test')

    # Environment variables automatically override config values
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from copy import deepcopy
import logging


class ConfigManager:
    """
    Centralized configuration management for table synthesizers.

    Handles loading, validation, and management of:
    - Model-specific configurations
    - Configuration profiles
    - Configuration templates
    - Environment variable overrides
    """

    def __init__(self, config_dir: Optional[str] = None,
                 global_config_path: Optional[str] = None):
        """
        Initialize ConfigManager.

        Args:
            config_dir: Directory containing configuration files (default: './config')
            global_config_path: Path to global configuration file
        """
        self.config_dir = Path(config_dir) if config_dir else Path('config')
        self.templates_dir = self.config_dir / 'templates'
        self.global_config_path = (Path(global_config_path) if global_config_path
                                  else self.config_dir / 'global_config.json')

        # Create directories if they don't exist
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.templates_dir.mkdir(parents=True, exist_ok=True)

        # Load global configuration
        self.global_config = self._load_global_config()

        # Logger
        self.logger = logging.getLogger(__name__)

    # ============================================================================
    # Global Configuration
    # ============================================================================

    def _load_global_config(self) -> Dict[str, Any]:
        """Load global configuration with defaults."""
        if self.global_config_path.exists():
            with open(self.global_config_path, 'r') as f:
                return json.load(f)

        # Default global configuration
        return {
            "version": "1.0",
            "environment": {
                "data_dir": "data",
                "config_dir": "config",
                "temp_cleanup": True,
                "checkpoint_retention": 5
            },
            "defaults": {
                "training": {
                    "epochs": 100,
                    "batch_size": 32,
                    "learning_rate": 0.001
                },
                "monitoring": {
                    "enable_wandb": False,
                    "log_frequency": 100,
                    "save_frequency": 1000
                }
            }
        }

    # ============================================================================
    # Configuration Loading
    # ============================================================================

    def load_config(self, model_name: str, profile: str = "default") -> Dict[str, Any]:
        """
        Load configuration for a specific model and profile.

        Args:
            model_name: Name of the model (e.g., 'TVAE', 'CTGAN')
            profile: Configuration profile ('default', 'quick', 'production')

        Returns:
            Dictionary containing merged configuration
        """
        config_path = self.config_dir / f'{model_name}.json'

        if not config_path.exists():
            self.logger.warning(f"Config file not found: {config_path}, using defaults")
            return self._get_default_config(model_name, profile)

        with open(config_path, 'r') as f:
            config_data = json.load(f)

        # Get the profile configuration
        if 'profiles' not in config_data:
            self.logger.warning(f"No profiles found in {config_path}")
            return config_data

        if profile not in config_data['profiles']:
            self.logger.warning(f"Profile '{profile}' not found, using 'default'")
            profile = 'default'

        # Start with default profile as base
        base_config = deepcopy(config_data['profiles'].get('default', {}))

        # Merge with requested profile if different
        if profile != 'default':
            profile_config = config_data['profiles'].get(profile, {})
            base_config = self._deep_merge(base_config, profile_config)

        # Apply environment variable overrides
        base_config = self.apply_env_overrides(base_config, model_name)

        return base_config

    def _get_default_config(self, model_name: str, profile: str) -> Dict[str, Any]:
        """Get default configuration when config file doesn't exist."""
        config = deepcopy(self.global_config['defaults'])

        # Add model-specific defaults based on model type
        if model_name in ['TVAE', 'LTM_VAE']:
            config['architecture'] = {
                'embedding_dim': 128,
                'compress_dims': [256, 128],
                'decompress_dims': [128, 256]
            }
        elif model_name in ['CTGAN', 'PATECTGAN']:
            config['architecture'] = {
                'embedding_dim': 128,
                'generator_dim': [256, 256],
                'discriminator_dim': [256, 256],
                'pac': 10
            }
        elif model_name == 'TabDDPM':
            config['training']['num_timesteps'] = 1000

        # Apply profile modifications
        if profile == 'quick':
            config['training']['epochs'] = 2
        elif profile == 'production':
            config['training']['epochs'] = 300
            config['monitoring']['enable_wandb'] = True

        return config

    def get_available_profiles(self, model_name: str) -> List[str]:
        """Get list of available profiles for a model."""
        config_path = self.config_dir / f'{model_name}.json'

        if not config_path.exists():
            return ['default', 'quick', 'production']

        with open(config_path, 'r') as f:
            config_data = json.load(f)

        return list(config_data.get('profiles', {}).keys())

    # ============================================================================
    # Template Management
    # ============================================================================

    def load_template(self, template_name: str) -> Dict[str, Any]:
        """
        Load a configuration template.

        Args:
            template_name: Name of the template (e.g., 'quick_test', 'production')

        Returns:
            Dictionary containing template configuration
        """
        template_path = self.templates_dir / f'{template_name}.json'

        if not template_path.exists():
            raise FileNotFoundError(f"Template not found: {template_path}")

        with open(template_path, 'r') as f:
            template = json.load(f)

        return template

    def apply_template(self, model_name: str, template_name: str) -> Dict[str, Any]:
        """
        Apply a template to a model's base configuration.

        Args:
            model_name: Name of the model
            template_name: Name of the template to apply

        Returns:
            Merged configuration
        """
        # Load base configuration
        base_config = self.load_config(model_name, 'default')

        # Load template
        template = self.load_template(template_name)

        # Check if template is applicable to this model
        if 'applicable_models' in template:
            if model_name not in template['applicable_models']:
                self.logger.warning(
                    f"Template '{template_name}' may not be suitable for {model_name}"
                )

        # Apply template overrides
        overrides = template.get('overrides', {})
        config = self._apply_nested_overrides(base_config, overrides)

        return config

    def load_config_with_template(self, model_name: str, template: str,
                                  overrides: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Load configuration with template and additional overrides.

        Args:
            model_name: Name of the model
            template: Template name to apply
            overrides: Additional overrides to apply

        Returns:
            Final merged configuration
        """
        # Apply template
        config = self.apply_template(model_name, template)

        # Apply additional overrides
        if overrides:
            config = self._apply_nested_overrides(config, overrides)

        # Apply environment variable overrides
        config = self.apply_env_overrides(config, model_name)

        return config

    # ============================================================================
    # Environment Variable Integration
    # ============================================================================

    def apply_env_overrides(self, config: Dict, model_name: str) -> Dict[str, Any]:
        """
        Apply environment variable overrides to configuration.

        Environment variables follow the pattern:
        - Global: TABLE_SYNTH_<PARAM_NAME>
        - Model-specific: TABLE_SYNTH_<MODEL>_<PARAM_NAME>

        Args:
            config: Base configuration dictionary
            model_name: Name of the model

        Returns:
            Configuration with environment overrides applied
        """
        config = deepcopy(config)

        # Check for global environment variables
        for key in config.keys():
            env_var = f'TABLE_SYNTH_{key.upper()}'
            if env_var in os.environ:
                config[key] = self._parse_env_value(os.environ[env_var])

        # Check for nested training parameters
        if 'training' in config:
            for key in config['training'].keys():
                env_var = f'TABLE_SYNTH_{key.upper()}'
                if env_var in os.environ:
                    config['training'][key] = self._parse_env_value(os.environ[env_var])

        # Check for model-specific overrides
        model_prefix = f'TABLE_SYNTH_{model_name.upper()}_'
        for env_var, value in os.environ.items():
            if env_var.startswith(model_prefix):
                param_name = env_var[len(model_prefix):].lower()
                # Try to place in appropriate section
                if 'training' in config and param_name in ['epochs', 'batch_size', 'learning_rate']:
                    config['training'][param_name] = self._parse_env_value(value)
                elif 'architecture' in config:
                    config['architecture'][param_name] = self._parse_env_value(value)

        return config

    def get_env_override(self, param_path: str, model_name: Optional[str] = None) -> Any:
        """
        Get a specific environment variable override.

        Args:
            param_path: Dot-separated parameter path (e.g., 'training.epochs')
            model_name: Optional model name for model-specific override

        Returns:
            Override value or None
        """
        # Model-specific check
        if model_name:
            env_var = f'TABLE_SYNTH_{model_name.upper()}_{param_path.replace(".", "_").upper()}'
            if env_var in os.environ:
                return self._parse_env_value(os.environ[env_var])

        # Global check
        env_var = f'TABLE_SYNTH_{param_path.replace(".", "_").upper()}'
        if env_var in os.environ:
            return self._parse_env_value(os.environ[env_var])

        return None

    def _parse_env_value(self, value: str) -> Any:
        """Parse environment variable value to appropriate type."""
        # Try to parse as JSON first (handles complex types)
        try:
            return json.loads(value)
        except (json.JSONDecodeError, ValueError):
            pass

        # Try int
        try:
            return int(value)
        except ValueError:
            pass

        # Try float
        try:
            return float(value)
        except ValueError:
            pass

        # Try boolean
        if value.lower() in ['true', 'false']:
            return value.lower() == 'true'

        # Return as string
        return value

    # ============================================================================
    # Configuration Management
    # ============================================================================

    def save_config(self, model_name: str, config: Dict, profile: str = "default"):
        """
        Save configuration to file.

        Args:
            model_name: Name of the model
            config: Configuration dictionary
            profile: Profile name to save under
        """
        config_path = self.config_dir / f'{model_name}.json'

        # Load existing config or create new
        if config_path.exists():
            with open(config_path, 'r') as f:
                full_config = json.load(f)
        else:
            full_config = {
                'model_name': model_name,
                'version': '1.0',
                'profiles': {}
            }

        # Update the profile
        full_config['profiles'][profile] = config

        # Save
        with open(config_path, 'w') as f:
            json.dump(full_config, f, indent=2)

        self.logger.info(f"Saved config for {model_name} (profile: {profile})")

    def create_custom_profile(self, model_name: str, base_profile: str,
                             custom_profile: str, overrides: Dict):
        """
        Create a custom profile based on an existing profile.

        Args:
            model_name: Name of the model
            base_profile: Base profile to start from
            custom_profile: Name for the new profile
            overrides: Dictionary of overrides to apply
        """
        # Load base profile
        base_config = self.load_config(model_name, base_profile)

        # Apply overrides
        custom_config = self._deep_merge(base_config, overrides)

        # Save as new profile
        self.save_config(model_name, custom_config, custom_profile)

        self.logger.info(f"Created custom profile '{custom_profile}' for {model_name}")

    # ============================================================================
    # Utility Methods
    # ============================================================================

    def _deep_merge(self, base: Dict, override: Dict) -> Dict:
        """Deep merge two dictionaries."""
        result = deepcopy(base)

        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = deepcopy(value)

        return result

    def _apply_nested_overrides(self, config: Dict, overrides: Dict) -> Dict:
        """Apply nested overrides using dot notation (e.g., 'training.epochs')."""
        config = deepcopy(config)

        for key, value in overrides.items():
            if '.' in key:
                # Handle nested keys
                keys = key.split('.')
                current = config
                for k in keys[:-1]:
                    if k not in current:
                        current[k] = {}
                    current = current[k]
                current[keys[-1]] = value
            else:
                config[key] = value

        return config

    def validate_config(self, model_name: str, config: Dict) -> tuple[bool, List[str]]:
        """
        Validate configuration for a model.

        Args:
            model_name: Name of the model
            config: Configuration to validate

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []

        # Check for required training parameters
        if 'training' in config:
            required = ['epochs', 'batch_size']
            for param in required:
                if param not in config['training']:
                    errors.append(f"Missing required parameter: training.{param}")

        # Model-specific validation
        if model_name in ['CTGAN', 'PATECTGAN']:
            if 'architecture' in config and 'pac' in config['architecture']:
                pac = config['architecture']['pac']
                batch_size = config.get('training', {}).get('batch_size', 32)
                if batch_size % pac != 0:
                    errors.append(f"batch_size ({batch_size}) must be divisible by pac ({pac})")

        return (len(errors) == 0, errors)

    def __repr__(self) -> str:
        """String representation."""
        return f"ConfigManager(config_dir='{self.config_dir}')"