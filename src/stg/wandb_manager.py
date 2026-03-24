import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
import wandb


class WandbManager:
    """
    Centralized Weights & Biases experiment tracking manager for table synthesizers.

    This class handles experiment initialization, metric logging, artifact management,
    and configuration loading for all synthesizer models in the framework.
    """

    def __init__(self, model_name: str, config_dir: str = "config/wandb"):
        """
        Initialize WandbManager for a specific model.

        Args:
            model_name: Name of the synthesizer model (e.g., 'TVAE', 'CTGAN')
            config_dir: Directory containing wandb configuration files
        """
        self.model_name = model_name
        self.config_dir = Path(config_dir)
        self.global_config = self._load_global_config()
        self.model_config = self._load_model_config()
        self.run = None

    def _load_global_config(self) -> Dict[str, Any]:
        """Load global wandb configuration."""
        global_config_path = self.config_dir / "global_wandb.json"
        if global_config_path.exists():
            with open(global_config_path, 'r') as f:
                return json.load(f)
        return {
            "project_name": "table-synthesizers",
            "logging_settings": {
                "log_frequency": 10,
                "save_checkpoints": True,
                "log_artifacts": True
            }
        }

    def _load_model_config(self) -> Dict[str, Any]:
        """Load model-specific wandb configuration."""
        model_config_path = self.config_dir / f"{self.model_name}_wandb.json"
        if model_config_path.exists():
            with open(model_config_path, 'r') as f:
                return json.load(f)

        # Fallback to template if model-specific config doesn't exist
        template_path = self._find_applicable_template()
        if template_path and template_path.exists():
            with open(template_path, 'r') as f:
                template = json.load(f)
                return {
                    "model_name": self.model_name,
                    "metrics": template["metrics"],
                    "hyperparameters": ["epochs", "batch_size", "learning_rate"],
                    "artifacts": {"model_checkpoints": True, "generated_samples": True},
                    "custom_plots": template.get("custom_plots", ["loss_curves"])
                }

        # Default configuration
        return {
            "model_name": self.model_name,
            "metrics": {
                "training": ["train_loss", "epoch_time"],
                "validation": ["val_loss"],
                "generation": ["sample_quality_score"]
            },
            "hyperparameters": ["epochs", "batch_size", "learning_rate"],
            "artifacts": {"model_checkpoints": True, "generated_samples": True},
            "custom_plots": ["loss_curves"]
        }

    def _find_applicable_template(self) -> Optional[Path]:
        """Find the most applicable metric template for the model."""
        templates_dir = self.config_dir / "templates"
        if not templates_dir.exists():
            return None

        # Template priority mapping
        template_mapping = {
            "TVAE": "vae_metrics.json",
            "LTM_VAE": "vae_metrics.json",
            "AutoDiff": "vae_metrics.json",
            "CTGAN": "gan_metrics.json",
            "PATECTGAN": "gan_metrics.json",
            "TabDDPM": "diffusion_metrics.json"
        }

        template_name = template_mapping.get(self.model_name)
        if template_name:
            return templates_dir / template_name

        # Default to vae_metrics if available
        return templates_dir / "vae_metrics.json" if (templates_dir / "vae_metrics.json").exists() else None

    def init_experiment(self, config: Dict[str, Any], tags: Optional[List[str]] = None,
                       notes: Optional[str] = None) -> wandb.run:
        """
        Initialize a wandb experiment run.

        Args:
            config: Hyperparameters and configuration for the experiment
            tags: Optional tags for the experiment
            notes: Optional notes describing the experiment

        Returns:
            wandb.run: The initialized wandb run
        """
        # Filter config to only include hyperparameters defined in model config
        filtered_config = {}
        hyperparams = self.model_config.get("hyperparameters", [])
        for param in hyperparams:
            if param in config:
                filtered_config[param] = config[param]

        # Add model metadata
        filtered_config["model_name"] = self.model_name

        # Initialize wandb run
        self.run = wandb.init(
            project=self.global_config.get("project_name", "table-synthesizers"),
            name=f"{self.model_name}_{wandb.util.generate_id()}",
            config=filtered_config,
            tags=tags or [self.model_name],
            notes=notes
        )

        return self.run

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None,
                   phase: str = "training"):
        """
        Log metrics to wandb.

        Args:
            metrics: Dictionary of metric name -> value pairs
            step: Optional step number (epoch, iteration, etc.)
            phase: Phase of training ("training", "validation", "generation")
        """
        if not self.run:
            return

        # Filter metrics based on model configuration
        allowed_metrics = self.model_config.get("metrics", {}).get(phase, [])
        if allowed_metrics:
            filtered_metrics = {k: v for k, v in metrics.items() if k in allowed_metrics}
        else:
            filtered_metrics = metrics

        # Log with step if provided
        if step is not None:
            self.run.log(filtered_metrics, step=step)
        else:
            self.run.log(filtered_metrics)

    def log_artifact(self, file_path: str, artifact_name: str, artifact_type: str,
                    description: Optional[str] = None):
        """
        Log an artifact to wandb.

        Args:
            file_path: Path to the file to upload
            artifact_name: Name of the artifact
            artifact_type: Type of artifact (e.g., 'model', 'dataset', 'samples')
            description: Optional description of the artifact
        """
        if not self.run or not self.global_config.get("logging_settings", {}).get("log_artifacts", True):
            return

        artifact = wandb.Artifact(
            name=artifact_name,
            type=artifact_type,
            description=description or f"{artifact_type} for {self.model_name}"
        )
        artifact.add_file(file_path)
        self.run.log_artifact(artifact)

    def log_model_checkpoint(self, checkpoint_path: str, epoch: int):
        """
        Log a model checkpoint as an artifact.

        Args:
            checkpoint_path: Path to the checkpoint file
            epoch: Epoch number for the checkpoint
        """
        if not self.model_config.get("artifacts", {}).get("model_checkpoints", False):
            return

        self.log_artifact(
            file_path=checkpoint_path,
            artifact_name=f"{self.model_name}_checkpoint_epoch_{epoch}",
            artifact_type="model",
            description=f"Model checkpoint at epoch {epoch}"
        )

    def log_generated_samples(self, samples_path: str, num_samples: int):
        """
        Log generated samples as an artifact.

        Args:
            samples_path: Path to the generated samples file
            num_samples: Number of generated samples
        """
        if not self.model_config.get("artifacts", {}).get("generated_samples", False):
            return

        self.log_artifact(
            file_path=samples_path,
            artifact_name=f"{self.model_name}_samples_{num_samples}",
            artifact_type="dataset",
            description=f"{num_samples} synthetic samples generated by {self.model_name}"
        )

    def create_custom_plots(self, plot_data: Dict[str, Any]):
        """
        Create custom plots based on model configuration.

        Args:
            plot_data: Dictionary containing data for different plot types
        """
        if not self.run:
            return

        custom_plots = self.model_config.get("custom_plots", [])

        for plot_type in custom_plots:
            if plot_type in plot_data:
                data = plot_data[plot_type]

                if plot_type == "loss_curves" and "epochs" in data and "losses" in data:
                    wandb.log({
                        "loss_curve": wandb.plot.line_series(
                            xs=data["epochs"],
                            ys=data["losses"],
                            keys=list(data["losses"].keys()) if isinstance(data["losses"], dict) else ["loss"],
                            title="Training Loss Curves",
                            xname="Epoch"
                        )
                    })

                elif plot_type == "latent_space_visualization" and "embeddings" in data:
                    # Log embeddings table for wandb's built-in visualization
                    self.run.log({"latent_embeddings": wandb.Table(data=data["embeddings"])})

                # Add more custom plot types as needed

    def finish_experiment(self):
        """Finish the wandb experiment."""
        if self.run:
            wandb.finish()
            self.run = None

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.finish_experiment()