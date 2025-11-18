from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import logging
import yaml  # make sure PyYAML is in requirements.txt

logger = logging.getLogger(__name__)


class ConfigNotFoundError(FileNotFoundError):
    """Raised when a config file is missing."""


class InvalidConfigError(ValueError):
    """Raised when a config file cannot be parsed or is invalid."""


def get_project_root() -> Path:
    """
    Returns the project root directory.

    Assumes this file is at: <root>/orchestrator/core/config_loader.py
    So root is 3 levels up from this file.
    """
    return Path(__file__).resolve().parents[2]


def _load_yaml(path: Path) -> Dict[str, Any]:
    """
    Load a YAML file and return as dict.

    Raises:
        ConfigNotFoundError: if file does not exist
        InvalidConfigError: if parsing fails or yaml is empty
    """
    if not path.exists():
        logger.error(f"Config file not found: {path}")
        raise ConfigNotFoundError(f"Config file not found: {path}")

    try:
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        logger.exception(f"Failed to parse YAML config: {path}")
        raise InvalidConfigError(f"Failed to parse YAML config: {path}") from exc

    if data is None:
        logger.error(f"Config file is empty or invalid: {path}")
        raise InvalidConfigError(f"Config file is empty or invalid: {path}")

    logger.debug(f"Loaded config from {path}")
    return data


def load_global_config() -> Dict[str, Any]:
    """
    Load the global orchestrator config from config/orchestrator.yaml
    relative to project root.
    """
    root = get_project_root()
    config_path = root / "config" / "orchestrator.yaml"
    logger.info(f"Loading global config from {config_path}")
    return _load_yaml(config_path)


def load_pipeline_config(pipeline_name: str) -> Dict[str, Any]:
    """
    Load a specific pipeline's config.

    Expects:
        orchestrator/pipelines/<pipeline_name>/config.yaml
    """
    root = get_project_root()
    config_path = root / "orchestrator" / "pipelines" / pipeline_name / "config.yaml"
    logger.info(f"Loading pipeline config for '{pipeline_name}' from {config_path}")
    return _load_yaml(config_path)
