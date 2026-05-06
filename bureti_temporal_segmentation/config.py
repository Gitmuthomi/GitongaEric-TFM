"""
Central configuration loader for the Bureti temporal segmentation project.

"""

import os
from pathlib import Path

from dotenv import load_dotenv


_REPO_ROOT = Path(__file__).resolve().parent
load_dotenv(_REPO_ROOT / ".env")


def get_config() -> dict:
    """
    Load and validate project configuration from environment variables.

    Raises FileNotFoundError if DATA_DIR or SSL4EO_WEIGHTS are set explicitly
    but do not exist on disk, to catch misconfigured .env files early.

    Returns
    -------
    dict
        Configuration dictionary with the following keys:

        project_dir : pathlib.Path
            Repository root directory.
        data_dir : pathlib.Path
            Directory containing the multitemporal patch data
        output_dir : pathlib.Path
            Root output directory for checkpoints, metrics, and figures.
            Subdirectories per experiment are created
            by each training script.
        ssl4eo_weights : pathlib.Path
            Path to the converted 12-channel SSL4EO-S12 ResNet50 weights
            in Keras format (.keras).
        conda_env : str
            Name of the conda environment used in SLURM scripts.
        seeds : list of int
            Three decoder-initialisation seeds used across all experiments.
        num_classes : int
            Number of output classes
        ignore_label : int
            Pixel label to exclude from all metric computations (background
            / nodata).
        patch_size : int
            Spatial dimension of each patch in pixels.
        num_bands : int
            Number of Sentinel-2 bands accepted by the SSL4EO-S12 encoder.
    """
    project_dir = Path(os.getenv("PROJECT_DIR", _REPO_ROOT))

    config = {
        "project_dir": project_dir,
        "data_dir": Path(
            os.getenv(
                "DATA_DIR",
                project_dir / "Data" / "Buret_Multitemporal_Data",
            )
        ),
        "output_dir": Path(
            os.getenv("OUTPUT_DIR", project_dir / "outputs")
        ),
        "ssl4eo_weights": Path(
            os.getenv(
                "SSL4EO_WEIGHTS",
                project_dir / "ssl4eo_resnet50_12ch.keras",
            )
        ),
        # Environment
        "conda_env": os.getenv("CONDA_ENV_NAME", "tea_seg"),

        "seeds": [42, 123, 456],
        "num_classes": 3,
        "ignore_label": 255,
        "patch_size": 256,
        "num_bands": 12,
    }

    # Only fail if a path was explicitly set 
    # and still does not exist. 
    _required = {
        "DATA_DIR": config["data_dir"],
        "SSL4EO_WEIGHTS": config["ssl4eo_weights"],
    }
    for env_key, path in _required.items():
        if os.getenv(env_key) and not path.exists():
            raise FileNotFoundError(
                f"{env_key} is set in .env but the path does not exist:\n"
                f"  {path}\n"
                "Check your .env file or run the data preparation pipeline first."
            )

    return config
