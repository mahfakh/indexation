"""
CBIR Application - Dataset Setup Script

This script creates symlinks or copies the dataset files from their original location
to the project's data directory structure.

Usage:
    python scripts/setup_data.py --source "C:/Users/USER/Desktop/ter_indexation_p1_ateliers2_cours (1)/ter_indexation_p1_ateliers2_cours"
    python scripts/setup_data.py --source /path/to/dataset --copy  # Copy instead of symlink
"""

import os
import sys
import argparse
import shutil
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def setup_data(source_dir: str, project_root: str, copy_files: bool = False):
    """Set up dataset directory structure with symlinks or copies."""
    source = Path(source_dir)
    root = Path(project_root)
    data_dir = root / "data"

    # Verify source exists
    if not source.exists():
        logger.error(f"Source directory does not exist: {source}")
        sys.exit(1)

    # Create data directory structure
    for subdir in ["cache/features", "cache/indexes", "dataset_a"]:
        (data_dir / subdir).mkdir(parents=True, exist_ok=True)
    logger.info(f"Created data directory structure at {data_dir}")

    # Map source subdirectories to target locations
    mappings = {
        "idb/Objects": "Objects",
        "idb/Texture/jpg": "Textures",
    }

    for src_rel, dst_name in mappings.items():
        src_path = source / src_rel
        dst_path = data_dir / dst_name

        if not src_path.exists():
            logger.warning(f"Source path does not exist: {src_path}")
            continue

        if dst_path.exists():
            logger.info(f"Target already exists: {dst_path}")
            continue

        if copy_files:
            logger.info(f"Copying {src_path} -> {dst_path}")
            shutil.copytree(str(src_path), str(dst_path))
        else:
            logger.info(f"Creating symlink {dst_path} -> {src_path}")
            try:
                # On Windows, directory symlinks need special handling
                if sys.platform == "win32":
                    # Try symlink first (needs admin or developer mode)
                    try:
                        os.symlink(str(src_path), str(dst_path), target_is_directory=True)
                        logger.info(f"Symlink created successfully")
                    except OSError:
                        logger.warning(
                            "Symlink failed (requires admin/developer mode on Windows). "
                            "Creating junction instead..."
                        )
                        import subprocess
                        subprocess.run(
                            ["cmd", "/c", "mklink", "/J", str(dst_path), str(src_path)],
                            check=True, capture_output=True
                        )
                        logger.info(f"Junction created successfully")
                else:
                    os.symlink(str(src_path), str(dst_path))
            except Exception as e:
                logger.error(f"Failed to create link: {e}")
                logger.info("Falling back to copy...")
                shutil.copytree(str(src_path), str(dst_path))

    # Also link pre-existing VGG16 features if available
    vgg_features = source / "features"
    vgg_target = data_dir / "cache" / "features" / "vgg16"
    if vgg_features.exists() and not vgg_target.exists():
        if copy_files:
            logger.info(f"Copying VGG16 features...")
            shutil.copytree(str(vgg_features), str(vgg_target))
        else:
            try:
                if sys.platform == "win32":
                    try:
                        os.symlink(str(vgg_features), str(vgg_target), target_is_directory=True)
                    except OSError:
                        import subprocess
                        subprocess.run(
                            ["cmd", "/c", "mklink", "/J", str(vgg_target), str(vgg_features)],
                            check=True, capture_output=True
                        )
                else:
                    os.symlink(str(vgg_features), str(vgg_target))
                logger.info(f"Linked VGG16 features: {vgg_target}")
            except Exception as e:
                logger.warning(f"Could not link VGG16 features: {e}")

    # Create .env file if it doesn't exist
    env_file = root / ".env"
    if not env_file.exists():
        env_example = root / ".env.example"
        if env_example.exists():
            shutil.copy(str(env_example), str(env_file))
            logger.info(f"Created .env from .env.example")

    logger.info("Dataset setup complete!")
    logger.info(f"  Objects: {data_dir / 'Objects'}")
    logger.info(f"  Textures: {data_dir / 'Textures'}")

    # Print dataset summary
    for name in ["Objects", "Textures"]:
        path = data_dir / name
        if path.exists():
            if name == "Objects":
                categories = [d for d in path.iterdir() if d.is_dir()]
                total = sum(len(list(c.iterdir())) for c in categories)
                logger.info(f"  {name}: {len(categories)} categories, {total} images")
            else:
                images = list(path.glob("*.jpg")) + list(path.glob("*.png"))
                logger.info(f"  {name}: {len(images)} images")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Setup CBIR dataset")
    parser.add_argument(
        "--source",
        required=True,
        help="Path to the source dataset directory (ter_indexation_p1_ateliers2_cours)"
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy files instead of creating symlinks"
    )
    parser.add_argument(
        "--project-root",
        default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        help="Project root directory (default: parent of scripts/)"
    )

    args = parser.parse_args()
    setup_data(args.source, args.project_root, args.copy)
