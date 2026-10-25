"""
Application configuration using pydantic-settings.

All values can be overridden via environment variables or a .env file.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Tuple


class Settings(BaseSettings):
    # Dataset paths (set via .env or environment variables)
    DATASET_A_PATH: str = "./data/dataset_a"
    DATASET_B_OBJECTS_PATH: str = "./data/Objects"
    DATASET_B_TEXTURES_PATH: str = "./data/Textures"

    # Cache & index storage
    FEATURES_CACHE_DIR: str = "./data/cache/features"
    INDEX_DIR: str = "./data/cache/indexes"
    DB_PATH: str = "./data/cbir.db"

    # Static file serving — FastAPI mounts this at /static
    # Set to the parent of Objects/Textures so all images are accessible
    STATIC_DIR: str = "./data"

    # Search defaults
    MAX_RESULTS: int = 50

    # Descriptor defaults
    DEFAULT_COLOR_BINS: int = 64
    DEFAULT_LBP_POINTS: int = 24
    DEFAULT_LBP_RADIUS: int = 3
    DEFAULT_HOG_ORIENTATIONS: int = 9
    DEFAULT_HOG_PIXELS_PER_CELL: Tuple[int, int] = (8, 8)
    DEFAULT_HOG_CELLS_PER_BLOCK: Tuple[int, int] = (2, 2)

    # Image normalisation size before descriptor extraction
    IMAGE_RESIZE: Tuple[int, int] = (256, 256)

    # Logging
    LOG_LEVEL: str = "INFO"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        # Allow extra fields so the .env file can contain comments/extras
        extra="ignore",
    )


settings = Settings()
