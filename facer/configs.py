import logging
from pathlib import Path

DATASET_PATH = Path("dataset")

TRAIN_PATH = Path(DATASET_PATH, "train")

VALIDATION_PATH = Path(DATASET_PATH, "val")

UNKNOWN_PATH = Path(DATASET_PATH, "unknown")

OUTPUT_PATH = Path("output")

DEFAULT_MODEL = "hog"

BOUNDING_BOX_COLOR = "blue"

TEXT_COLOR = "white"

IMAGE_FORMATS = ["png", "jpg", "jpeg"]


LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": (
                "[%(asctime)s][%(levelname)s]"
                "[%(name)s][%(funcName)s:%(lineno)s][%(message)s]"
            )
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "default",
        }
    },
    "loggers": {
        "root": {
            "handlers": ["console"],
            "level": logging.DEBUG,
            "propagate": False,
        },
    },
}
