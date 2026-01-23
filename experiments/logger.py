"""
Logging utils
"""

import logging
import sys

FORMATTER = logging.Formatter(
    "[%(asctime)-19.19s %(levelname)-1.1s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
LOGGER = logging.getLogger("tzq-sbi-dl")
LOGGER.setLevel(logging.DEBUG)
HANDLER = logging.StreamHandler(stream=sys.stdout)
HANDLER.setFormatter(FORMATTER)
LOGGER.addHandler(HANDLER)
