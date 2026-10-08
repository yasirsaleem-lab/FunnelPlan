"""
Subsample raw corpora to the configured cap and save processed versions.
Run this first to prepare all datasets.
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.data_loader import get_processed
from config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    logger.info("Starting data subsampling...")

    logger.info("Processing NQ...")
    get_processed("nq", force_reload=True)

    logger.info("Processing HotpotQA...")
    get_processed("hotpotqa", force_reload=True)

    logger.info("All datasets processed and saved.")


if __name__ == "__main__":
    main()