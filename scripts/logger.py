# logger.py
import logging
from rich.logging import RichHandler
from pathlib import Path

# Ensure the log folder exists
Path("results/logs").mkdir(parents=True, exist_ok=True)

# File handler (plain text)
file_handler = logging.FileHandler("results/logs/pipeline.log")
file_handler.setLevel(logging.DEBUG)

# Rich console handler (pretty)
console_handler = RichHandler(rich_tracebacks=True)
console_handler.setLevel(logging.INFO)

# Combine both
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[file_handler, console_handler]
)

logger = logging.getLogger("pipeline")
