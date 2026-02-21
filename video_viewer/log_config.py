import logging
import sys

def setup_logging(level=logging.WARNING, log_file=None):
    """Configure logging for the video viewer application."""
    fmt = "%(asctime)s [%(name)s] %(levelname)s: %(message)s"
    handlers = [logging.StreamHandler(sys.stderr)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(level=level, format=fmt, handlers=handlers)
