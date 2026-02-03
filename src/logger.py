import logging
import os
import sys


def setup_logging(run_dir: str, log_file: str = "log.txt"):
    os.makedirs(run_dir, exist_ok=True)

    log_path = os.path.join(run_dir, log_file)
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    root_logger = logging.getLogger()

    if not root_logger.handlers:
        root_logger.setLevel(logging.INFO)

        file_handler = logging.FileHandler(log_path, mode="a")
        file_formatter = logging.Formatter(
            "%(asctime)s - [%(name)s] - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(file_formatter)

        console_handler = logging.StreamHandler(sys.stdout)
        console_formatter = logging.Formatter("[%(name)s] %(message)s")
        console_handler.setFormatter(console_formatter)

        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)

    logging.info(f"Run directory: {run_dir}")
    logging.info(f"Checkpoints directory: {ckpt_dir}")

    return ckpt_dir


def log_to_file(name, value):
    logger = logging.getLogger(name)
    logger.info(value)
