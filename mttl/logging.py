import logging
import os

logger = logging.getLogger("mttl")


def setup_logging(log_dir: str = None):
    logging.basicConfig(
        format="%(asctime)s %(levelname)s --> %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.setLevel(logging.INFO)
    logging.getLogger("openai").setLevel(logging.WARNING)

    if log_dir:
        log_file_path = os.path.join(log_dir, "log.txt")
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        handler_exists = any(
            isinstance(handler, logging.FileHandler)
            and handler.baseFilename == log_file_path
            for handler in logger.handlers
        )

        if not handler_exists:
            logger.addHandler(logging.FileHandler(log_file_path))
            logger.info(
                "New experiment, log will be at %s",
                log_file_path,
            )
