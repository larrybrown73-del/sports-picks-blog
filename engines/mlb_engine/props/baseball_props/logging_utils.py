from __future__ import annotations

import logging

_logged_once_keys: set[str] = set()


def log_once(
    key: str,
    logger: logging.Logger,
    level: int,
    message: str,
    *args: object,
) -> bool:
    """Emit a log message at most once per process for the given key."""
    if key in _logged_once_keys:
        return False
    _logged_once_keys.add(key)
    logger.log(level, message, *args)
    return True


def reset_log_once() -> None:
    """Clear once-per-run log keys (for tests)."""
    _logged_once_keys.clear()


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a configured module logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False
    return logger
