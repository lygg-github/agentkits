import logging

logger = logging.getLogger("agentkits")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"),
    )
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)
