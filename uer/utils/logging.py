import logging


def init_logger(args):
    log_format = logging.Formatter("[%(asctime)s %(levelname)s] %(message)s")
    logger = logging.getLogger()
    logger.setLevel(args.log_level)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logger.handlers = [console_handler]

    if args.log_path is not None:
        file_handler = logging.FileHandler(args.log_path, encoding="UTF-8")
        file_handler.setLevel(args.log_file_level)
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)

    return logger
