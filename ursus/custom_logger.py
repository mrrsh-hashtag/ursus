import logging
import config as cfg

def setup():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(name)-20.20s %(levelname)-8s: %(message)s')
    # formatter = logging.Formatter('%(asctime)s %(filename)s - %(funcName)-.20s %(levelname)-8s: %(message)s')

    if cfg.log_file: 
        file_handler = logging.FileHandler(cfg.log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    log = logging.getLogger(__name__)
