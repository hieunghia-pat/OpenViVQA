import atexit
import logging
from termcolor import colored
from datetime import datetime
import functools
import sys
import os

class ColorfulFormatter(logging.Formatter):
    def __init__(self, *args, **kwargs):
        super(ColorfulFormatter, self).__init__(*args, **kwargs)

    def formatMessage(self, record):
        created_time = datetime.fromtimestamp(record.created)
        asctime = created_time.strftime(self.datefmt)
        levelname = record.levelname
        message = record.message
        log = self._fmt % {"asctime": asctime, "levelname": levelname, "message": message}

        if record.levelno == logging.DEBUG:
            log = colored(log, "blue")
        elif record.levelno == logging.INFO:
            log = colored(log, "green")
        if record.levelno == logging.WARNING:
            log = colored(log, "yellow")
        elif record.levelno == logging.ERROR or record.levelno == logging.CRITICAL:
            log = colored(log, "red")
        
        return log

@functools.lru_cache()  # so that calling setup_logger multiple times won't add many handlers
def setup_logger(output=None, distributed_rank=0, *, color=True, name="OpenViVQA"):
    """
    Initialize the OpenViVQA logger and set its verbosity level to "DEBUG".

    Args:
        output (str): a file name or a directory to save log. If None, will not save log file.
            If ends with ".txt" or ".log", assumed to be a file name.
            Otherwise, logs will be saved to `output/log.txt`.
        name (str): the root module name of this logger

    Returns:
        logging.Logger: a logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    FORMAT = "[%(asctime)s] %(levelname)s: %(message)s"
    plain_formatter = logging.Formatter(FORMAT, datefmt="%d/%m/%Y %H:%M:%S")
    # stdout logging: master only
    if distributed_rank == 0:
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(logging.DEBUG)
        if color:
            formatter = ColorfulFormatter(fmt=FORMAT, datefmt="%d/%m/%Y %H:%M:%S")
        else:
            formatter = plain_formatter
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    # file logging: all workers
    if output is not None:
        if output.endswith(".txt") or output.endswith(".log"):
            filename = output
        else:
            filename = os.path.join(output, "log.txt")
        if distributed_rank > 0:
            filename = filename + ".rank{}".format(distributed_rank)
        if os.path.isdir(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))

        fh = logging.StreamHandler(_cached_log_stream(filename))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(plain_formatter)
        logger.addHandler(fh)

    return logger

# cache the opened file object, so that different calls to `setup_logger`
# with the same file name can safely write to the same file.
@functools.lru_cache(maxsize=None)
def _cached_log_stream(filename):
    # use 1K buffer if writing to cloud storage
    io = open(filename, "a", buffering=1024 if "://" in filename else -1)
    atexit.register(io.close)
    return io