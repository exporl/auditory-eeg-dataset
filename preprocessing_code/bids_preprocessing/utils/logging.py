import logging


class DefaultFormatter(logging.Formatter):

    default_format = "%(asctime)s | %(process)-3d | %(levelname)-8s | %(message)s"
    def __init__(self, *args, fmt=default_format, **kwargs):
        super().__init__(*args, fmt=fmt, **kwargs)

class ColorFormatter(DefaultFormatter):
    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"

    COLOR_MAPPING = {
        logging.DEBUG: grey,
        logging.INFO: grey,
        logging.WARNING: yellow,
        logging.ERROR: red,
        logging.CRITICAL: bold_red,
    }

    def format(self, record):
        output = super(ColorFormatter, self).format(record)
        return self.COLOR_MAPPING[record.levelno] + output + self.reset


def default_logging(**kwargs):
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(ColorFormatter())
    extra_handlers = kwargs.get('handlers', [])
    if 'handlers' in kwargs:
        del kwargs['handlers']
    return logging.basicConfig(level=logging.INFO, handlers=[ch] + extra_handlers, **kwargs)
