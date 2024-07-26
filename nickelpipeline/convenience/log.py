import pkg_resources
import json
import logging.config

def default_logger(module_name):

    # Load the JSON configuration from a file
    config = load_logging_config()
    config['handlers']['file']['filename'] = f"log_{module_name}.log"

    # Configure logging with the loaded configuration
    logging.config.dictConfig(config)
    # logger = logging.getLogger(module_name)
    # return logger

# Load the JSON configuration from a file within a package
def load_logging_config():
    with pkg_resources.resource_stream('nickelpipeline.convenience', 'logging_config.json') as f:
        config = json.load(f)
    return config

def change_log_file(logger, log_path):
    # Change the filename of the file handler
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            handler.baseFilename = log_path
            # Close the old stream and open a new one
            handler.close()
            handler.stream = open(handler.baseFilename, handler.mode, encoding=handler.encoding)

def adjust_global_logger(log_level='INFO', name='all_others'):

    # Load the JSON configuration from a file and adjust
    output_file = f"log_{__name__.split('.')[-1]}.log"
    with pkg_resources.resource_stream('nickelpipeline.convenience', 'logging_config.json') as f:
        config = json.load(f)
        config['handlers']['file']['filename'] = output_file
        config['handlers']['console']['level'] = log_level

    # Configure logging with the loaded configuration
    logging.config.dictConfig(config)

