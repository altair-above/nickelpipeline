import pkg_resources
import json
import logging.config


def default_logger(module_name):

    # Load the JSON configuration from a file
    config = load_logging_config()
    config['handlers']['file']['filename'] = f"log_{module_name}.log"

    # Configure logging with the loaded configuration
    logging.config.dictConfig(config)
    logger = logging.getLogger(module_name)
    return logger

# Load the JSON configuration from a file within a package
def load_logging_config():
    with pkg_resources.resource_stream('nickelpipeline.convenience', 'logging_config.json') as f:
        config = json.load(f)
    return config