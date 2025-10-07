import logging

def worker_init_fn(worker_id):
    """
    Re-initializes the logger for each worker process.
    This is essential for seeing log messages from multiprocessing workers in a notebook.
    """
    # Get the root logger
    logger = logging.getLogger()
    
    # Remove any existing handlers
    logger.handlers = []
    
    # Add a new stream handler to output to the console
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '[%(asctime)s] [%(levelname)s] [Worker ID: %(process)d]: %(message)s'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    # Set the logging level
    logger.setLevel(logging.INFO)