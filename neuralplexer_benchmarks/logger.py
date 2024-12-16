from logging import Logger, LoggerAdapter, getLogger

DEFAULT_LOGGER_NAME = "neuralplexer"


def get_logger(
    name: str = DEFAULT_LOGGER_NAME, *, logger: Logger | LoggerAdapter | None = None
) -> Logger | LoggerAdapter:
    """Get a Logger based on input arguments.

    If a Logger object is passed in, return this object.
    otherwise:
        - If a Prefect run context exists, get the Prefect run logger
        - Else, get a standard Logger based on name.

    Args:
        name (str, optional): Name of the logger. Defaults to DEFAULT_LOGGER_NAME.
        logger (Optional[Logger], optional): Existing logger to be used. Defaults to None.

    Returns:
        Logger: a Logger object.
    """
    if logger is not None:
        return logger

    return getLogger(name)
