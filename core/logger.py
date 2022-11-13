from functools import wraps
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


def log(comment=None):
    """

    :param comment: LOG message before func exec
    :return: wrapped func
    """
    def log_comment(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                if comment:
                    logger.info(comment)
                res = func(*args, **kwargs)
                logger.info(f'Passed: {func.__name__.capitalize()}')
                # some shit right there
                # avoiding circular imports
                if str(res.__class__) in ("<class 'agent.AbstractAgent'>", "<class 'agent.LinkedAgent'>"):
                    logger.info(f'Agent file was read successfully: {res.name}')


                return res
            except BaseException as e:
                logger.exception(f"Some error occured: {e}")
                raise e

        return wrapper

    return log_comment
