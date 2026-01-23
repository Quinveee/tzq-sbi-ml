"""
Local launcher.
"""


def launch(*, worker, cfg):
    """
    Docstring for launch

    :param worker: Partially instantiated worker function
    :param cfg: Parsed configuration object to be passed to the worker
    """
    worker(cfg=cfg)
