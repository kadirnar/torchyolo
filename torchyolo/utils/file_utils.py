def create_dir(_dir) -> str:
    """
    Create directory if it doesn't exist
    Args:
        _dir: str
    """
    import os

    if not os.path.exists(_dir):
        os.makedirs(_dir)

    return _dir
