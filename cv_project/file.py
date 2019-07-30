import os

def relpath(path):
    """Returns the relative path to the script's location
    Arguments:
    path -- a string representation of a path.
    """
    return os.path.join(os.path.dirname(__file__), path)
