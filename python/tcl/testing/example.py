from . import _ffi_api

def hello():
    """A simple hello function."""
    return _ffi_api.HelloWorld()

