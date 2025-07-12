"""Load MLC LLM library and _ffi_api functions."""

import ctypes
import os
import sys

import tvm
import tvm.base

from . import libinfo

# version
__version__ = libinfo.__version__


def _load_lib():
    """Load lib"""
    if sys.platform.startswith("win32") and sys.version_info >= (3, 8):
        for path in libinfo.get_dll_directories():
            os.add_dll_directory(path)
    # pylint: disable=protected-access
    lib_name = "tcl" if tvm.base._RUNTIME_ONLY else "tcl_module"
    # pylint: enable=protected-access
    lib_path = libinfo.find_lib_path(lib_name, optional=False)
    return ctypes.CDLL(lib_path[0]), lib_path[0]


_LIB, _LIB_PATH = _load_lib()