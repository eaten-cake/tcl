# top-level package for tcl
from .base import __version__

from .jit.base import jit
from .jit.tensor import Tensor

from . import testing