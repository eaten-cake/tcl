from setuptools import setup

import os

CURRENT_DIR = os.path.dirname(__file__)


def get_version():
    libinfo_py = os.path.join(CURRENT_DIR, "./tcl/libinfo.py")
    libinfo = {"__file__": libinfo_py}
    exec(compile(open(libinfo_py, "rb").read(), libinfo_py, "exec"), libinfo, libinfo)
    print(libinfo)
    return libinfo["__version__"]


__version__ = get_version()


setup(
    name="tcl",
    version=__version__,
    description="A Python package for TCL",
    author="Ruxiao Yin",
)