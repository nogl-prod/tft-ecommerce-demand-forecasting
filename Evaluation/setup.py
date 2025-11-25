#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Filename : _metrics.py
# Author : Tuhin Mallick

from pathlib import Path

from setuptools import find_packages, setup

# Package meta-data.
NAME = 'Evaluation'
DESCRIPTION = "Result evaluation package for NOGL.ai."
URL = "https://nogl.ai"
EMAIL = "tuhin.mallick@nogl.ai"
AUTHOR = "Tuhin Mallick"
REQUIRES_PYTHON = ">=3.6.0"


# packages required for this module to be executed
def list_reqs(fname="requirements.txt"):
    with open(fname) as fd:
        return fd.read().splitlines()


long_description = DESCRIPTION

# Load the package's VERSION file as a dictionary.
about = {}
ROOT_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = ROOT_DIR / NAME
with open(ROOT_DIR / "VERSION") as f:
    _version = f.read().strip()
    about["__version__"] = _version


setup(
    name=NAME,
    version=about["__version__"],
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(exclude=("tests",)),
    package_data={NAME: [_version]},
    install_requires=list_reqs(),
    extras_require={},
    include_package_data=True,
    zip_safe=False,
    license="",
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: Implementation :: PyPy",
    ],
)