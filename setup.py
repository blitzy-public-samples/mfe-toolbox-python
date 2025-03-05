#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
A minimal setup.py file that defers to pyproject.toml for configuration.
This file exists for backward compatibility with older Python packaging tools
that don't support pyproject.toml yet.
"""

import setuptools

if __name__ == "__main__":
    # Use setuptools.setup() with no arguments to defer to pyproject.toml
    # This approach provides compatibility with both modern and legacy Python packaging
    setuptools.setup()
