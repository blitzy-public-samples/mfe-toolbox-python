#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
A minimal setup.py file that defers to pyproject.toml for configuration.
This file provides backward compatibility with older Python packaging tools
while maintaining the modern pyproject.toml-based configuration as the primary
source of metadata.
"""

import setuptools

if __name__ == "__main__":
    # Defer to setuptools.setup() with pyproject.toml configuration
    setuptools.setup()
