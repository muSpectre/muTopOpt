#!/usr/bin/env python3
#
# Copyright 2026 Lars Pastewka
#
# MIT License (see LICENSE)
#
"""
Elasticity driver of muTopOpt, kept at the repository root so that
``python simulate.py`` keeps working. The driver itself lives in
``muTopOpt.simulate`` and is installed as the ``mutopopt`` command.
"""

from muTopOpt.simulate import main

if __name__ == "__main__":
    main()
