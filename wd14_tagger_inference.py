#!/usr/bin/env python3
"""Stable CLI entry point for the SwarmUI WD14Tagger extension."""

import os
import sys


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)


def main() -> None:
    from wd14tagger.cli import main as cli_main
    cli_main()


if __name__ == "__main__":
    main()
