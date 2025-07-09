#!/usr/bin/env python3
"""
Console script entry point for WakeyWakey CLI.
This allows the package to be called as 'wakeywakey' from the command line.
"""

from .main import main

def cli_entry_point():
    """Entry point for the wakeywakey console command."""
    main()

if __name__ == '__main__':
    cli_entry_point() 