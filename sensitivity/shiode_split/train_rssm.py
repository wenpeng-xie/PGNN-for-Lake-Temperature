#!/usr/bin/env python3
"""Shiode rssm: fixed seeds, 56/14/30, internal-val checkpoint selection."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
from engine import cli
if __name__ == '__main__':
    cli('rssm')
