#!/usr/bin/env python3
import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from playground.experiments import REGISTRY
from playground.experiments import apply, cka_compare


def build_parser():
    parser = argparse.ArgumentParser(
        prog="circuit_lab",
        description="Circuit experiment playground. Run: python playground/circuit_lab.py <command> --help",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    apply.add_args(sub)
    cka_compare.add_args(sub)
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    experiment = REGISTRY[args.command](args)
    experiment.run()
