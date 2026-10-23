"""Command line interface: ``mspdetect {check-config,run,metrics} config.yaml``."""

from __future__ import annotations

import argparse
import logging
import sys

import yaml
from pydantic import ValidationError

from mspdetect import __version__
from mspdetect.batch import run_detection, run_metrics
from mspdetect.config import BatchConfig


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mspdetect",
        description="Mid-sagittal plane detection in head and neck CTs.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("-v", "--verbose", action="store_true", help="show debug messages")
    commands = parser.add_subparsers(dest="command", required=True)
    for name, help_text in [
        ("check-config", "validate a configuration file and print the full settings"),
        ("run", "detect the plane of all cases"),
        ("metrics", "compute metrics for all cases with a detected plane"),
    ]:
        command = commands.add_parser(name, help=help_text)
        command.add_argument("config", help="YAML configuration file")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point of the ``mspdetect`` command."""
    args = _parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    try:
        config = BatchConfig.from_yaml(args.config)
    except (OSError, ValidationError, yaml.YAMLError) as e:
        print(f"Invalid configuration {args.config}:\n{e}", file=sys.stderr)  # noqa: T201
        return 2

    if args.command == "check-config":
        print(yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False))  # noqa: T201
        return 0
    run = run_detection if args.command == "run" else run_metrics
    _, n_failed = run(config)
    return 1 if n_failed else 0


if __name__ == "__main__":
    sys.exit(main())
