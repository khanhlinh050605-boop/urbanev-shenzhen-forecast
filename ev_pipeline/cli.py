"""Command line entry point."""
import argparse
from pathlib import Path

from . import config
from .config import Settings


def parse_args(argv=None):
    defaults = Settings()
    parser = argparse.ArgumentParser(prog="ev_pipeline")
    parser.add_argument("--data", type=Path, default=defaults.data_path)
    parser.add_argument("--out", type=Path, default=defaults.output_path)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--reuse-saved-predictions", action="store_true")
    args = parser.parse_args(argv)
    return Settings(
        data_path=args.data,
        output_path=args.out,
        seed=args.seed,
        reuse_saved_predictions=args.reuse_saved_predictions,
    )


def main(argv=None):
    settings = parse_args(argv)
    config.set_seed(settings.seed)
    # Imported late on purpose: modules bind ``SEED`` when they are first imported.
    from .pipeline import run

    return run(settings)
