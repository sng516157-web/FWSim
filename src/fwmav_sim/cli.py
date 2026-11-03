"""Command-line entry point for the fwmav-sim console script.

Milestone 0 provides the package skeleton only; subcommands (run / validate /
report / compare / rerun) are added by later milestones per the action plan.
"""

from __future__ import annotations

import argparse

import fwmav_sim


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="fwmav-sim",
        description="FWMAV flight-dynamics and control workbench.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {fwmav_sim.__version__}")
    parser.parse_args()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
