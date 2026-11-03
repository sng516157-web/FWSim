"""Command-line entry point for the fwmav-sim console script.

M1 subcommands: ``run`` (fixed-body experiment) and ``report`` (regenerate
figures from a stored run record). Further subcommands (validate / compare /
rerun) arrive with later milestones per the action plan.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import fwmav_sim


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="fwmav-sim",
        description="FWMAV flight-dynamics and control workbench.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {fwmav_sim.__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="run an experiment YAML (M1: fixed-body)")
    p_run.add_argument("experiment", type=Path)
    p_run.add_argument("--out", type=Path, default=Path("outputs"))

    p_report = sub.add_parser("report", help="regenerate figures from a stored run record")
    p_report.add_argument("record_dir", type=Path)

    p_view = sub.add_parser("view", help="live 3D MeshCat view of an experiment or run record")
    p_view.add_argument("source", type=Path, help="experiment YAML or run-record directory")
    p_view.add_argument("--speed", type=float, default=0.05,
                        help="playback speed fraction (0.05 = 20x slowdown)")

    args = parser.parse_args()

    if args.command == "run":
        from fwmav_sim.simulation.run import run_experiment

        record_dir = run_experiment(args.experiment, args.out)
        print(f"run record: {record_dir}")
        return 0

    if args.command == "report":
        from fwmav_sim.experiment.record import load_record
        from fwmav_sim.viz.force_plots import make_report_figures

        ts, _meta = load_record(args.record_dir)
        out_dir = Path(args.record_dir) / "figures"
        for path in make_report_figures(ts, out_dir):
            print(f"figure: {path}")
        return 0

    if args.command == "view":
        from fwmav_sim.viz.meshcat_scene import replay_record, serve_live

        if args.source.is_dir():
            replay_record(args.source, playback_speed=args.speed)
        else:
            serve_live(args.source, playback_speed=args.speed)
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
