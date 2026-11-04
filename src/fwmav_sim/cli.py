"""Command-line entry point for the fwmav-sim console script.

Subcommands: ``run`` (experiment YAML), ``report`` (regenerate figures from a
stored run record), ``average-forces`` (Task 7: M1 record -> averaged-force
JSON), ``view`` (live MeshCat scene). Further subcommands (validate / compare
/ rerun) arrive with later milestones per the action plan.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import fwmav_sim


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="fwmav-sim",
        description="FWMAV flight-dynamics and control workbench.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {fwmav_sim.__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="run an experiment YAML")
    p_run.add_argument("experiment", type=Path)
    p_run.add_argument("--out", type=Path, default=Path("outputs"))

    p_report = sub.add_parser("report", help="regenerate figures from a stored run record")
    p_report.add_argument("record_dir", type=Path)

    p_avg = sub.add_parser("average-forces",
                           help="compute cycle-averaged forces from a fixed-body run record")
    p_avg.add_argument("record_dir", type=Path)
    p_avg.add_argument("-o", "--out", type=Path, required=True)

    p_rerun = sub.add_parser("rerun", help="reproduce a stored run record and diff outputs")
    p_rerun.add_argument("record_dir", type=Path)
    p_rerun.add_argument("--out", type=Path, default=Path("outputs"))

    p_cmp = sub.add_parser("compare", help="compare a run record against a dataset (M4)")
    p_cmp.add_argument("record_dir", type=Path)
    p_cmp.add_argument("dataset_dir", type=Path)
    p_cmp.add_argument("--sim-channel", required=True,
                       help="simulated channel, e.g. wing_right/F_total")
    p_cmp.add_argument("--exp-column", default=None,
                       help="dataset column (default: sim channel with / -> _)")
    p_cmp.add_argument("--component", type=int, default=None,
                       help="vector component index for vector channels")
    p_cmp.add_argument("--filter-cutoff", type=float, default=None,
                       help="Hz; apply the same zero-phase 4th-order lowpass to both series")
    p_cmp.add_argument("--out", type=Path, default=Path("outputs"))

    p_view = sub.add_parser("view", help="live 3D MeshCat view of an experiment or run record")
    p_view.add_argument("source", type=Path, help="experiment YAML or run-record directory")
    p_view.add_argument("--speed", type=float, default=0.05,
                        help="playback speed fraction (0.05 = 20x slowdown)")
    p_view.add_argument("--force-scale", type=float, default=5.0,
                        help="force arrow display scale, metres per newton")

    p_sens = sub.add_parser("sensitivity",
                            help="one-at-a-time parameter sensitivity (post-M5 item 2)")
    p_sens.add_argument("experiment", type=Path)
    p_sens.add_argument("--param", action="append", default=None,
                        help="parameter path (repeatable); default: all scalar vehicle/"
                             "kinematics/environment parameters")
    p_sens.add_argument("--rel", type=float, default=0.05,
                        help="relative perturbation step (default 0.05 = 5 %%)")
    p_sens.add_argument("--metric", default=None,
                        help="metric for the ranked table (default: kind-specific)")
    p_sens.add_argument("--out", type=Path, default=Path("outputs"))

    sub.add_parser("gui", help="launch the PySide6 workbench (requires fwmav-sim[gui])")

    args = parser.parse_args()

    if args.command == "run":
        from fwmav_sim.simulation.run import run_experiment

        record_dir = run_experiment(args.experiment, args.out)
        print(f"run record: {record_dir}")
        return 0

    if args.command == "report":
        from fwmav_sim.experiment.record import load_record
        from fwmav_sim.viz.force_plots import make_report_figures
        from fwmav_sim.viz.state_plots import make_state_figures

        ts, _meta = load_record(args.record_dir)
        out_dir = Path(args.record_dir) / "figures"
        kind = ts.meta.get("kind")
        if kind == "fixed_body":
            paths = make_report_figures(ts, out_dir)
        else:
            paths = make_state_figures(ts, out_dir)
        for path in paths:
            print(f"figure: {path}")
        return 0

    if args.command == "average-forces":
        from fwmav_sim.dynamics.cycle_averaged import save_average_forces

        out = save_average_forces(args.record_dir, args.out)
        print(f"averaged forces: {out}")
        return 0

    if args.command == "rerun":
        import numpy as np

        from fwmav_sim.simulation.run import run_experiment

        config = Path(args.record_dir) / "config.yaml"
        new_dir = run_experiment(config, args.out)
        import fwmav_sim.experiment.record as rec

        ts_old, _ = rec.load_record(args.record_dir)
        ts_new, _ = rec.load_record(new_dir)
        if ts_old.channel_names() != ts_new.channel_names():
            print("rerun MISMATCH: channel sets differ")
            return 1
        max_diff = 0.0
        bitexact = True
        for name in ts_old.channel_names():
            a, b = ts_old[name], ts_new[name]
            if a.shape != b.shape:
                print(f"rerun MISMATCH: {name} shape {a.shape} vs {b.shape}")
                return 1
            if not np.array_equal(a, b):
                bitexact = False
                max_diff = max(max_diff, float(np.abs(a - b).max()))
        if bitexact:
            print(f"rerun: bit-identical outputs ({new_dir})")
        else:
            print(f"rerun: max abs diff {max_diff:.3e} ({new_dir})")
        return 0 if bitexact else 2

    if args.command == "compare":
        from fwmav_sim.analysis.compare import compare
        from fwmav_sim.datasets.filtering import FilterSpec

        fs = (FilterSpec(cutoff_hz=args.filter_cutoff,
                         rationale="CLI --filter-cutoff; applied to both series")
              if args.filter_cutoff else None)
        report_dir = compare(args.record_dir, args.dataset_dir, args.sim_channel,
                             args.exp_column, args.out, fs, component=args.component)
        print(f"comparison report: {report_dir}")
        return 0

    if args.command == "view":
        from fwmav_sim.viz.meshcat_scene import replay_record, serve_live

        if args.source.is_dir():
            replay_record(args.source, playback_speed=args.speed, force_scale=args.force_scale)
        else:
            serve_live(args.source, playback_speed=args.speed, force_scale=args.force_scale)
        return 0

    if args.command == "gui":
        from fwmav_sim.ui.app import main as gui_main

        return gui_main()

    if args.command == "sensitivity":
        import json

        from fwmav_sim.analysis.sensitivity import report_markdown, sensitivity_analysis

        base, entries = sensitivity_analysis(args.experiment, params=args.param,
                                             rel_step=args.rel)
        metric = args.metric or ("meanFz_total" if "meanFz_total" in base
                                 else next(iter(base)))
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = args.out / f"{stamp}_sensitivity_{args.experiment.stem}"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "sensitivity.json").write_text(
            json.dumps({"experiment": str(args.experiment), "rel_step": args.rel,
                        "baseline": base,
                        "entries": [{"path": e.path, "base_value": e.base_value,
                                     "step": e.step, "sensitivities": e.sensitivities,
                                     "status": e.status} for e in entries]},
                       indent=2), encoding="utf-8")
        md = report_markdown(str(args.experiment), base, entries, metric)
        (out_dir / "report.md").write_text(md, encoding="utf-8")
        print(md)
        print(f"sensitivity record: {out_dir}")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
