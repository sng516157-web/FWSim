"""Command-line entry point for the fwmav-sim console script.

Subcommands: ``run`` (experiment YAML), ``report`` (regenerate figures from a
stored run record), ``average-forces`` (Task 7: M1 record -> averaged-force
JSON), ``view`` (live MeshCat scene). Further subcommands (validate / compare
/ rerun) arrive with later milestones per the action plan.
"""

from __future__ import annotations

import argparse
import json
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

    p_live = sub.add_parser("live", help="live dynamics: integrate the free-body "
                                         "plant and stream to MeshCat (not a replay)")
    p_live.add_argument("experiment", type=Path)
    p_live.add_argument("--speed", type=float, default=0.1,
                        help="sim seconds per wall second (0.1 = 10x slowdown)")
    p_live.add_argument("--force-scale", type=float, default=5.0)
    p_live.add_argument("--no-record", action="store_true",
                        help="don't record the session (records save by default)")

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

    p_urdf = sub.add_parser("import-urdf",
                            help="import CAD-exported URDF mass properties into a "
                                 "vehicle YAML skeleton (DR-004 spike; wings manual)")
    p_urdf.add_argument("urdf", type=Path)
    p_urdf.add_argument("-o", "--out", type=Path, required=True)
    p_urdf.add_argument("--name", default=None)

    p_fit = sub.add_parser("fit-sysid",
                           help="fit aero coefficients to a calibration dataset (M6 T9)")
    p_fit.add_argument("experiment", type=Path,
                       help="fixed-body experiment YAML (vehicle + kinematics + environment)")
    p_fit.add_argument("dataset_dir", type=Path)
    p_fit.add_argument("--columns", nargs=3, required=True, metavar=("FX", "FY", "FZ"),
                       help="dataset force column names (world frame)")
    p_fit.add_argument("--free", nargs="+", default=None,
                       help="coefficients to fit (default: cl_0 cl_slope cd_0 cd_slope)")
    p_fit.add_argument("-o", "--out", type=Path, required=True,
                       help="output vehicle-fragment YAML (provenance: fitted)")

    p_gui = sub.add_parser("gui", help="launch the PySide6 workbench (requires fwmav-sim[gui])")
    p_gui.add_argument("--legacy", action="store_true",
                       help="launch the M5-M8 batch-oriented window instead of the "
                            "M9 live-first workbench")

    p_bench = sub.add_parser("benchmark",
                             help="time experiment runs (speed vs fidelity, M6.75)")
    p_bench.add_argument("experiments", nargs="*", type=Path, default=None,
                         help="experiment YAMLs (default: the standard suite)")
    p_bench.add_argument("--json", type=Path, default=None)

    p_abl = sub.add_parser("ablate",
                           help="on/off comparison per optional physics block (M6.75)")
    p_abl.add_argument("experiment", type=Path)
    p_abl.add_argument("--json", type=Path, default=None)

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

    if args.command == "live":
        from fwmav_sim.viz.live import serve_live_dynamics

        serve_live_dynamics(args.experiment, speed=args.speed,
                            force_scale=args.force_scale, record=not args.no_record)
        return 0

    if args.command == "gui":
        from fwmav_sim.ui.app import main as gui_main

        return gui_main(legacy=args.legacy)

    if args.command == "benchmark":
        from fwmav_sim.analysis.benchmark import format_table, run_benchmark

        rows = run_benchmark(args.experiments or None, out_json=args.json)
        print(format_table(rows))
        if args.json:
            print(f"benchmark record: {args.json}")
        return 0

    if args.command == "ablate":
        from fwmav_sim.analysis.ablation import format_ablation, run_ablation

        base, rows = run_ablation(args.experiment)
        print(format_ablation(str(args.experiment), base, rows))
        if args.json:
            args.json.write_text(json.dumps({"baseline": base, "rows": rows}, indent=2),
                                 encoding="utf-8")
            print(f"ablation record: {args.json}")
        return 0

    if args.command == "sensitivity":
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
                                     "step": e.step, "derivatives": e.derivatives,
                                     "sensitivities": e.sensitivities,
                                     "status": e.status} for e in entries]},
                       indent=2), encoding="utf-8")
        md = report_markdown(str(args.experiment), base, entries, metric)
        (out_dir / "report.md").write_text(md, encoding="utf-8")
        print(md)
        print(f"sensitivity record: {out_dir}")
        return 0

    if args.command == "import-urdf":
        import yaml

        from fwmav_sim.vehicle.urdf_import import import_urdf

        vehicle = import_urdf(args.urdf, name=args.name)
        header = (
            f"# Vehicle skeleton imported from URDF (DR-004 spike): {args.urdf.name}\n"
            "# Bodies/joints carry CAD mass properties (provenance: estimated).\n"
            "# Frame convention: URDF frames must already match this project\n"
            "# (+x forward, +z down body frame) — pre-rotate at CAD export if not.\n"
            "# TODO: author wings/aero manually; joints are serial-chain only.\n")
        args.out.write_text(
            header + yaml.safe_dump(vehicle.model_dump(mode="json"), sort_keys=False),
            encoding="utf-8")
        print(f"vehicle skeleton: {args.out}")
        return 0

    if args.command == "fit-sysid":
        from fwmav_sim.analysis.sysid import (
            DEFAULT_FREE,
            fit_coefficients,
            load_fit_dataset,
            write_fragment,
        )
        from fwmav_sim.environment.spec import environment_from_spec
        from fwmav_sim.experiment.config import load_experiment, load_vehicle

        exp = load_experiment(args.experiment)
        if exp.kind != "fixed_body":
            print("fit-sysid requires a fixed_body experiment")
            return 1
        vehicle = load_vehicle(args.experiment.parent / exp.vehicle)
        env = environment_from_spec(exp.environment)
        t, F, _manifest = load_fit_dataset(args.dataset_dir, tuple(args.columns))
        result = fit_coefficients(vehicle, exp.kinematics, env, t, F,
                                  free=tuple(args.free) if args.free else DEFAULT_FREE,
                                  dataset_ref=str(args.dataset_dir))
        out = write_fragment(result, str(args.dataset_dir), args.out)
        print(f"rms {result.rms_baseline:.4g} -> {result.rms_residual:.4g} "
              f"({result.n_samples} samples, free: {', '.join(result.free)})")
        print(f"fitted fragment: {out}")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
