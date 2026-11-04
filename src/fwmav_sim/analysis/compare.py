"""Comparison pipeline (M4 Task 7): fwmav-sim compare <record> <dataset>.

Steps: load run record + dataset manifest/CSV; synchronize wingbeat phase
(offset/drift reported, not silently corrected); filter both series with the
same spec when the dataset declares one; compute metrics; write a report
directory with metrics.json, figures, alignment results, filter specs,
dataset labels and provenance, and the simulator git hash.

validation_status rules (brief §10 M4, §14):
  - synthetic dataset  -> "SYNTHETIC_PROOF" (pipeline demonstration only)
  - real dataset       -> "COMPARISON_COMPLETE" (never "validated" without a
                          defined metric criterion approved per dataset)
  - no real dataset    -> the Leng-comparison tests stay skipped: "PENDING"
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from fwmav_sim.analysis.metrics import mae, nrmse, pearson, phase_lag, rmse, welch_spectra
from fwmav_sim.datasets.alignment import dominant_frequency, estimate_offset, resample_to_phase
from fwmav_sim.datasets.filtering import FilterSpec, apply_filter
from fwmav_sim.datasets.importer import load_dataset
from fwmav_sim.experiment.record import load_record


def compare(
    record_dir: str | Path,
    dataset_dir: str | Path,
    sim_channel: str,
    exp_column: str | None = None,
    out_root: str | Path = "outputs",
    filter_spec: FilterSpec | None = None,
    component: int | None = None,
) -> Path:
    """Run the comparison and write the report directory. Returns its path."""
    record_dir = Path(record_dir)
    dataset_dir = Path(dataset_dir)
    ts, meta = load_record(record_dir)
    ds = load_dataset(dataset_dir)
    exp_column = exp_column or sim_channel.replace("/", "_")

    df = ds.to_dataframe()
    t_exp = df.index.to_numpy(float)
    x_exp = df[exp_column].to_numpy(float)
    x_sim = ts[sim_channel]
    if x_sim.ndim > 1:
        if component is None:
            raise ValueError(
                f"sim channel {sim_channel!r} is vector-valued; pass --component")
        x_sim = x_sim[:, component]
    t_sim = ts.t

    # Wingbeat frequency: trust the sim record's metadata when present (the
    # wingbeat is the flapping period, not whichever force harmonic is
    # strongest); fall back to FFT. The dataset's dominant frequency is
    # reported as a consistency check.
    f_meta = ts.meta.get("flap_frequency_hz")
    f_sim_fft = dominant_frequency(t_sim, x_sim)
    f_flap = float(f_meta) if f_meta else f_sim_fft
    f_exp_fft = dominant_frequency(t_exp, x_exp)

    # consistent filtering (same spec on both, brief §5.9 item 3)
    applied_filter = None
    if filter_spec is not None:
        x_sim = apply_filter(t_sim, x_sim, filter_spec)
        x_exp = apply_filter(t_exp, x_exp, filter_spec)
        applied_filter = filter_spec.model_dump()

    period = 1.0 / f_flap
    phase_off, drift = estimate_offset(t_sim, x_sim, t_exp, x_exp, f_flap)

    # phase-domain comparison (phase offset removed for the overlay/metrics);
    # bin count adapted to the coarsest sampling so no bin starves
    samples_per_period = min(1.0 / (f_flap * float(np.median(np.diff(t_sim)))),
                             1.0 / (f_flap * float(np.median(np.diff(t_exp)))))
    n_bins = max(10, min(200, int(samples_per_period / 2)))
    # shared absolute phase anchor (the sim clock zero): without this, each
    # series anchors phase at its own first sample and the profiles rotate by
    # the series-start difference
    centres, sim_prof = resample_to_phase(t_sim, x_sim, period, n_bins, t_ref=0.0)
    _, exp_prof = resample_to_phase(t_exp + phase_off, x_exp, period, n_bins, t_ref=0.0)
    valid = ~(np.isnan(sim_prof) | np.isnan(exp_prof))
    if valid.sum() < 4:
        raise ValueError("too few overlapping phase bins; increase sampling or duration")

    # time-domain residual on a common grid within the overlap
    dt = 1.0 / (20.0 * f_flap)
    t0 = max(t_sim[0], t_exp[0] + phase_off)
    t1 = min(t_sim[-1], t_exp[-1] + phase_off)
    grid = np.arange(t0, t1, dt)
    s_g = np.interp(grid, t_sim, x_sim)
    e_g = np.interp(grid, t_exp + phase_off, x_exp)

    metrics = {
        "rmse": rmse(sim_prof[valid], exp_prof[valid]),
        "nrmse": nrmse(sim_prof[valid], exp_prof[valid]),
        "mae": mae(sim_prof[valid], exp_prof[valid]),
        "pearson": pearson(sim_prof[valid], exp_prof[valid]),
        "phase_lag_rad_at_dominant_harmonic": phase_lag(grid, s_g, e_g, f_flap),
        "phase_bins_compared": int(valid.sum()),
    }
    freqs, psd_s, psd_e = welch_spectra(grid, s_g, e_g)

    from fwmav_sim.viz.comparison import make_comparison_figures

    report_dir = Path(out_root) / f"comparison_{record_dir.name}_vs_{dataset_dir.name}"
    figures = make_comparison_figures(
        centres, sim_prof, exp_prof, grid, s_g, e_g, freqs, psd_s, psd_e,
        channel=sim_channel, out_dir=report_dir,
    )

    status = "SYNTHETIC_PROOF" if ds.synthetic else "COMPARISON_COMPLETE"
    report = {
        "sim_record": str(record_dir),
        "dataset": str(dataset_dir),
        "dataset_label": ds.label,
        "dataset_synthetic": ds.synthetic,
        "dataset_source": ds.source,
        "sim_channel": sim_channel,
        "exp_column": exp_column,
        "wingbeat_frequency_hz": f_flap,
        "sim_fft_dominant_hz": f_sim_fft,
        "exp_fft_dominant_hz": f_exp_fft,
        "alignment": {
            "phase_offset_s": phase_off,
            "phase_offset_fraction_of_wingbeat": phase_off * f_flap,
            "drift_s_per_s": drift,
            "note": "integer-wingbeat ambiguity is inherent; offset is modulo one wingbeat",
        },
        "filter_applied": applied_filter,
        "metrics": metrics,
        "figures": [str(p) for p in figures],
        "validation_status": status,
        "validation_note": (
            "No validation badge without a real dataset and a pre-defined metric "
            "criterion (brief §10 M4). SYNTHETIC_PROOF means pipeline-only."
        ),
        "sim_git_hash": meta.get("git_hash"),
    }
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "comparison_report.json").write_text(json.dumps(report, indent=2))
    return report_dir
