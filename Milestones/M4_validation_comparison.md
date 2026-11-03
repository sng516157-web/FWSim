# Milestone 4 — Validation and Experiment Comparison

> **Stage prompt for Kimi Code.** Read first:
> - `../FWMAV_Simulator_Project_Brief.md` — especially §5.8, §5.9, §10 M4, §11.4, §12
> - `../FWMAV_Milestone_Action_Plan.md` — stack §1, conventions §3
> - M3's `experiment/record.py` — comparison consumes run records, not ad-hoc arrays
>
> Conflicts: the brief wins; stop and flag. Route A: pure Python only.

**Goal:** native experimental-comparison workflow — import, synchronize, filter, compare, report — with honest calibration/validation labelling (brief §10 M4).
**Estimated effort:** 1.5–2 weeks.
**Prerequisite:** M3 accepted (reproducible run records exist).

**Blocked-input risk (state it, don't hide it):** Leng, Ng & Tang (IMAV 2024) original time series may be unavailable (brief §12). If so, this milestone ships the complete pipeline proven on a synthetic dataset, and every data-dependent validation item stays explicitly `PENDING`. That is the brief-compliant outcome — not a failure to conceal.

---

## 4.1 Ground rules for this stage

- Simulation and experiment must represent **equivalent measured quantities** (brief §5.9): a load cell measures resultant force including inertial content — never compare it against aerodynamic force alone.
- Calibration and validation datasets are **separate and schema-enforced** (brief §3.5, §5.8). A dataset used to fit parameters can never afterwards carry the `validation` label (brief §11.4).
- No validation badge or claim without a real dataset and a defined comparison metric (brief §10 M4). Record failed comparisons honestly.
- Any filtering is documented and applied consistently to both series where appropriate (brief §5.9 item 3).

## 4.2 Locked stack for this stage

Adds **pandas ≥ 2.2** (import/resampling) and `scipy.signal` (filtering, spectra, correlation). Nothing else.

---

## 4.3 Tasks (in order)

### Task 1 — `datasets/importer.py`

pandas-based CSV importer. Import manifest (YAML next to each dataset) declares: per-column quantity + unit + frame, sampling rate, dataset label (`calibration | validation`), source/provenance, uncertainty where available. Importer converts to SI at the boundary and records the conversion. Rejects unlabeled or unit-less columns.

```python
class ExperimentalDataset(BaseModel):
    label: Literal["calibration", "validation"]
    quantities: dict[str, QuantitySpec]   # unit, frame, uncertainty
    source: str
    def to_dataframe(self) -> pd.DataFrame: ...   # SI, indexed by time
```

### Task 2 — `datasets/alignment.py`

Wingbeat-phase synchronization: estimate dominant flapping harmonic (FFT), cross-correlate simulated vs experimental reference channel, report time offset and drift; resample both onto a common phase grid. Offset and residual drift are outputs in the report, not silent corrections.

### Task 3 — Filtering (`datasets/filtering.py`)

Zero-phase `scipy.signal.filtfilt` with a `FilterSpec` (type, order, cutoff) stored in both the dataset metadata and the comparison report. The same spec applies to simulated and experimental series where the comparison requires it; the report states exactly what was filtered and why.

### Task 4 — `analysis/metrics.py`

RMSE, NRMSE, MAE, Pearson correlation, phase lag at dominant harmonic, Welch spectra comparison. Each metric function hand-checkable and unit-tested.

### Task 5 — `viz/comparison.py`

Overlay plots (simulated vs experimental), residual time series, spectra. Where relevant, figures **distinguish aerodynamic, inertial and measured-resultant forces** (brief §10 M4 acceptance) — reuse M3's per-term force outputs to reconstruct the simulated load-cell-equivalent quantity.

### Task 6 — Synthetic sanity dataset

`datasets/synthetic/`: generate a known-truth "experiment" from an M3 run record (downsampled, known phase offset, known added noise), labelled `synthetic: true`, `label: calibration`. Proves the pipeline end-to-end; must never be presented as validation.

### Task 7 — Comparison report generator

`fwmav-sim compare <run_record_dir> <dataset_dir>` → report directory: metrics table, overlay/residual/spectra figures, alignment results, filter specs, dataset labels and provenance, simulator git hash. A `validation_status` field is `PENDING` unless a real dataset + defined metric exist (brief §10 M4).

---

## 4.4 Test inventory

**Unit (§11.1):** importer unit conversion and label enforcement; alignment recovers a known synthetic phase offset to `< 0.1%` of a wingbeat; each metric vs hand-computed value.

**Numerical (§11.3):** synthetic round trip — residuals within the known injected noise level; filtered-then-compared synthetic case recovers known truth.

**Scientific validation (§11.4):** the Leng et al. comparison exists as `tests/validation/test_leng2024_suspended.py` marked `pytest.mark.skip(reason="awaiting original parameter values and time series")` until the data arrive; mechanism-ablation comparison hooks wired to M3's isolation runs.

## 4.5 Acceptance commands

```bash
pytest tests/ -q          # includes the skipped validation test, listed with reason
fwmav-sim compare outputs/<m3_run> datasets/synthetic/hover_case/
```

## 4.6 Milestone report

`docs/milestone_reports/M4.md`: pipeline demonstration on the synthetic dataset with full metrics; alignment/filtering metadata example; the calibration-vs-validation labelling demonstration; and an explicit status block — either real-data comparison results (with metrics and failures recorded honestly) or `PENDING: awaiting original experimental data` with exactly what is needed to unblock (parameter table, suspended-string time series, sampling rates, units).

## 4.7 Explicit non-goals for this stage

Parameter fitting/system identification workflows (infrastructure records provenance now; fitting comes later); free-flight validation claims; automated "validation passed" badges; UI.
