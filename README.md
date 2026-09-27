# WIRE-FL

**WIRE-FL** is a federated-learning framework for wearable sensing in which every client upload is a serialized,
byte-capped packet that the server can screen and audit. It trains a small NumPy classifier across subjects (each
subject is a client) and evaluates it with strict leave-one-subject-out protocols on two wearable tasks:

- **SLEEP** – wake vs. sleep from wrist accelerometry, heart rate and steps (wake is the minority class);
- **WESAD** – stress vs. non-stress from wrist and chest physiological signals.

Everything is pure NumPy / SciPy / scikit-learn (metrics only); there is no deep-learning framework and no GPU
requirement.

## What is in the method

| Stage | Component | Where |
|---|---|---|
| Features | Sliding-window signal features, per-subject temporal-context features, subject-relative scaling, causal elapsed-time features, declared and validated look-ahead | `wire_fl/features.py` |
| Model | Linear skip connection + one-hidden-layer tanh residual branch (default 48 hidden units), exposed as one flat parameter vector split into named blocks | `wire_fl/model.py` |
| Local training | Proximal, Adam-style local steps with gradient masking to the client's available inputs and norm-clipped updates | `wire_fl/model.py` |
| Uplink codec | Serialized `WV20` packets under a **hard byte cap**: block-wise rate allocation, top-magnitude sparse packets and dense packets at every bit depth are all built and the smallest-error packet that fits is sent; full-alphabet MSE-optimal quantiser; strict decoder that fails closed | `wire_fl/codec.py` |
| Error feedback | Client-side residual memory with explicit rollback, committed only for updates that were actually applied | `wire_fl/codec.py`, `wire_fl/federated.py` |
| Fleet budget | Per-round byte allocation across heterogeneous devices under per-device caps and a common floor | `wire_fl/codec.py` |
| Server audit | Each client sends a CountSketch of what it transmitted; the server checks it against the decoded payload per client, with a tolerance that depends only on server-known constants | `wire_fl/security.py` |
| Server screening | Leave-one-out geometric-median consensus over sketches flags reversed or oversized updates and down-weights them | `wire_fl/security.py` |
| Aggregation | Participant-balanced weights, per-coordinate support normalisation, server momentum | `wire_fl/security.py`, `wire_fl/federated.py` |
| Decision layer | Participant-balanced logistic calibration, threshold selection, two-state sequence decoder (blend strength tuned on validation subjects) | `wire_fl/metrics.py`, `wire_fl/decoding.py` |
| Personalisation | Shrunk, capped personal bias fitted on a short labelled prefix and applied only if a leave-one-out check on that prefix says it helps | `wire_fl/personalization.py` |
| Evaluation | Leave-one-subject-out splits with disjoint fit / validation / calibration / test participants and leakage audits | `wire_fl/preprocessing.py`, `wire_fl/pipeline.py` |

For every held-out subject the remaining subjects are split (by a seeded permutation, never by outcomes) into
**fit** subjects (federated training), **validation** subjects (learning rate, decision threshold, decoder strength),
and **calibration** subjects (probability calibration and transition matrix). The held-out subject touches none of
these steps.

## Installation

Python 3.10 or newer (developed and tested on 3.11).

```bash
git clone https://github.com/<your-account>/WIRE-FL.git
cd WIRE-FL
python -m venv .venv && source .venv/bin/activate      # optional
pip install -e .                                        # or: pip install -r requirements.txt
```

Development tools (tests, formatting): `pip install -e ".[dev]"`.

## Data

The datasets are not redistributed. Download them from their original sources and arrange them as follows.

**WESAD** (wrist and chest wearable stress dataset) – the original per-subject pickle files:

```text
data/WESAD/S2/S2.pkl
data/WESAD/S3/S3.pkl
...
data/WESAD/S17/S17.pkl
```

The default subject list is S2–S11 and S13–S17 (15 subjects). Override it with `--wesad-subjects`.

**Sleep** – the PhysioNet dataset *Motion and heart rate from a wrist-worn wearable and labeled sleep from
polysomnography*:

```text
data/sleep/heart_rate/<id>_heartrate.txt
data/sleep/labels/<id>_labeled_sleep.txt
data/sleep/motion/<id>_acceleration.txt
data/sleep/steps/<id>_steps.txt
```

Only load trusted local files: the WESAD files are Python pickles. Extracted features are cached in
`wire_fl_cache/` under a key that includes the source-file signatures, the configuration and a hash of the package
source, so a stale cache is never reused silently.

## Quick start

Software check on a small built-in synthetic fixture (seconds, no data needed):

```bash
python -m wire_fl run --mode smoke --out results/smoke
```

Real data (paths are examples):

```bash
python -m wire_fl run --mode pilot \
    --wesad-dir data/WESAD --sleep-dir data/sleep --out results/pilot     # 8 held-out subjects per dataset, 1 seed
python -m wire_fl run --mode study \
    --wesad-dir data/WESAD --sleep-dir data/sleep --out results/study     # every subject held out once, 3 seeds
```

Presets:

| Preset | Rounds | Held-out subjects per dataset | Seeds | Use |
|---|---|---|---|---|
| `smoke` | 3 | 1 (synthetic data) | 1 | software check |
| `pilot` | 50 | 8 | 1 | first real-data run |
| `study` | 100 | all | 3 | full leave-one-subject-out run |

Useful options: `--datasets SLEEP WESAD`, `--seeds 13 42`, `--rounds N`, `--byte-budget BYTES`, `--max-folds N`,
`--hidden N`, `--no-sequence-decode`, `--context-time-mode causal|retrospective`, `--sensor-profile all|wrist`,
`--wesad-subjects S2 S3 ...`, `--cache-dir DIR`. Run `python -m wire_fl run --help` for the full list.

A run resumes automatically: finished runs are cached under `<out>/runs/` and interrupted federated training resumes
from its checkpoint.

### Splitting a study across processes

```bash
python -m wire_fl run --mode study --shard 0/4 --wesad-dir ... --sleep-dir ... --out results/study
python -m wire_fl run --mode study --shard 1/4 --wesad-dir ... --sleep-dir ... --out results/study
# ... shards 2/4 and 3/4 ...
python -m wire_fl merge --out results/study --shards 4      # merges CSVs and writes the summary
```

Sharding changes only which held-out subjects each process handles, never how a run is computed. `merge` refuses to
merge an incomplete set of shards.

### Robustness scenarios

To exercise the screening and audit, malicious clients can be simulated:

```bash
python -m wire_fl run --mode pilot --attack sign_flip --bad-fraction 0.2 --out results/sign_flip_0.2 ...
```

Supported behaviours: `sign_flip`, `scaled`, `gaussian`, `label_flip`, `sketch_forgery`, `free_rider`, `nullspace`.

## Outputs

Each run writes to `--out`:

| File | Content |
|---|---|
| `RESULTS_SUMMARY.md` | Self-describing summary: run status, configuration, participant-level results, per-subject table, communication/protocol diagnostics, personalisation, metric glossary, file index |
| `participant_metrics.csv` | One row per held-out subject and seed with every metric and run identifiers |
| `system_metrics.csv` | Uplink/downlink bytes, distortion, timing, screening and audit diagnostics per run |
| `personalization.csv` | Shared vs. personalised metrics per held-out subject |
| `completion_ledger.csv` | Status of every requested run (completed / failed) |
| `summary_by_dataset.csv`, `per_subject_results.csv`, `communication_summary.csv`, `personalization_summary.csv` | Tables used in the summary |
| `run_manifest.json`, `run_status.json`, `data_provenance.json`, `data_summary.csv` | Configuration, splits, environment, data fingerprints |
| `runs/<key>/` | Per-run `artifact.pkl` (deployable bundle), `predictions.csv`, `rounds.csv`, `client_events.csv`, `result.json`, `manifest.json` |

The statistical unit is the held-out participant: seeds are averaged within a participant before any summary is
computed, and bootstrap intervals are reported only when at least `min_ci_subjects` (6) participants were evaluated.
`RESULTS_SUMMARY.md` states whether the run is complete; a run with failed jobs must not be reported as a full result.

## Using a trained artifact

```python
from wire_fl import WIREPredictor

predictor = WIREPredictor.load("results/pilot/runs/<key>/artifact.pkl")
proba = predictor.predict_proba(frame)   # calibrated, sequence-decoded probability of class 1
label = predictor.predict(frame)         # decision at the validation-selected threshold
```

`frame` is a feature frame as produced by `wire_fl.features.load_data` (it must contain `client_id` and
`label_start`, which the sequence decoder uses to smooth each subject's own recording in time). Artifacts are Python
pickles; load only artifacts you created yourself.

A complete, runnable example on the synthetic fixture is in `examples/quickstart.py`.

## Python API

```python
from dataclasses import replace
from wire_fl import get_config, run_loso, Scenario
from wire_fl.config import data_config
from wire_fl.features import load_data

cfg = replace(get_config("pilot"), byte_budget=4000)
data_cfg = data_config("data/WESAD", "data/sleep")
frame, provenance = load_data(data_cfg)
metrics, systems, personal, ledger, manifest = run_loso(frame, provenance, cfg, "results/custom", Scenario(), data_config=data_cfg)
```

## Repository layout

```text
wire_fl/
  config.py           data configuration, hyper-parameters, presets
  features.py         WESAD / sleep feature extraction, context features, data validation
  preprocessing.py    LOSO splits, leakage audits, fit-only preprocessing, weights
  model.py            residual NumPy model and local training
  codec.py            byte-capped packet codec, error feedback, fleet allocation
  security.py         CountSketch, screening, per-client audit, aggregation, attack simulation
  federated.py        federated training loop, protocol and scenario definitions
  metrics.py          metrics, calibration, threshold selection, bootstrap
  decoding.py         two-state sequence decoder
  personalization.py  gated shrunk-bias personalisation
  pipeline.py         per-subject pipeline, LOSO driver, sharding
  reporting.py        summary tables and RESULTS_SUMMARY.md
  inference.py        WIREPredictor
  cli.py              command line interface
tests/                pytest suite
examples/             runnable examples
```

## Testing

```bash
pip install -e ".[dev]"
pytest
```

The suite covers codec round-trips and hard byte caps, error-feedback telescoping and rollback, allocation
constraints, split/leakage audits, look-ahead validation of the context features, sequence-decoder behaviour,
gradient correctness, sketch linearity, per-client audit, screening, support-normalised aggregation, checkpoint
resume, artifact/prediction consistency and a full leave-one-subject-out run on a tiny dataset written in the
original file formats.

## Reproducibility

Every random stream is derived from a named hash (`wire_fl.utils.rng_for`), so results depend only on the
configuration, the data and the seed - not on call order. Runs are single-threaded by default (`n_threads=1`). The
environment (Python, NumPy, pandas, SciPy, scikit-learn versions) and a hash of the package source are recorded in
`run_manifest.json`.

## Security boundary and limitations

- WIRE-FL is a numerical simulation of a federated protocol. Byte counts are application-level bytes of the
  serialized packets; they exclude network framing and transport encryption.
- The server sees each client's decoded update. There is no secure aggregation and no end-to-end differential-privacy
  guarantee.
- The sketch audit detects a mismatch between a client's payload and its declared sketch; it cannot detect a payload
  change that lies in the null space of the (publicly derivable) projection, nor a consistent free rider.
- Screening assumes that honest clients form the majority direction in sketch space; it reports rounds in which no
  usable consensus exists instead of passing them silently.
- Preprocessing statistics are fit centrally on the fit subjects' rows; only model training is federated.
- Evaluation with few held-out subjects has wide uncertainty. Use the `study` preset for final numbers and inspect
  `ci_reportable` in the summary.

## Citation

If you use this code, please cite the accompanying paper (details to be added):

```text
<author list>, "WIRE-FL: ...", <venue>, <year>.
```

## License

Add a `LICENSE` file of your choice before publishing.
