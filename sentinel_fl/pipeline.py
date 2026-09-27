"""Per-subject training/evaluation pipeline and the leave-one-subject-out driver."""

from __future__ import annotations

import json
import math
import pickle
import platform
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.special import expit
from threadpoolctl import threadpool_limits

from .config import VERSION, code_fingerprint
from .decoding import SequenceDecoder
from .federated import LR_SEARCH, SENTINEL_FL, Scenario, run_federated, summarize_state
from .metrics import Calibrator, metric_dict, select_threshold
from .personalization import personalization_rows
from .preprocessing import Preprocessor, audit_partitions, make_splits, role_frame
from .utils import atomic_pickle, digest, hash_frame, write_json

SHARD_FILES = ("participant_metrics", "system_metrics", "personalization", "completion_ledger")


def choose_lr(fit, val, prep, cfg, seed, root, code_hash):
    """Select the learning rate on validation subjects with short full-precision federated runs.

    Returns ``(cfg_with_selected_lr, selection_record)``. Results are cached by content hash.
    """
    searchcfg = replace(cfg, rounds=cfg.hpo_rounds, eval_every=min(cfg.eval_every, cfg.hpo_rounds))
    key = digest(dict(fit=hash_frame(fit), val=hash_frame(val), cfg=asdict(searchcfg), seed=seed, code=code_hash))
    path = Path(root) / "selection" / f"{key}.json"
    if path.exists():
        result = json.loads(path.read_text())
        return replace(cfg, lr=result["lr"]), result
    choices = []
    for lr in cfg.hp_lrs:
        trial = replace(searchcfg, lr=lr)
        checkpoint = path.with_name(key + f"_{lr:g}.pkl")
        _, st = run_federated(fit, val, prep, trial, LR_SEARCH, Scenario(), seed, checkpoint, code_hash)
        choices.append(dict(lr=lr, val_macro_f1=st["best_score"], train_s=st["train_seconds"]))
    winner = max(choices, key=lambda x: (x["val_macro_f1"], -abs(math.log(x["lr"] / 0.003))))
    result = dict(
        lr=winner["lr"],
        trials=choices,
        criterion="validation-subject mean macro-F1 at threshold 0.5",
        tuning_train_s=sum(x["train_s"] for x in choices),
    )
    write_json(path, result)
    return replace(cfg, lr=winner["lr"]), result


def evaluate_artifact(predict_logits, val, cal, test, cfg, minority=1):
    """Calibrate, decode and evaluate a trained model on a held-out subject.

    Probability calibration uses the calibration subjects; the decision threshold and the
    sequence-decoder strength are chosen on the validation subjects. Returns
    ``(metrics, predictions, personalisation_rows, calibration_record)``.
    """
    calibrator = Calibrator().fit(predict_logits(cal), cal)
    zval = predict_logits(val)
    p_val = calibrator.predict(zval)
    decoder = SequenceDecoder(cfg.sequence_strengths, cfg.sequence_min_gap_ratio)
    decode_info = dict(strength=0.0, status="disabled")
    if cfg.sequence_decode:
        decoder.fit(cal)
        decode_info = decoder.tune(val, p_val)
        threshold = float(decode_info["threshold"])
    else:
        threshold = select_threshold(val, p_val)
    z = predict_logits(test)
    p_cal = calibrator.predict(z)
    p = decoder.smooth(test, p_cal) if cfg.sequence_decode else p_cal
    y_test = test.y.to_numpy(int)
    metrics = metric_dict(y_test, p, threshold, minority)
    metrics.update(
        calibration_status=calibrator.status,
        sequence_strength=float(decode_info.get("strength", 0.0)),
        sequence_status=decode_info.get("status", "disabled"),
        sequence_stay_class0=float(decode_info.get("stay_class0", np.nan)),
        sequence_stay_class1=float(decode_info.get("stay_class1", np.nan)),
    )
    pred = test[["client_id", "label_start", "label_end", "y"]].copy()
    pred["p_raw"] = expit(z)
    pred["p_calibrated"] = p_cal
    pred["p"] = p
    pred["threshold"] = threshold
    rows = personalization_rows(predict_logits, calibrator, decoder, test, cfg, threshold, minority)
    record = dict(
        a=calibrator.a,
        b=calibrator.b,
        status=calibrator.status,
        threshold=threshold,
        sequence=decode_info,
        decoder=decoder,
    )
    return metrics, pred, rows, record


def run_fold(df, sp, cfg, scenario, seed, root, code_hash, synthetic=False):
    """Train SENTINEL-FL with one held-out subject and evaluate it; results are cached under ``root/runs``."""
    audit_partitions(sp)
    fit, val = role_frame(df, sp, "fit"), role_frame(df, sp, "val")
    cal, test = role_frame(df, sp, "cal"), role_frame(df, sp, "test")
    prep = Preprocessor().fit(fit, cfg.feature_clip)
    cfg, tuning = choose_lr(fit, val, prep, cfg, seed, root, code_hash)
    key = digest(
        dict(code=code_hash, data=hash_frame(df), split=sp, cfg=asdict(cfg), scenario=asdict(scenario), seed=seed)
    )
    folder = Path(root) / "runs" / key[:24]
    done = folder / "result.pkl"
    if done.exists():
        with done.open("rb") as f:
            return pickle.load(f)
    folder.mkdir(parents=True, exist_ok=True)
    manifest = dict(
        key=key,
        code_hash=code_hash,
        split=sp,
        cfg=asdict(cfg),
        scenario=asdict(scenario),
        seed=seed,
        tuning=tuning,
        features=prep.features,
    )
    write_json(folder / "manifest.json", manifest)
    model, state = run_federated(fit, val, prep, cfg, SENTINEL_FL, scenario, seed, folder / "checkpoint.pkl", code_hash)

    def predict(g):
        return model.logits(prep.transform(g))

    system = summarize_state(state)
    pd.DataFrame(state["logs"]).to_csv(folder / "rounds.csv", index=False)
    pd.DataFrame(state["events"]).to_csv(folder / "client_events.csv", index=False)
    minority = 0 if sp["dataset"] == "SLEEP" else 1
    metrics, preds, personal, calibration = evaluate_artifact(predict, val, cal, test, cfg, minority)
    probe = prep.transform(test)[:1]
    model.logits(probe)
    times = []
    for _ in range(30):
        tt = time.perf_counter()
        model.logits(probe)
        times.append((time.perf_counter() - tt) * 1000)
    metrics.update(inference_ms_median=float(np.median(times)), inference_ms_p95=float(np.quantile(times, 0.95)))
    tag = dict(
        dataset=sp["dataset"],
        client_id=sp["held"],
        seed=seed,
        scenario=scenario.name,
        attack=scenario.attack,
        bad_fraction=scenario.bad_fraction,
        synthetic=bool(synthetic),
        run_key=key,
        features=len(prep.features),
        n_fit_subjects=len(sp["fit"]),
        n_validation_subjects=len(sp["val"]),
        n_calibration_subjects=len(sp["cal"]),
    )
    metrics.update(tag)
    model.set_flat(state["theta"])
    last_p = expit(calibration["a"] * np.clip(predict(test), -30, 30) + calibration["b"])
    if cfg.sequence_decode:
        last_p = calibration["decoder"].smooth(test, last_p)
    last = metric_dict(test.y.to_numpy(int), last_p, calibration["threshold"], minority)
    metrics["final_round_macro_f1"] = last["macro_f1"]
    model.set_flat(state["best_theta"])
    public_calibration = {k: v for k, v in calibration.items() if k != "decoder"}
    atomic_pickle(
        folder / "artifact.pkl",
        dict(
            version=VERSION,
            dataset=sp["dataset"],
            held_out=sp["held"],
            model=model,
            preprocessor=prep,
            calibration=public_calibration,
            decoder=calibration["decoder"],
            threshold=calibration["threshold"],
            sequence_decode=bool(cfg.sequence_decode),
            minority_class=minority,
        ),
    )
    for row in personal:
        row.update(tag)
    preds.to_csv(folder / "predictions.csv", index=False)
    result = dict(
        metrics=metrics,
        system=system,
        personalization=personal,
        calibration=public_calibration,
        tuning=tuning,
        run_dir=str(folder),
        manifest=manifest,
    )
    write_json(folder / "result.json", result)
    atomic_pickle(done, result)
    return result


def split_shard(splits, shard=(0, 1)):
    """Deterministic disjoint subset of held-out subjects for shard ``k`` of ``n``."""
    k, n = int(shard[0]), int(shard[1])
    if n < 1 or not 0 <= k < n:
        raise ValueError("shard must be (k, n) with 0 <= k < n")
    if n == 1:
        return splits, ""
    return [sp for i, sp in enumerate(splits) if i % n == k], f"_shard{k}of{n}"


def merge_shards(root, expected_shards=None):
    """Concatenate per-shard CSVs into the canonical file names; refuses incomplete sets."""
    root = Path(root)
    merged = {}
    for stem in SHARD_FILES:
        parts = sorted(root.glob(f"{stem}_shard*of*.csv"))
        if not parts:
            continue
        seen = {p.name.split("_shard")[1].split(".csv")[0] for p in parts}
        if expected_shards is not None and len(seen) != int(expected_shards):
            raise RuntimeError(
                f"{stem}: found {len(seen)} shards, expected {expected_shards}. Refusing to merge an incomplete run."
            )
        out = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
        out.to_csv(root / f"{stem}.csv", index=False)
        merged[stem] = len(out)
        print(f"merged {len(parts)} shards -> {stem}.csv ({len(out)} rows)")
    if "completion_ledger" in merged:
        led = pd.read_csv(root / "completion_ledger.csv")
        if led.duplicated(["dataset", "client_id", "seed", "scenario"]).any():
            raise RuntimeError("Overlapping shards: the same job appears twice in the merged ledger.")
    return merged


def run_loso(df, provenance, cfg, root, scenario=None, shard=(0, 1), datasets=None, data_config=None):
    """Leave-one-subject-out evaluation of SENTINEL-FL over every requested held-out subject and seed.

    Writes ``participant_metrics``, ``system_metrics``, ``personalization`` and
    ``completion_ledger`` CSVs (with a shard suffix if sharded) plus ``run_manifest*.json`` and
    returns the four frames and the manifest. A failed job is recorded in the ledger and the run
    continues, so no partial result can pass as complete.
    """
    scenario = scenario or Scenario()
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    synthetic = bool(provenance.get("synthetic", False))
    if datasets:
        df = df[df.dataset.isin(list(datasets))].copy()
    all_splits = make_splits(df, cfg)
    splits, shard_tag = split_shard(all_splits, shard)
    code_hash = code_fingerprint()
    manifest = dict(
        version=VERSION,
        code_hash=code_hash,
        provenance=provenance,
        config=asdict(cfg),
        scenario=asdict(scenario),
        data_config=data_config,
        splits=all_splits,
        shard=list(shard),
        environment=dict(
            python=sys.version,
            platform=platform.platform(),
            numpy=np.__version__,
            pandas=pd.__version__,
            scipy=scipy.__version__,
            sklearn=sklearn.__version__,
        ),
        inference_unit="held-out participant; seeds averaged within participant",
        planned_runs=len(splits) * len(cfg.seeds),
        total_planned_runs=len(all_splits) * len(cfg.seeds),
    )
    write_json(root / f"run_manifest{shard_tag}.json", manifest)
    print(
        f"Planned {len(splits)} held-out subjects x {len(cfg.seeds)} seeds = {manifest['planned_runs']} runs",
        flush=True,
    )
    rows, systems, personal, ledger = [], [], [], []
    seen = 0
    t_start = time.perf_counter()
    for sp in splits:
        cohort = df[df.dataset == sp["dataset"]].copy()
        for seed in cfg.seeds:
            tag = dict(dataset=sp["dataset"], client_id=sp["held"], seed=seed, scenario=scenario.name)
            try:
                with threadpool_limits(limits=cfg.n_threads):
                    result = run_fold(cohort, sp, cfg, scenario, seed, root, code_hash, synthetic)
                rows.append(dict(result["metrics"]))
                systems.append(dict(result["system"], **tag))
                personal.extend(dict(x) for x in result["personalization"])
                ledger.append(dict(tag, status="completed", error="", run_dir=result["run_dir"]))
            except Exception as e:  # noqa: BLE001 - recorded in the ledger, never silently skipped
                ledger.append(dict(tag, status="failed", error=f"{type(e).__name__}: {e}"))
                print("FAILED", sp["held"], seed, str(e)[:200], flush=True)
            seen += 1
            pd.DataFrame(rows).to_csv(root / f"participant_metrics{shard_tag}.csv", index=False)
            pd.DataFrame(systems).to_csv(root / f"system_metrics{shard_tag}.csv", index=False)
            pd.DataFrame(personal).to_csv(root / f"personalization{shard_tag}.csv", index=False)
            pd.DataFrame(ledger).to_csv(root / f"completion_ledger{shard_tag}.csv", index=False)
            rate = (time.perf_counter() - t_start) / max(seen, 1)
            print(
                f"{seen}/{manifest['planned_runs']} runs recorded ({sum(x['status'] == 'failed' for x in ledger)} failed); "
                f"{rate:.1f} s/run",
                flush=True,
            )
    return pd.DataFrame(rows), pd.DataFrame(systems), pd.DataFrame(personal), pd.DataFrame(ledger), manifest
