"""Command-line interface: ``python -m sentinel_fl run|merge|summarize``."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from .config import PRESETS, __version__, data_config, get_config
from .federated import Scenario
from .features import load_data
from .pipeline import merge_shards, run_loso
from .reporting import write_summary
from .utils import write_json

ATTACKS = ("clean", "sign_flip", "scaled", "gaussian", "label_flip", "sketch_forgery", "free_rider", "nullspace")


def _parse_shard(text):
    k, n = text.split("/")
    return int(k), int(n)


def build_parser():
    p = argparse.ArgumentParser(
        prog="sentinel-fl", description="SENTINEL-FL: byte-capped, screened, audited federated learning on wearable data."
    )
    p.add_argument("--version", action="version", version=f"sentinel-fl {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    r = sub.add_parser("run", help="train and evaluate with leave-one-subject-out")
    r.add_argument(
        "--mode",
        choices=sorted(PRESETS),
        default="study",
        help="smoke = synthetic software test; pilot = 8 held-out subjects per dataset, 1 seed; study = every subject, 3 seeds",
    )
    r.add_argument("--out", default=None, help="output folder (default: results/<mode>)")
    r.add_argument("--wesad-dir", default=None, help="folder containing S2/S2.pkl, S3/S3.pkl, ...")
    r.add_argument("--sleep-dir", default=None, help="folder containing heart_rate/, labels/, motion/ and steps/")
    r.add_argument(
        "--wesad-subjects", nargs="+", default=None, help="WESAD subject IDs to use (default: S2-S11 and S13-S17)"
    )
    r.add_argument("--cache-dir", default=None, help="feature cache folder (default: sentinel_fl_cache)")
    r.add_argument("--datasets", nargs="+", choices=["SLEEP", "WESAD"], default=None)
    r.add_argument("--seeds", nargs="+", type=int, default=None)
    r.add_argument("--rounds", type=int, default=None)
    r.add_argument("--byte-budget", type=int, default=None, help="hard cap of one client upload in bytes")
    r.add_argument("--max-folds", type=int, default=None, help="held-out subjects per dataset (default: preset)")
    r.add_argument("--hidden", type=int, default=None)
    r.add_argument("--no-sequence-decode", action="store_true")
    r.add_argument("--context-time-mode", choices=["causal", "retrospective"], default=None)
    r.add_argument(
        "--sensor-profile",
        choices=["all", "wrist"],
        default=None,
        help="WESAD sensors: all (wrist+chest) or wrist only",
    )
    r.add_argument(
        "--attack", choices=ATTACKS, default="clean", help="simulate malicious clients to exercise screening and audit"
    )
    r.add_argument("--bad-fraction", type=float, default=0.0)
    r.add_argument(
        "--shard", type=_parse_shard, default=(0, 1), metavar="K/N", help="run shard K of N held-out subject groups"
    )

    m = sub.add_parser("merge", help="merge per-shard CSVs and write the summary")
    m.add_argument("--out", required=True)
    m.add_argument("--shards", type=int, default=None, help="expected number of shards")

    s = sub.add_parser("summarize", help="(re)write summary tables and RESULTS_SUMMARY.md from a finished run folder")
    s.add_argument("--out", required=True)
    return p


def _run(args):
    cfg = get_config(args.mode)
    overrides = {}
    if args.seeds:
        overrides["seeds"] = tuple(args.seeds)
    if args.rounds is not None:
        overrides["rounds"] = args.rounds
    if args.byte_budget is not None:
        overrides["byte_budget"] = args.byte_budget
    if args.max_folds is not None:
        overrides["max_folds"] = args.max_folds
    if args.hidden is not None:
        overrides["hidden"] = args.hidden
    if args.no_sequence_decode:
        overrides["sequence_decode"] = False
    cfg = replace(cfg, **overrides)
    data_over = {}
    if args.cache_dir:
        data_over["cache_dir"] = args.cache_dir
    if args.wesad_subjects:
        data_over["wesad_subjects"] = list(args.wesad_subjects)
    if args.context_time_mode:
        data_over["context_time_mode"] = args.context_time_mode
    if args.sensor_profile:
        data_over["sensor_profile"] = args.sensor_profile
    dcfg = data_config(args.wesad_dir, args.sleep_dir, **data_over)
    out = Path(args.out or f"results/{args.mode}")
    out.mkdir(parents=True, exist_ok=True)
    df, provenance = load_data(dcfg, synthetic=(args.mode == "smoke"))
    summary = df.groupby(["dataset", "task"]).agg(
        subjects=("client_id", "nunique"), windows=("y", "size"), positive_prevalence=("y", "mean")
    )
    print(summary.to_string())
    write_json(out / "data_provenance.json", provenance)
    summary.to_csv(out / "data_summary.csv")
    scenario = Scenario(
        name=args.attack if args.attack == "clean" else f"{args.attack}_{args.bad_fraction:g}",
        attack=args.attack,
        bad_fraction=0.0 if args.attack == "clean" else args.bad_fraction,
    )
    _, _, _, ledger, _ = run_loso(df, provenance, cfg, out, scenario, args.shard, args.datasets, dcfg)
    failed = int((ledger.status != "completed").sum())
    if args.shard[1] == 1:
        write_summary(out, cfg.min_ci_subjects)
        print(f"Summary written to {out / 'RESULTS_SUMMARY.md'}")
    else:
        print(
            f"Shard {args.shard[0]}/{args.shard[1]} finished. When all shards are done: sentinel-fl merge --out {out} --shards {args.shard[1]}"
        )
    if failed:
        raise SystemExit(
            f"{failed} run(s) failed; see completion_ledger.csv. Fix and rerun to resume; do not report a partial result."
        )


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == "run":
        _run(args)
    elif args.command == "merge":
        merge_shards(args.out, args.shards)
        write_summary(args.out)
        print(f"Summary written to {Path(args.out) / 'RESULTS_SUMMARY.md'}")
    else:
        write_summary(args.out)
        print(f"Summary written to {Path(args.out) / 'RESULTS_SUMMARY.md'}")


if __name__ == "__main__":
    main()
