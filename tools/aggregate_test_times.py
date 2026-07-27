#!/usr/bin/env python3
"""Build a RISC-V-native ``test-times.json`` from pytorch-ci test-report artifacts.

Upstream PyTorch balances test shards using per-file timings that
``test/run_test.py`` reads from ``.additional_ci_files/test-times.json``. 

This script builds the file from our own test-report XMLs instead. Point it at
one or more directories of downloaded ``*-test-*`` artifacts and it emits JSON in
the exact shape ``load_test_times_from_file`` expects:

    {"<BUILD_ENVIRONMENT>": {"<TEST_CONFIG>": {"test_ops_gradients": 4098.2}}}

With ``--exclusions`` it instead reports which test files must be dropped for a
given shard count to fit a wall-clock budget, using PyTorch's real
``calculate_shards`` so the estimate matches what CI will actually do.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Any


# A shard that times out or is cancelled still uploads its test-report artifact,
# and partial reports are the common case for us (of 86 recent completed ci.yml
# runs, only 3 succeeded). Those stubs hold a single testcase where a healthy run
# holds thousands: merging them naively gave test_foreach the observations
# [1373s, 1478s, 0.0s] -- a 402x spread whose median badly under-sizes the shard.
# So an observation is only trusted if it saw at least this fraction of the most
# testcases anyone has ever reported for that file. With the filter, worst-case
# cross-run spread drops to 1.76x and the typical spread is 1.08x.
MIN_CASE_RATIO = 0.8

DEFAULT_BUILD_ENVIRONMENT = "pytorch-linux-noble-riscv64-py3.12-gcc14"
DEFAULT_TEST_CONFIG = "default"


def load_known_tests(pytorch_root: Path) -> list[str]:
    """The canonical test-file list, straight from PyTorch's own discovery."""
    sys.path.insert(0, str(pytorch_root))
    try:
        from tools.testing.discover_tests import TESTS  # noqa: PLC0415

        return list(TESTS)
    except ImportError as e:
        raise SystemExit(
            f"could not import tools.testing.discover_tests from {pytorch_root}: {e}\n"
            "Pass --pytorch pointing at a pytorch/pytorch checkout."
        ) from e


def sanitize_file_name(file: str) -> str:
    """Mirror of ``run_test.py``'s ``sanitize_file_name``."""
    return file.replace("\\", ".").replace("/", ".").replace(" ", "_")


def build_name_resolver(known_tests: list[str]) -> dict[str, str]:
    """Map a report directory name back to the test name ``run_test.py`` uses.

    Report directories are named after the *invoking file* run through
    ``sanitize_file_name``, which is lossy: ``/``, ``\\`` and spaces all collapse
    to ``.``, so ``export/test_sparse`` becomes ``export.test_sparse`` and cannot
    be inverted by string surgery. Resolve against the real test list instead --
    guessing would silently produce keys that never match anything.
    """
    resolver: dict[str, str] = {}
    for test in known_tests:
        resolver.setdefault(sanitize_file_name(test), test)
    return resolver


def parse_report(path: Path) -> tuple[float, int]:
    """Total ``testcase`` time and count in one report. ``(0.0, 0)`` if unusable."""
    try:
        root = ET.parse(path).getroot()
    except (ET.ParseError, OSError) as e:
        print(f"warning: skipping unparseable report {path}: {e}", file=sys.stderr)
        return 0.0, 0

    total = 0.0
    count = 0
    for case in root.iter("testcase"):
        try:
            total += float(case.get("time") or 0.0)
        except ValueError:
            pass
        count += 1
    return total, count


def aggregate_run(run_dir: Path) -> dict[str, tuple[float, int]]:
    """Aggregate every report under one CI run into per-invoking-file totals.

    Sibling XMLs in the same directory are *summed*: ``run_test.py`` writes one
    report per pytest sub-shard, and those cover disjoint slices of the file.
    (Medians across runs come later -- conflating the two axes either halves or
    doubles every number.)
    """
    times: dict[str, float] = defaultdict(float)
    cases: dict[str, int] = defaultdict(int)

    for report in run_dir.rglob("*.xml"):
        invoking = report.parent.name
        if not invoking:
            continue
        seconds, count = parse_report(report)
        times[invoking] += seconds
        cases[invoking] += count

    # A file whose every report is empty tells us nothing -- a crashed or skipped
    # invocation, not a fast one. Dropping it here keeps it out of the merge
    # entirely, rather than letting a 0.0s "measurement" through and under-sizing
    # a shard. (75 of 572 directories in run 30213211196 look like this.)
    return {
        name: (times[name], cases[name]) for name in times if cases[name] > 0
    }


def discover_run_dirs(artifacts_root: Path) -> list[Path]:
    """Find the per-run directories holding downloaded artifacts.

    ``refresh-test-times.yml`` downloads into ``artifacts/<run-id>/``, but a
    single hand-downloaded run is also useful, so fall back to treating the root
    itself as one run.
    """
    run_dirs = sorted(d for d in artifacts_root.iterdir() if d.is_dir())
    if run_dirs and any(d.rglob("*.xml") for d in run_dirs):
        return run_dirs
    return [artifacts_root]


def merge_observations(
    per_run: dict[str, list[tuple[float, int]]],
) -> tuple[dict[str, float], int]:
    """Collapse per-run observations into one time per file, dropping stubs."""
    merged: dict[str, float] = {}
    dropped = 0

    for name, observations in per_run.items():
        max_cases = max(count for _, count in observations)
        kept = [
            seconds
            for seconds, count in observations
            if count >= MIN_CASE_RATIO * max_cases
        ]
        dropped += len(observations) - len(kept)
        # Median, not mean: one pathological run should not move the estimate.
        if kept:
            merged[name] = statistics.median(kept)

    return merged, dropped


def collect(artifacts_root: Path, resolver: dict[str, str]) -> dict[str, float]:
    """Parse every run under ``artifacts_root`` into resolved per-file times."""
    per_run: dict[str, list[tuple[float, int]]] = defaultdict(list)
    unresolved: set[str] = set()
    run_dirs = discover_run_dirs(artifacts_root)

    for run_dir in run_dirs:
        run_totals = aggregate_run(run_dir)
        if not run_totals:
            continue
        for sanitized, (seconds, count) in run_totals.items():
            real_name = resolver.get(sanitized)
            if real_name is None:
                unresolved.add(sanitized)
                continue
            per_run[real_name].append((seconds, count))
        print(
            f"{run_dir.name}: {len(run_totals)} invoking files, "
            f"{sum(t for t, _ in run_totals.values()) / 3600:.1f}h",
            file=sys.stderr,
        )

    for name in sorted(unresolved):
        # Emitting the key anyway would look like data while matching nothing.
        print(f"warning: no test matches report dir {name!r}, dropping", file=sys.stderr)

    merged, dropped = merge_observations(per_run)
    print(
        f"\nmerged {len(merged)} files from {len(run_dirs)} run(s), "
        f"{sum(merged.values()) / 3600:.1f}h summed, "
        f"{dropped} truncated observation(s) dropped",
        file=sys.stderr,
    )
    return merged


def load_existing(path: Path, build_env: str, test_config: str) -> dict[str, float]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        print(f"warning: ignoring unreadable {path}: {e}", file=sys.stderr)
        return {}
    return data.get(build_env, {}).get(test_config, {})


def shard_estimates(
    times: dict[str, float],
    num_shards: int,
    pytorch_root: Path,
    excluded: set[str] | None = None,
) -> list[float]:
    """Per-shard time estimates from PyTorch's own sharding algorithm.

    Reuses ``calculate_shards`` rather than reimplementing it, so the numbers
    reflect the serial/parallel split and pytest sub-sharding that CI will apply.
    """
    sys.path.insert(0, str(pytorch_root))
    from tools.testing.test_run import TestRun  # noqa: PLC0415

    # test_selections tries to import torch for the CUDA probes, guarding only
    # ImportError. A source checkout with no built extension raises OSError from
    # dlopen instead, which would abort us. NUM_PROCS defaults correctly for our
    # CPU-only RISC-V runners either way, so stub the module out if torch is
    # unusable rather than requiring a full build just to estimate shards.
    try:
        import torch  # noqa: F401, PLC0415
    except Exception:
        sys.modules.setdefault("torch", None)  # type: ignore[assignment]

    from tools.testing.test_selections import calculate_shards  # noqa: PLC0415

    excluded = excluded or set()
    selected = [TestRun(name) for name in sorted(times) if name not in excluded]
    kept_times = {k: v for k, v in times.items() if k not in excluded}
    shards = calculate_shards(num_shards, selected, kept_times, test_class_times=None)
    return [total for total, _ in shards]


def compute_exclusions(
    times: dict[str, float],
    num_shards: int,
    budget_hours: float,
    pytorch_root: Path,
) -> list[str]:
    """Greedily drop the slowest files until every shard fits the budget.

    The RISC-V cost distribution is flat -- the worst file is only ~4.5% of total
    measured time -- so there is no small set of pathological tests to remove.
    Dropping longest-first is what keeps the excluded count minimal.
    """
    budget_seconds = budget_hours * 3600
    excluded: set[str] = set()
    candidates = sorted(times, key=lambda name: times[name], reverse=True)

    for name in candidates:
        estimates = shard_estimates(times, num_shards, pytorch_root, excluded)
        if estimates and max(estimates) <= budget_seconds:
            break
        excluded.add(name)
        print(
            f"  excluding {name} ({times[name] / 60:.1f}min), "
            f"worst shard now {max(estimates) / 3600:.2f}h",
            file=sys.stderr,
        )

    return sorted(excluded)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifacts",
        type=Path,
        required=True,
        help="directory of downloaded test-report artifacts (artifacts/<run-id>/...)",
    )
    parser.add_argument(
        "--pytorch",
        type=Path,
        required=True,
        help="path to a pytorch/pytorch checkout (for the test list and sharding)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("test-times.json"),
        help="where to write test-times.json (default: %(default)s)",
    )
    parser.add_argument(
        "--merge-into",
        type=Path,
        help="carry forward times from this existing file for files not measured now",
    )
    parser.add_argument(
        "--build-environment",
        default=DEFAULT_BUILD_ENVIRONMENT,
        help="top-level key; must match BUILD_ENVIRONMENT in CI (default: %(default)s)",
    )
    parser.add_argument(
        "--test-config",
        default=DEFAULT_TEST_CONFIG,
        help="inner key; must match TEST_CONFIG in CI (default: %(default)s)",
    )
    parser.add_argument(
        "--exclusions",
        action="store_true",
        help="print the files to exclude to fit the budget, instead of writing JSON",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=10,
        help="shard count for --exclusions and --report (default: %(default)s)",
    )
    parser.add_argument(
        "--budget-hours",
        type=float,
        default=5.0,
        help="per-shard wall-clock target for --exclusions (default: %(default)s)",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="print per-shard estimates for the resulting times",
    )
    args = parser.parse_args()

    if not args.artifacts.is_dir():
        raise SystemExit(f"--artifacts {args.artifacts} is not a directory")

    known_tests = load_known_tests(args.pytorch)
    resolver = build_name_resolver(known_tests)
    times = collect(args.artifacts, resolver)

    if args.merge_into:
        existing = load_existing(
            args.merge_into, args.build_environment, args.test_config
        )
        carried = {k: v for k, v in existing.items() if k not in times}
        if carried:
            print(
                f"carrying forward {len(carried)} file(s) not measured this time",
                file=sys.stderr,
            )
        times = {**carried, **times}

    if not times:
        raise SystemExit("no usable test times found; refusing to write an empty file")

    if args.exclusions:
        print(
            f"fitting {len(times)} files into {args.num_shards} shards "
            f"under {args.budget_hours}h:",
            file=sys.stderr,
        )
        excluded = compute_exclusions(
            times, args.num_shards, args.budget_hours, args.pytorch
        )
        remaining = shard_estimates(times, args.num_shards, args.pytorch, set(excluded))
        print(
            f"\nexcluded {len(excluded)} of {len(times)} files; "
            f"worst shard {max(remaining) / 3600:.2f}h",
            file=sys.stderr,
        )
        # stdout stays machine-readable: exactly what TESTS_TO_EXCLUDE wants.
        print(" ".join(excluded))
        return

    payload: dict[str, Any] = {
        args.build_environment: {
            args.test_config: {k: round(v, 3) for k, v in sorted(times.items())}
        }
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.output} with {len(times)} files", file=sys.stderr)

    if args.report:
        estimates = shard_estimates(times, args.num_shards, args.pytorch)
        for i, total in enumerate(estimates, 1):
            print(f"  shard {i}/{args.num_shards}: {total / 3600:.2f}h", file=sys.stderr)
        print(f"  worst: {max(estimates) / 3600:.2f}h", file=sys.stderr)


if __name__ == "__main__":
    main()
