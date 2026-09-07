#!/usr/bin/env python3
"""Attribute native_batch construction with isolated control and diagnostic builds."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess


STAGES = ["binding", "support", "prepare", "batch_creation", "jacobian_allocation",
          "observation_storage", "native_registration", "binding_cleanup", "channel_cleanup"]


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_csv(path):
    with Path(path).open() as stream:
        return list(csv.DictReader(stream))


def mean(rows, key):
    return statistics.mean(float(row[key]) for row in rows)


def summarize(rows):
    groups = {}
    references = {(r["round"], r["snapshot"]): r for r in rows if r["group"] == "control"}
    for row in rows:
        groups.setdefault(row["group"], []).append(row)
    result = {"groups": [], "stages": [], "max_relative_final_cost_difference": 0.0}
    for name, samples in sorted(groups.items()):
        ref = [references[(r["round"], r["snapshot"])] for r in samples]
        error = max(abs(float(r["final_cost"]) - float(a["final_cost"])) /
                    max(1.0, abs(float(a["final_cost"]))) for r, a in zip(samples, ref))
        result["max_relative_final_cost_difference"] = max(result["max_relative_final_cost_difference"], error)
        values = sorted(float(r["build_ms"]) for r in samples)
        entry = {"group": name, "samples": len(samples), "mean_build_ms": statistics.mean(values),
                 "p95_build_ms": values[math.ceil(.95 * len(values)) - 1],
                 "mean_total_ms": mean(samples, "total_ms"), "mean_destroy_ms": mean(samples, "destroy_ms"),
                 "matched_control_build_ms": mean(ref, "build_ms"),
                 "build_perturbation_percent": 100 * (mean(samples, "build_ms") / mean(ref, "build_ms") - 1),
                 "max_relative_final_cost_difference": error}
        result["groups"].append(entry)
        if not name.startswith("sample_"):
            continue
        entry["mean_empty_leaf_ns"] = mean(samples, "empty_leaf_ns")
        entry["mean_empty_child_gap_ns"] = mean(samples, "empty_child_gap_ns")
        entry["mean_empty_leaf_wall_ns"] = mean(samples, "empty_leaf_wall_ns")
        entry["mean_observations"] = mean(samples, "observations")
        entry["mean_sampled_observations"] = mean(samples, "sampled_observations")
        calibrated_sum = 0.0
        for stage in STAGES:
            # Empty-scope subtraction estimates probe costs, including the
            # unmeasured gaps around nested scopes. It does not remove codegen,
            # allocator, branch-prediction or cache perturbation; retain raw data.
            adjusted = [(float(r[stage + "_estimated_ns"]) -
                         float(r["empty_leaf_ns"]) * float(r[stage + "_estimated_calls"]) -
                         float(r["empty_child_gap_ns"]) * float(r[stage + "_estimated_children"])) * 1e-6
                        for r in samples]
            estimate = statistics.mean(adjusted)
            calibrated_sum += estimate
            per_round = {}
            for r, value in zip(samples, adjusted):
                per_round.setdefault(r["round"], []).append(value)
            result["stages"].append({"group": name, "stage": stage,
                                     "raw_sampled_mean_ms": mean(samples, stage + "_raw_ns") * 1e-6,
                                     "extrapolated_uncorrected_ms": mean(samples, stage + "_estimated_ns") * 1e-6,
                                     "calibrated_estimate_ms": estimate,
                                     "percent_of_matched_control_build": 100 * estimate / mean(ref, "build_ms"),
                                     "mean_sampled_scopes": mean(samples, stage + "_calls"),
                                     "mean_estimated_scopes": mean(samples, stage + "_estimated_calls"),
                                     "round_estimate_ms": {k: statistics.mean(v) for k, v in per_round.items()}})
        entry["calibrated_stage_sum_ms"] = calibrated_sum
        entry["unattributed_control_balance_ms"] = mean(ref, "build_ms") - calibrated_sum
        if name == "sample_1":
            entry["raw_unscoped_build_ms"] = mean(samples, "build_ms") - sum(
                mean(samples, stage + "_raw_ns") * 1e-6 for stage in STAGES)
    if result["max_relative_final_cost_difference"] > 1e-7:
        raise RuntimeError("Diagnostic/control costs differ; do not interpret timings")
    return result


def main():
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshots", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--max-snapshots", type=int, default=101)
    parser.add_argument("--validate", type=int, default=101)
    parser.add_argument("--control", type=Path, default=repo / "devel/lib/slict/slict_construction_control")
    parser.add_argument("--diagnostic", type=Path, default=repo / "devel/lib/slict/slict_construction_diagnostics")
    args = parser.parse_args()
    if args.threads < 1 or args.max_snapshots < 1 or args.validate < 0:
        parser.error("Invalid counts")
    args.output.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env.update(OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", OMP_NUM_THREADS="1",
               OMP_DYNAMIC="FALSE", OMP_MAX_ACTIVE_LEVELS="1")
    manifest = {"platform": platform.platform(), "cpu_model": next(
        (line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
         if line.startswith("model name")), "unknown"), "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "threads": args.threads, "construction_serial": True,
        "controlled_environment": {k: env[k] for k in ["OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                                                       "OMP_NUM_THREADS", "OMP_DYNAMIC", "OMP_MAX_ACTIVE_LEVELS"]},
        "timing": "Same single-step Solve/native_batch; build_ms is official only in control. Diagnostic stages are exclusive, observation samples weighted by stride. Empty probes provide an approximate correction, not exact native stage times.",
        "sources": {}, "binaries": {}, "runs": []}
    paths = list((repo / "src/comparison").glob("*")) + list((repo / "third_party/splbatch/include/splbatch").glob("*.hpp"))
    paths += [Path(__file__).resolve(), repo / "CMakeLists.txt", repo / "include/slict/solver_comparison.h"]
    manifest["sources"] = {str(p.relative_to(repo)): digest(p) for p in paths if p.is_file()}
    files = sorted(args.snapshots.glob("*.slict"))[:args.max_snapshots]
    if not files:
        parser.error("No snapshots found")
    manifest["snapshots"] = {str(p.resolve()): digest(p) for p in files}
    for name, binary in [("control", args.control), ("diagnostic", args.diagnostic)]:
        linkage = subprocess.check_output(["ldd", str(binary.resolve())], text=True)
        (args.output / (name + "_linkage.txt")).write_text(linkage)
        comparison_lines = [line for line in linkage.splitlines() if "libslict_comparison" in line]
        if len(comparison_lines) != 1 or ("_diagnostics" in comparison_lines[0]) != (name == "diagnostic"):
            raise RuntimeError("Control and diagnostic library isolation failed")
        library = Path(comparison_lines[0].split("=>", 1)[1].split()[0])
        manifest["binaries"][name] = {"path": str(binary.resolve()), "sha256": digest(binary),
                                      "library": str(library), "library_sha256": digest(library)}

    def run(group, round_id, stride, offset, validate=0):
        diagnostic = group != "control"
        binary = args.diagnostic if diagnostic else args.control
        stem = f"{group}_r{round_id}"
        output = args.output / (stem + ".csv")
        command = [str(binary.resolve()), "--snapshots", str(args.snapshots.resolve()),
                   "--output", str(output.resolve()), "--threads", str(args.threads),
                   "--repeat", "1", "--warmup", "0" if validate else "1",
                   "--stride", str(stride), "--offset", str(offset),
                   "--enabled", "0" if group in ("control", "disabled") else "1",
                   "--validate", str(validate), "--max-snapshots", str(args.max_snapshots)]
        print(stem, flush=True)
        with (args.output / (stem + ".log")).open("w") as log:
            subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        current = read_csv(output)
        if len(current) != len(files) or any(r["usable"] != "1" for r in current):
            raise RuntimeError("Missing or unusable samples")
        for row in current:
            row["round"] = str(round_id)
            row["group"] = group
        manifest["runs"].append({"group": group, "round": round_id, "command": command,
                                 "rows": len(current), "csv_sha256": digest(output), "validation": bool(validate)})
        (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        return current

    if args.validate:
        run("validation", -1, 1, 0, args.validate)
    rows = []
    for r in range(16):
        jobs = [("control", 1, 0), ("sample_16", 16, r)]
        if r in (0, 3, 6, 9, 12):
            jobs += [("disabled", 1, 0), ("sample_1", 1, 0)]
        if r % 4 == 0:
            jobs.append(("sample_4", 4, r // 4))
        shift = r % len(jobs)
        jobs = jobs[shift:] + jobs[:shift]
        for group, stride, offset in jobs:
            rows.extend(run(group, r, stride, offset))
    # Rotating offsets cover every observation once across each full sampling
    # cycle, including first observations of new batches.
    for group, stride in [("sample_4", 4), ("sample_16", 16)]:
        by_snapshot = {}
        for r in rows:
            if r["group"] == group:
                by_snapshot.setdefault(r["snapshot"], []).append(r)
        for samples in by_snapshot.values():
            if len(samples) != stride or sum(int(r["sampled_observations"]) for r in samples) != int(samples[0]["observations"]):
                raise RuntimeError("Sampling offsets did not cover every observation exactly once")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (args.output / "all.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    result = summarize(rows)
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
