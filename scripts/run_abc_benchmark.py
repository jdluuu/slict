#!/usr/bin/env python3
"""Benchmark identical saved windows in isolated, interleaved backend processes."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import statistics
import subprocess
import time


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshots", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--binary", type=Path, default=Path(__file__).resolve().parents[1] / "devel/lib/slict/slict_solver_benchmark")
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 4, 8, 16])
    parser.add_argument("--backends", nargs="+", choices=["native", "ceres_scalar", "ceres_batch", "native_batch"],
                        default=["native", "ceres_scalar", "ceres_batch"])
    parser.add_argument("--iterations", type=int, nargs="+", default=[1])
    parser.add_argument("--ceres-initial-radius", type=float, default=1e4,
                        help="Same initial LM radius for B/C on every frozen problem; independent of live continuation")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--max-snapshots", type=int, default=200)
    parser.add_argument("--validate", type=int, default=3)
    args = parser.parse_args()
    if len(set(args.backends)) != len(args.backends):
        parser.error("Duplicate backends")
    if not args.binary.is_file() or not args.snapshots.exists():
        parser.error("Binary and snapshots must exist")
    if args.rounds < 1 or args.max_snapshots < 1 or args.validate < 0 or min(args.threads + args.iterations) < 1:
        parser.error("Counts, threads and iterations must be positive")
    if len(set(args.threads)) != len(args.threads) or len(set(args.iterations)) != len(args.iterations):
        parser.error("Duplicate thread or iteration configurations")
    if not math.isfinite(args.ceres_initial_radius) or not 0 < args.ceres_initial_radius <= 1e16:
        parser.error("Ceres initial radius must be in (0, 1e16]")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env.update(OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", OMP_NUM_THREADS="1",
               OMP_DYNAMIC="FALSE", OMP_MAX_ACTIVE_LEVELS="1")
    linkage = subprocess.check_output(["ldd", str(args.binary.resolve())], text=True)
    (output / "linkage.txt").write_text(linkage)
    libraries = {}
    for line in linkage.splitlines():
        match = re.match(r"\s*(\S+) => (/\S+)", line)
        if match and any(name in match[1] for name in ("slict_comparison", "ceres", "cholmod", "spqr", "suitesparseconfig", "gomp")):
            path = Path(match[2]).resolve()
            libraries[match[1]] = {"path": str(path), "sha256": digest(path)}
    repo = Path(__file__).resolve().parents[1]
    sources = list((repo / "src/comparison").glob("*")) + [
        repo / "CMakeLists.txt", repo / "include/slict/solver_comparison.h", Path(__file__).resolve()]
    sources += list((repo / "third_party/splbatch/include/splbatch").glob("*.hpp"))
    cpu_model = next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                      if line.startswith("model name")), "unknown")
    manifest = {"platform": platform.platform(), "cpu_count": os.cpu_count(),
                "cpu_model": cpu_model,
                "cpu_affinity": sorted(os.sched_getaffinity(0)),
                "binary": str(args.binary.resolve()), "binary_sha256": digest(args.binary),
                "shared_libraries": libraries,
                "source_sha256": {str(path.relative_to(repo)): digest(path) for path in sources if path.is_file()},
                "git_head": subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip(),
                "controlled_environment": {key: env[key] for key in (
                    "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "OMP_DYNAMIC", "OMP_MAX_ACTIVE_LEVELS")},
                "snapshots": str(args.snapshots.resolve()), "threads": args.threads,
                "iterations": args.iterations, "rounds": args.rounds,
                "backends": args.backends,
                "ceres_initial_trust_region_radius": args.ceres_initial_radius,
                "timing": "total_ms includes reset/build/solve/destruction; excludes common validation, final cost diagnostic, I/O and marginalization",
                "runs": []}
    # Validate separately so allocation during H/J checks cannot pollute per-backend peak RSS.
    if args.validate:
        validation = [str(args.binary.resolve()), "--snapshots", str(args.snapshots.resolve()),
                      "--ceres-initial-radius", str(args.ceres_initial_radius),
                      "--max-snapshots", str(args.validate), "--validate", str(args.validate),
                      "--repeat", "1", "--warmup", "0", "--output", str(output / "validation.csv")]
        with (output / "validation.log").open("w") as log:
            subprocess.run(validation, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    rows = []
    backends = args.backends
    for threads in args.threads:
        for iterations in args.iterations:
            for round_id in range(args.rounds):
                shift = round_id % len(backends)
                order = backends[shift:] + backends[:shift]
                if len(backends) > 2 and round_id % 2:
                    order = list(reversed(order))
                for backend in order:
                    stem = f"{backend}_t{threads}_i{iterations}_r{round_id}"
                    csv_path = output / (stem + ".csv")
                    command = [str(args.binary.resolve()), "--snapshots", str(args.snapshots.resolve()),
                               "--backend", backend, "--threads", str(threads), "--iterations", str(iterations),
                               "--ceres-initial-radius", str(args.ceres_initial_radius),
                               "--repeat", "1", "--warmup", "1", "--validate", "0",
                               "--max-snapshots", str(args.max_snapshots), "--output", str(csv_path)]
                    print(stem, flush=True)
                    start = time.monotonic()
                    with (output / (stem + ".log")).open("w") as log:
                        subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
                    with csv_path.open() as stream:
                        current = list(csv.DictReader(stream))
                    for row in current:
                        row["round"] = round_id
                    rows.extend(current)
                    manifest["runs"].append({"name": stem, "seconds": time.monotonic() - start,
                                             "rows": len(current), "command": command})
                    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if not rows:
        raise RuntimeError("No measurements produced")
    with (output / "all.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    groups = {}
    paired = {}
    for row in rows:
        key = (int(row["threads"]), int(row["iteration_limit"]), row["backend"])
        groups.setdefault(key, []).append(row)
        pair_key = (row["snapshot"], int(row["threads"]), int(row["iteration_limit"]), row["round"])
        if row["backend"] in paired.setdefault(pair_key, {}):
            raise RuntimeError("Duplicate paired sample")
        paired[pair_key][row["backend"]] = row
    if any(set(pair) != set(backends) for pair in paired.values()):
        raise RuntimeError("Backends processed different snapshot sets; refusing an unpaired comparison")
    summary = []
    for (threads, iterations, backend), samples in sorted(groups.items()):
        times = sorted(float(row["total_ms"]) for row in samples)
        entry = {"threads": threads, "iteration_limit": iterations, "backend": backend, "samples": len(samples),
                 "mean_ms": statistics.mean(times), "p50_ms": statistics.median(times),
                 "p95_ms": times[math.ceil(.95 * len(times)) - 1], "p99_ms": times[math.ceil(.99 * len(times)) - 1],
                 "peak_rss_mib": max(int(row["process_peak_rss_kib"]) for row in samples) / 1024,
                 "mean_final_cost": statistics.mean(float(row["final_cost"]) for row in samples),
                 "cost_increase_samples": sum(float(row["final_cost"]) > float(row["initial_cost"]) +
                                              1e-9 * max(1., abs(float(row["initial_cost"]))) for row in samples),
                 "unusable": sum(row["usable"] != "1" for row in samples)}
        for metric in ("build_ms", "evaluate_ms", "assemble_ms", "linear_ms", "solve_ms", "destroy_ms", "diagnostic_ms"):
            entry["mean_" + metric] = statistics.mean(float(row[metric]) for row in samples)
        entry["mean_cost_reduction"] = statistics.mean(float(row["initial_cost"]) - float(row["final_cost"]) for row in samples)
        summary.append(entry)
    pair_summary = []
    for threads in args.threads:
        for iterations in args.iterations:
            pairs = [p for key, p in paired.items() if key[1:3] == (threads, iterations)]
            def speed(left, right):
                return statistics.median(float(p[left]["total_ms"]) / float(p[right]["total_ms"]) for p in pairs)
            def cost_error(left, right):
                return max(abs(float(p[left]["final_cost"]) - float(p[right]["final_cost"])) /
                           max(1., abs(float(p[left]["final_cost"]))) for p in pairs)
            entry = {"threads": threads, "iteration_limit": iterations}
            if {"native", "ceres_batch"} <= set(backends):
                entry["median_native_over_batch"] = speed("native", "ceres_batch")
            if {"ceres_scalar", "ceres_batch"} <= set(backends):
                error = cost_error("ceres_scalar", "ceres_batch")
                entry.update(median_scalar_over_batch=speed("ceres_scalar", "ceres_batch"),
                             max_B_C_relative_cost_difference=error, B_C_cost_agreement_passes=error <= 1e-7)
            if {"native", "native_batch"} <= set(backends):
                error = cost_error("native", "native_batch")
                entry.update(median_native_over_native_batch=speed("native", "native_batch"),
                             max_A_D_relative_cost_difference=error, A_D_cost_agreement_passes=error <= 1e-7)
            pair_summary.append(entry)
    result = {"groups": summary, "paired": pair_summary}
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = ["# Frozen SLICT A/B/C measurements", "", manifest["timing"] + ".", "",
             "A/D share native damping/clipping/SparseLU; B/C share LM settings. Equal iteration limits across solver families do not imply equal convergence.", "",
             "| Threads | Iteration limit | Backend | N | Mean ms | P95 ms | Final cost mean | Peak RSS MiB |",
             "|---:|---:|---|---:|---:|---:|---:|---:|"]
    for item in summary:
        lines.append(f"| {item['threads']} | {item['iteration_limit']} | {item['backend']} | {item['samples']} | {item['mean_ms']:.3f} | {item['p95_ms']:.3f} | {item['mean_final_cost']:.6g} | {item['peak_rss_mib']:.1f} |")
    lines += ["", "Paired ratios and B/C cost agreement are in `summary.json`; per-window measurements are in `all.csv`."]
    (output / "report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    if any(not pair.get(key, True) for pair in pair_summary for key in
           ("B_C_cost_agreement_passes", "A_D_cost_agreement_passes")):
        raise RuntimeError("Equivalent backends' final costs differ by more than 1e-7; inspect summary.json before interpreting timings")


if __name__ == "__main__":
    main()
