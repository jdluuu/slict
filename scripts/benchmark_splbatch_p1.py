#!/usr/bin/env python3
"""Compare preserved and current libraries in isolated, rotated runs."""
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


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True,
                        help="Preserved repository tree containing devel/lib and source hashes")
    parser.add_argument("--snapshots", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 4, 8, 16])
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--max-snapshots", type=int, default=101)
    parser.add_argument("--validate", type=int, default=101)
    parser.add_argument("--after-label", choices=["p1", "p2"], default="p1")
    args = parser.parse_args()
    after = args.after_label
    if min(args.threads + [args.rounds, args.max_snapshots]) < 1 or args.validate < 0:
        parser.error("Invalid counts")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    common = os.environ.copy()
    common.update(OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", OMP_NUM_THREADS="1",
                  OMP_DYNAMIC="FALSE", OMP_MAX_ACTIVE_LEVELS="1")
    variants = {}
    manifest = {"platform": platform.platform(), "cpu_affinity": sorted(os.sched_getaffinity(0)),
                "threads": args.threads, "rounds": args.rounds, "variants": {}, "runs": [],
                "controlled_environment": {k: common[k] for k in ["OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS", "OMP_NUM_THREADS", "OMP_DYNAMIC", "OMP_MAX_ACTIVE_LEVELS"]},
                "timing": "No construction probes. Independent processes, full warmup pass, one solver step; rotated A, before/after C and before/after D. No overlapping measured processes."}
    files = sorted(args.snapshots.glob("*.slict"))[:args.max_snapshots]
    if not files:
        parser.error("No snapshots")
    manifest["snapshots"] = {str(p.resolve()): digest(p) for p in files}
    for name, root in [("before", args.before.resolve()), (after, repo)]:
        binary = root / "devel/lib/slict/slict_solver_benchmark"
        library = root / "devel/lib/libslict_comparison.so"
        env = common.copy()
        env["LD_LIBRARY_PATH"] = str(library.parent) + os.pathsep + common.get("LD_LIBRARY_PATH", "")
        linkage = subprocess.check_output(["ldd", str(binary)], env=env, text=True)
        (output / f"{name}_linkage.txt").write_text(linkage)
        loaded = [line.split("=>", 1)[1].split()[0] for line in linkage.splitlines()
                  if "libslict_comparison.so =>" in line]
        if loaded != [str(library)]:
            raise RuntimeError(f"{name} loads wrong comparison library: {loaded}")
        symbols = subprocess.check_output(["nm", "-C", str(library)], text=True)
        if "construction_diagnostics::" in symbols:
            raise RuntimeError("Formal benchmark library contains construction probes")
        paths = list((root / "src/comparison").glob("*"))
        paths += list((root / "third_party/splbatch/include/splbatch").glob("*.hpp"))
        manifest["variants"][name] = {"binary": str(binary), "binary_sha256": digest(binary),
            "library": str(library), "library_sha256": digest(library),
            "source_sha256": {str(p.relative_to(root)): digest(p) for p in paths if p.is_file()}}
        variants[name] = (binary, env)
    if manifest["variants"]["before"]["library_sha256"] == manifest["variants"][after]["library_sha256"]:
        raise RuntimeError("Before and after libraries are identical")
    rows = []

    def run(version, backend, threads, round_id, validate=0):
        binary, env = variants[version]
        name = f"{version}_{backend}_t{threads}_r{round_id}"
        path = output / f"{name}.csv"
        command = [str(binary), "--snapshots", str(args.snapshots.resolve()),
                   "--backend", backend, "--threads", str(threads), "--iterations", "1",
                   "--repeat", "1", "--warmup", "0" if validate else "1",
                   "--validate", str(validate), "--max-snapshots", str(args.max_snapshots),
                   "--output", str(path)]
        print(name, flush=True)
        with (output / f"{name}.log").open("w") as log:
            subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        with path.open() as stream:
            samples = list(csv.DictReader(stream))
        if len(samples) != len(files) or any(r["usable"] != "1" for r in samples):
            raise RuntimeError("Missing or unusable snapshots")
        manifest["runs"].append({"command": command, "rows": len(samples),
                                  "csv_sha256": digest(path), "validation": bool(validate)})
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        if not validate:
            for row in samples:
                row.update(version=version, round=round_id)
            rows.extend(samples)

    if args.validate:
        run(after, "native_batch", 4, -1, args.validate)
    jobs = [("before", "native"), ("before", "ceres_batch"), (after, "ceres_batch"),
            ("before", "native_batch"), (after, "native_batch")]
    for threads in args.threads:
        for r in range(args.rounds):
            shift = r % len(jobs)
            order = jobs[shift:] + jobs[:shift]
            if r % 2:
                order = list(reversed(order))
            for version, backend in order:
                run(version, backend, threads, r)
    with (output / "all.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    groups, paired = {}, {}
    for row in rows:
        groups.setdefault((int(row["threads"]), row["backend"], row["version"]), []).append(row)
        key = (row["threads"], row["snapshot"], row["round"])
        pair = paired.setdefault(key, {})
        tag = (row["version"], row["backend"])
        if tag in pair:
            raise RuntimeError("Duplicate sample")
        pair[tag] = row
    if any(set(pair) != set(jobs) for pair in paired.values()):
        raise RuntimeError("Unpaired snapshot coverage")
    summary = {"groups": [], "paired": []}
    for (threads, backend, version), samples in sorted(groups.items()):
        item = dict(threads=threads, backend=backend, version=version, samples=len(samples))
        for metric in ["build_ms", "evaluate_ms", "assemble_ms", "linear_ms", "solve_ms",
                       "destroy_ms", "total_ms", "diagnostic_ms"]:
            values = sorted(float(r[metric]) for r in samples)
            item[metric] = {"mean": statistics.mean(values), "p95": values[math.ceil(.95 * len(values)) - 1]}
        item["peak_rss_mib"] = max(int(r["process_peak_rss_kib"]) for r in samples) / 1024
        summary["groups"].append(item)
    for threads in args.threads:
        pairs = [p for k, p in paired.items() if int(k[0]) == threads]
        for backend in ["native_batch", "ceres_batch"]:
            error = max(abs(float(p["before", backend]["final_cost"]) - float(p[after, backend]["final_cost"])) /
                        max(1., abs(float(p["before", backend]["final_cost"]))) for p in pairs)
            if error > 1e-7:
                raise RuntimeError("Before/after final costs differ")
            item = dict(threads=threads, backend=backend, max_relative_cost_difference=error)
            for metric in ["build_ms", "total_ms"]:
                before = statistics.mean(float(p["before", backend][metric]) for p in pairs)
                current = statistics.mean(float(p[after, backend][metric]) for p in pairs)
                item[metric + "_reduction_percent"] = 100 * (1 - current / before)
            summary["paired"].append(item)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["paired"], indent=2))


if __name__ == "__main__":
    main()
