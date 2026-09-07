#!/usr/bin/env python3
"""Run A/B/C with one solver step per update/deskew/association cycle."""
import argparse
from collections import defaultdict
import csv
import json
import math
import os
from pathlib import Path
import signal
import re
import socket
import statistics
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bag", type=Path, required=True)
    parser.add_argument("--dataset", choices=["ntuviral", "r3live"], required=True,
                        help="Sensor topics/calibration; ntuviral uses the paper's horizontal lidar")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backends", nargs="+", choices=["native", "ceres_scalar", "ceres_batch", "native_batch"],
                        default=["native", "ceres_scalar", "ceres_batch"])
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--outer-iterations", type=int, default=3,
                        help="Common solve/update/deskew/associate cycles per frame (each solver call is one step)")
    parser.add_argument("--duration", type=float, default=30, help="Seconds of bag data; 0 uses the full bag")
    parser.add_argument("--rate", type=float, default=0.5)
    parser.add_argument("--snapshot-stride", type=int, default=2)
    parser.add_argument("--snapshot-limit", type=int, default=500)
    parser.add_argument("--no-snapshots", action="store_true", help="Disable capture for comparable full-frame timing")
    parser.add_argument("--max-empty-lidar-frames", type=int, default=5,
                        help="Fail after more consecutive frames without LiDAR constraints")
    parser.add_argument("--timeout", type=float, default=600)
    args = parser.parse_args()
    if not args.bag.is_file() or args.threads < 1 or args.outer_iterations < 1 or args.rate <= 0:
        parser.error("A readable bag and positive threads/outer-iterations/rate are required")
    if args.duration < 0 or args.snapshot_stride < 1 or args.snapshot_limit < 0 or args.timeout <= 0 or args.max_empty_lidar_frames < 0:
        parser.error("Invalid duration/snapshot/timeout option")
    if len(set(args.backends)) != len(args.backends):
        parser.error("Duplicate backends")
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    records = []
    reference_stamps = None
    for backend in args.backends:
        run = root / backend
        run.mkdir()
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        env = os.environ.copy()
        env.update(ROS_MASTER_URI=f"http://127.0.0.1:{port}", ROS_IP="127.0.0.1",
                   ROS_HOME=str(run / "ros_home"), ROS_LOG_DIR=str(run / "ros_logs"),
                   OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", OMP_NUM_THREADS="1",
                   OMP_DYNAMIC="FALSE", OMP_MAX_ACTIVE_LEVELS="1")
        env.pop("ROS_HOSTNAME", None)
        command = ["roslaunch", "--port", str(port), "slict", "run_abc.launch",
                   f"bag_file:={args.bag.resolve()}", f"backend:={backend}",
                   f"dataset:={args.dataset}", f"threads:={args.threads}",
                   f"outer_iterations:={args.outer_iterations}",
                   f"output_dir:={run}", f"duration:={args.duration}", f"rate:={args.rate}",
                   f"snapshot_stride:={args.snapshot_stride}", f"snapshot_limit:={args.snapshot_limit}"]
        if backend == "native" and not args.no_snapshots:
            command.append(f"snapshot_dir:={root / 'snapshots'}")
        print(f"Starting {backend}: private ROS master port {port}", flush=True)
        start = time.monotonic()
        timed_out = False
        with (run / "roslaunch.log").open("w") as log:
            process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            try:
                code = process.wait(timeout=args.timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
            finally:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGINT)
                    try:
                        process.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGTERM)
                        process.wait(timeout=10)
                code = process.returncode
        path = run / "optimization.csv"
        rows = list(csv.DictReader(path.open())) if path.exists() else []
        failures = sum(row["usable"] != "1" for row in rows)
        final_rows = [row for row in rows if row["outer_iteration"] == "0"]
        empty_lidar = [int(row["lidar_observations"]) == 0 for row in final_rows]
        streak = max_empty_streak = 0
        for empty in empty_lidar:
            streak = streak + 1 if empty else 0
            max_empty_streak = max(max_empty_streak, streak)
        launch_log = (run / "roslaunch.log").read_text(errors="replace")
        fatal_exits = re.findall(r"process has died.*exit code (-?\d+)", launch_log)
        frame_path = run / "frames.csv"
        frame_rows = list(csv.DictReader(frame_path.open())) if frame_path.exists() else []
        cycles = defaultdict(list)
        for row in rows:
            cycles[row["frame"]].append(row)
        expected_order = list(range(args.outer_iterations - 1, -1, -1))
        cadence_ok = bool(rows) and bool(frame_rows) and all(
            [int(row["outer_iteration"]) for row in cycle] == expected_order
            and all(int(row["iteration_limit"]) == 1 and int(row["iterations"]) <= 1 for row in cycle)
            for cycle in cycles.values()) and all(
                all(int(row.get(key, -1)) == args.outer_iterations for key in
                    ("optimizer_calls", "post_opt_deskew_passes", "post_opt_association_passes"))
                for row in frame_rows)
        radius_continuity = backend in ("native", "native_batch") or (bool(rows) and all(
            float(row.get("ceres_final_trust_region_radius", 0)) > 0 for row in rows) and all(
            math.isclose(float(current.get("ceres_initial_trust_region_radius", 0)),
                         float(previous.get("ceres_final_trust_region_radius", 0)), rel_tol=1e-12)
            for previous, current in zip(rows, rows[1:])))
        frame_times = sorted(float(row["frame_ms"]) for row in frame_rows)
        stamps = [float(row["scan_end_time"]) for row in frame_rows]
        gaps = [b - a for a, b in zip(stamps, stamps[1:])]
        matched = reference_stamps is None or (len(stamps) == len(reference_stamps) and
                  all(abs(a - b) < 1e-6 for a, b in zip(stamps, reference_stamps)))
        regular = bool(gaps) and min(gaps) > 0 and max(gaps) < 1.5 * statistics.median(gaps)
        coverage = max(float(row["window_start_time"]) for row in rows) - min(float(row["window_start_time"]) for row in rows) if rows else 0
        record = {"backend": backend, "dataset": args.dataset, "iteration_limit": 1,
                  "outer_iterations": args.outer_iterations, "single_step_association_cadence": cadence_ok,
                  "ceres_radius_continuity": radius_continuity,
                  "returncode": code, "wall_seconds": time.monotonic() - start,
                  "optimization_rows": len(rows), "frames": len({row["frame"] for row in rows}),
                  "unusable": failures, "coverage_seconds": coverage, "fatal_node_exit_codes": fatal_exits,
                  "empty_lidar_frames": sum(empty_lidar), "max_empty_lidar_streak": max_empty_streak,
                  "captures_snapshots": backend == "native" and not args.no_snapshots,
                  "timed_out": timed_out, "frame_csv_rows": len(frame_rows), "same_scan_timestamps": matched,
                  "regular_scan_intervals": regular,
                  "frame_mean_ms": statistics.mean(frame_times) if frame_times else None,
                  "frame_p95_ms": frame_times[math.ceil(.95 * len(frame_times)) - 1] if frame_times else None,
                  "max_queued_packets": max((int(row["queued_packets"]) for row in frame_rows), default=0),
                  "command": command, "ros_master_uri": env["ROS_MASTER_URI"]}
        records.append(record)
        (root / "replay_manifest.json").write_text(json.dumps(records, indent=2) + "\n")
        print(json.dumps(record), flush=True)
        if code or timed_out or not rows or failures or fatal_exits or "terminate called" in launch_log:
            raise RuntimeError(f"{backend} failed; inspect {run / 'roslaunch.log'}")
        if not matched or not regular or len(frame_rows) != record["frames"]:
            raise RuntimeError(f"{backend} frame coverage is incomplete or differs from the first backend")
        if not cadence_ok:
            raise RuntimeError(f"{backend} did not follow one solve step per update/deskew/association cycle")
        if not radius_continuity:
            raise RuntimeError(f"{backend} reset Ceres trust-region state between single-step calls")
        if max_empty_streak > args.max_empty_lidar_frames:
            raise RuntimeError(f"{backend} lost LiDAR constraints for {max_empty_streak} consecutive frames; timing is not a successful SLAM result")
        if args.duration > 5 and coverage < args.duration - 4:
            raise RuntimeError(f"{backend} processed only {coverage:.2f}s of the requested {args.duration}s")
        reference_stamps = stamps
    print(f"Replays complete: {root}", flush=True)


if __name__ == "__main__":
    main()
