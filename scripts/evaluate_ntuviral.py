#!/usr/bin/env python3
"""Evaluate SLICT cubic splines at NTU VIRAL prism ground-truth timestamps.

Uses the saved control points to evaluate the SO(3) + R3 spline, compensates
the body-to-prism lever arm, and aligns positions by SE(3), without scaling.
The same ground-truth samples and interval are used for A/B/C.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re

import numpy as np
from scipy.spatial.transform import Rotation


BACKENDS = ("native", "ceres_scalar", "ceres_batch")


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_spline(path):
    with Path(path).open() as stream:
        metadata = dict(field.strip().split(":", 1) for field in stream.readline().split(","))
        knots = np.loadtxt(stream, delimiter=",", ndmin=2)
    dt, start = float(metadata["Dt"]), float(metadata["MinTime"])
    if int(metadata["Order"]) != 4 or len(knots) < 4 or knots.shape[1] != 9:
        raise ValueError(f"Expected a cubic SLICT spline: {path}")
    if not np.isfinite(knots).all() or not np.isfinite(dt) or dt <= 0:
        raise ValueError(f"Invalid spline values: {path}")
    if int(metadata["Knots"]) != len(knots) or not np.array_equal(knots[:, 0], np.arange(len(knots))):
        raise ValueError(f"Incomplete knot log: {path}")
    if not np.allclose(knots[:, 1] - start, np.arange(len(knots)) * dt, atol=2e-6, rtol=0):
        raise ValueError(f"Nonuniform knot timestamps: {path}")
    if not np.allclose(np.linalg.norm(knots[:, 5:9], axis=1), 1, atol=1e-6, rtol=0):
        raise ValueError(f"Invalid knot quaternions: {path}")
    return {"start": start, "end": start + (len(knots) - 3) * dt, "dt": dt,
            "positions": knots[:, 2:5], "rotations": Rotation.from_quat(knots[:, 5:9])}


def evaluate_spline(spline, timestamps):
    elapsed = np.asarray(timestamps) - spline["start"]
    # Match basalt::RdSpline::computeTIndex, including its double arithmetic.
    span = np.floor(elapsed / spline["dt"]).astype(int)
    if np.any(span < 0) or np.any(span + 4 > len(spline["positions"])):
        raise ValueError("Timestamp outside spline support")
    u = (elapsed - span * spline["dt"]) / spline["dt"]
    basis = np.column_stack(((1-u)**3, 3*u**3-6*u**2+4,
                             -3*u**3+3*u**2+3*u+1, u**3)) / 6
    position = np.einsum("ni,nij->nj", basis, spline["positions"][span[:, None] + np.arange(4)])
    cumulative = np.cumsum(basis[:, ::-1], axis=1)[:, ::-1]
    rotations = spline["rotations"]
    rotation = rotations[span]
    for i in range(3):
        relative = (rotations[span+i].inv() * rotations[span+i+1]).as_rotvec()
        rotation = rotation * Rotation.from_rotvec(relative * cumulative[:, i+1, None])
    return position, rotation


def load_prism_offset(path):
    # The dataset distributes OpenCV YAML, with a %YAML:1.0 header and matrix tag.
    text = Path(path).read_text()
    matrix = re.search(r"T_Body_Prism:.*?data:\s*\[([^]]+)\]", text, re.DOTALL)
    if matrix is None:
        raise ValueError("Missing T_Body_Prism in calibration")
    values = np.fromstring(matrix[1], sep=",")
    if len(values) != 16 or not np.isfinite(values).all():
        raise ValueError("Invalid prism calibration matrix")
    return values.reshape(4, 4)[:3, 3]


def align_positions(estimated, reference):
    a, b = estimated - estimated.mean(axis=0), reference - reference.mean(axis=0)
    u, singular, vt = np.linalg.svd(a.T @ b)
    if singular[0] <= 0 or singular[1] < singular[0] * 1e-10:
        raise ValueError("Trajectory does not constrain rigid alignment")
    correction = np.diag([1., 1., np.linalg.det(vt.T @ u.T)])
    rotation = vt.T @ correction @ u.T
    translation = reference.mean(axis=0) - rotation @ estimated.mean(axis=0)
    return estimated @ rotation.T + translation, rotation, translation


def load_ground_truth(args):
    if args.ground_truth:
        with args.ground_truth.open() as stream:
            rows = list(csv.DictReader(stream))
        samples = np.array([[float(row[k]) for k in ("timestamp", "x", "y", "z")] for row in rows])
    else:
        import rosbag
        with rosbag.Bag(str(args.bag)) as bag:
            samples = np.array([[m.header.stamp.to_sec(), m.pose.position.x, m.pose.position.y, m.pose.position.z]
                                for _, m, _ in bag.read_messages(topics=["/leica/pose/relative"])])
    if samples.ndim != 2 or samples.shape[1] != 4 or len(samples) < 3 or not np.isfinite(samples).all():
        raise ValueError("Invalid or missing ground truth")
    if np.any(np.diff(samples[:, 0]) <= 0):
        raise ValueError("Ground-truth timestamps must be strictly increasing")
    return samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    truth = parser.add_mutually_exclusive_group(required=True)
    truth.add_argument("--bag", type=Path, help="Original NTU VIRAL bag containing /leica/pose/relative")
    truth.add_argument("--ground-truth", type=Path, help="CSV columns: timestamp,x,y,z (header timestamps, seconds)")
    parser.add_argument("--prism-calibration", type=Path, required=True, help="Dataset leica_prism.yaml")
    parser.add_argument("--replay", type=Path, required=True, help="Directory containing the selected backend subdirectories")
    parser.add_argument("--output", type=Path, required=True, help="New output directory")
    parser.add_argument("--backends", nargs="+", choices=[*BACKENDS, "native_batch"], default=list(BACKENDS))
    args = parser.parse_args()
    if len(set(args.backends)) != len(args.backends):
        parser.error("Duplicate backends")
    backends = args.backends
    truth = load_ground_truth(args)
    offset = load_prism_offset(args.prism_calibration)
    splines, frames = {}, {}
    for backend in backends:
        run = args.replay / backend
        splines[backend] = load_spline(run / "trajectory/spline_log.csv")
        with (run / "frames.csv").open() as stream:
            frames[backend] = np.array([float(row["scan_end_time"]) for row in csv.DictReader(stream)])
        if len(frames[backend]) < 2 or np.any(np.diff(frames[backend]) <= 0):
            raise ValueError(f"Invalid frame timestamps: {backend}")
    if any(len(frames[b]) != len(frames[backends[0]]) or
           not np.allclose(frames[b], frames[backends[0]], rtol=0, atol=1e-6) for b in backends):
        raise ValueError("Backends processed different frame timestamps; compare coverage first")
    start = max(max(s["start"] for s in splines.values()), max(f[0] for f in frames.values()))
    end = min(min(s["end"] - 2e-6 for s in splines.values()), min(f[-1] for f in frames.values()))
    common = truth[(truth[:, 0] >= start) & (truth[:, 0] <= end)]
    if len(common) < 3:
        raise ValueError("Insufficient common ground-truth coverage")
    args.output.mkdir(parents=True, exist_ok=False)
    summary = {"ground_truth_source": str((args.ground_truth or args.bag).resolve()),
               "ground_truth_total": len(truth), "common_samples": len(common),
               "ground_truth_coverage_fraction": len(common) / len(truth),
               "common_start": float(common[0, 0]), "common_end": float(common[-1, 0]),
               "evaluation": "Evaluate cubic SO(3)+R3 spline at GT header timestamps, transform body to prism, rigid SE(3) position alignment; no scale or time-offset fitting",
               "prism_offset_body_m": offset.tolist(), "calibration_sha256": digest(args.prism_calibration),
               "groups": []}
    curves = {}
    for backend in backends:
        body, attitude = evaluate_spline(splines[backend], common[:, 0])
        prism = body + attitude.apply(offset)
        aligned, rotation, translation = align_positions(prism, common[:, 1:4])
        error = aligned - common[:, 1:4]
        norms = np.linalg.norm(error, axis=1)
        entry = {"backend": backend, "samples": len(common), "ate_rmse_m": float(np.sqrt(np.mean(norms**2))),
                 "median_m": float(np.median(norms)), "p95_m": float(np.quantile(norms, .95)),
                 "max_m": float(np.max(norms)), "axis_rmse_m": np.sqrt(np.mean(error**2, axis=0)).tolist(),
                 "alignment_rotation": rotation.tolist(), "alignment_translation": translation.tolist(),
                 "spline_sha256": digest(args.replay / backend / "trajectory/spline_log.csv")}
        summary["groups"].append(entry)
        curves[backend] = (aligned, norms)
        np.savetxt(args.output / f"{backend}_body.tum", np.column_stack((common[:, 0], body, attitude.as_quat())), fmt="%.9f")
        np.savetxt(args.output / f"{backend}_errors.csv", np.column_stack((common[:, 0], aligned, common[:, 1:4], norms)),
                   delimiter=",", header="timestamp,estimate_x,estimate_y,estimate_z,truth_x,truth_y,truth_z,error_m", comments="", fmt="%.9f")
    (args.output / "accuracy.json").write_text(json.dumps(summary, indent=2) + "\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    axes[0].plot(common[:, 1], common[:, 2], "k--", linewidth=1, label="Ground truth", zorder=5)
    labels = {"native": "A", "ceres_scalar": "B", "ceres_batch": "C", "native_batch": "D"}
    for entry in summary["groups"]:
        label = labels[entry["backend"]]
        aligned, error = curves[entry["backend"]]
        legend = f"{label}: {entry['backend']}"
        axes[0].plot(aligned[:, 0], aligned[:, 1], linewidth=1, label=legend)
        axes[1].plot(common[:, 0] - common[0, 0], error, linewidth=.8,
                     label=f"{label}: RMSE {entry['ate_rmse_m']:.4f} m")
    axes[0].set(xlabel="X [m]", ylabel="Y [m]", title="Prism trajectory, rigid alignment", aspect="equal")
    axes[1].set(xlabel="Time since first common GT sample [s]", ylabel="Position error [m]", title="Same ground-truth timestamps for all backends")
    for ax in axes:
        ax.grid(alpha=.25)
        ax.legend(fontsize=8)
    fig.savefig(args.output / "trajectories.png", dpi=180)
    fig.savefig(args.output / "trajectories.pdf")
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
