"""
ECAL MaskFormer Evaluation and Visualization.

Adapted from picocal_reco_transformer/visualize_events_with_geometry.py

Usage:
    # Step 1: Run test to generate h5 predictions
    apptainer exec --nv --bind /data/bfys/shenzh pixi.sif \
        pixi run python -m hepattn.experiments.ecal.main test \
        -c src/hepattn/experiments/ecal/configs/base.yaml \
        --ckpt_path <checkpoint.ckpt>

    # Step 2: Visualize
    apptainer exec --bind /data/bfys/shenzh pixi.sif \
        pixi run python -m hepattn.experiments.ecal.eval \
        --eval_path <eval.h5> \
        --data_dir /data/bfys/shenzh/ECAL_maskformer/data/val \
        --output_dir eval_plots
"""

import argparse
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
from matplotlib.lines import Line2D

from hepattn.utils.scaling import FeatureScaler

ECAL_SCALER = FeatureScaler(str(Path(__file__).resolve().parent / "configs" / "ecal_var_transform.yaml"))

plt.rcParams["text.usetex"] = False
plt.rcParams["figure.dpi"] = 150
plt.rcParams["font.size"] = 10


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -10, 10)))


def load_truth(npz_path):
    """Load ground truth from NPZ file."""
    with np.load(npz_path) as d:
        return {k: d[k] for k in d.files}


def inverse_transform_cluster_field(field, values):
    tensor = torch.as_tensor(values, dtype=torch.float32)
    return ECAL_SCALER.transforms[field].inverse_transform(tensor).cpu().numpy()


def cluster_preds_are_normalized(pred_reg):
    """Detect older eval files that still store normalized ECAL regression outputs."""
    for field in ("x", "y", "z"):
        key = f"cluster_{field}"
        if key not in pred_reg:
            continue

        values = np.asarray(pred_reg[key][0][:])
        finite = np.isfinite(values)
        if finite.any():
            return np.nanmax(np.abs(values[finite])) < 10.0

    return False


def read_cluster_regression_predictions(pred_reg):
    pred_is_normalized = cluster_preds_are_normalized(pred_reg)
    preds = {}

    for field in ("e", "x", "y"):
        key = f"cluster_{field}"
        if key not in pred_reg:
            continue

        values = pred_reg[key][0][:]
        if pred_is_normalized:
            values = inverse_transform_cluster_field(field, values)
        preds[f"pred_{field}"] = values

    return preds


def get_module_color(module_type):
    """Get distinct color for each module type (same as picocal)."""
    colors = {
        1: "#e41a1c",
        2: "#377eb8",
        3: "#4daf4a",
        4: "#984ea3",
        5: "#ff7f00",
        6: "#ffff33",
        7: "#a65628",
    }
    return colors.get(module_type, "#999999")


def draw_detector_geometry(ax, module_info, alpha=0.15):
    """Draw detector geometry as background (from pickle file)."""
    for mod in module_info:
        rect = patches.Rectangle(
            (mod["x_bl"], mod["y_bl"]),
            mod["dx"],
            mod["dy"],
            linewidth=0.5,
            edgecolor="black",
            facecolor=mod["color"],
            alpha=alpha,
        )
        ax.add_patch(rect)


def load_detector_geometry(pickle_path):
    """Load detector geometry from pickle file."""
    import pickle

    with open(pickle_path, "rb") as f:
        data = pickle.load(f)

    module_info = []
    for mod in data["modules"]:
        module_info.append(
            {
                "type": mod["type"],
                "x": mod["x"],
                "y": mod["y"],
                "dx": mod["dx"],
                "dy": mod["dy"],
                "x_bl": mod["x_bl"],
                "y_bl": mod["y_bl"],
                "color": get_module_color(mod["type"]),
            }
        )

    print(f"Loaded geometry: {len(module_info)} modules from {pickle_path}")
    return module_info


def visualize_event(truth, preds, event_idx, output_dir, module_info=None):
    """
    Visualize a single event: predictions vs ground truth.

    Style follows picocal_reco_transformer/visualize_events_with_geometry.py:
    - Detector geometry as light background (if available)
    - Cell hits as light gray dots
    - Ground truth clusters as colored squares with energy annotation
    - Predicted clusters as colored stars with energy + confidence annotation
    """
    # Calorimeter bounds
    x_min, x_max = -3200, 3200
    y_min, y_max = -2200, 2200

    fig, ax = plt.subplots(1, 1, figsize=(14, 12))

    cell_x = truth["cell.x"]
    cell_y = truth["cell.y"]
    num_cells = len(cell_x)

    true_valid = truth["cluster_valid"]
    true_assign = truth["cluster_cell_valid"]

    pred_valid = preds["cluster_valid"]
    pred_assign = preds["cluster_cell_valid"][:, :num_cells]

    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    # Draw detector geometry background (if available)
    if module_info is not None:
        draw_detector_geometry(ax, module_info, alpha=0.15)

    # Draw all hits (light gray)
    ax.scatter(cell_x, cell_y, c="lightgray", s=20, alpha=0.4, label=f"Hits ({num_cells})")

    # Draw cell assignments colored by predicted cluster
    pred_count = 0
    for i in range(len(pred_valid)):
        if not pred_valid[i]:
            continue
        mask = pred_assign[i]
        if mask.sum() == 0:
            continue
        color = colors[pred_count % len(colors)]
        ax.scatter(cell_x[mask], cell_y[mask], c=[color], s=25, alpha=0.6, zorder=5)
        pred_count += 1

    # Ground truth clusters (squares)
    gt_count = 0
    for i in range(len(true_valid)):
        if not true_valid[i]:
            continue
        color = colors[gt_count % len(colors)]
        cx, cy = truth["cluster.x"][i], truth["cluster.y"][i]
        e = truth["cluster.e"][i]
        ax.scatter(
            cx, cy, c=[color], s=600, marker="s", edgecolors="black", linewidth=2.5, alpha=0.85, zorder=10
        )
        ax.annotate(
            f"True {gt_count + 1}\nE={e:.0f} MeV",
            xy=(cx, cy),
            xytext=(10, 10),
            textcoords="offset points",
            fontsize=10,
            color=color,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9),
        )
        gt_count += 1

    # Predicted clusters (stars)
    pred_count = 0
    for i in range(len(pred_valid)):
        if not pred_valid[i]:
            continue
        color = colors[pred_count % len(colors)]

        # Use regression output for position/energy if available, else use cell centroid
        pred_x = preds.get("pred_x", None)
        pred_y = preds.get("pred_y", None)
        pred_e = preds.get("pred_e", None)
        conf = preds.get("confidence", None)

        if pred_x is not None:
            px, py = pred_x[i], pred_y[i]
        else:
            # Fallback: centroid of assigned cells
            mask = pred_assign[i]
            if mask.sum() > 0:
                px, py = cell_x[mask].mean(), cell_y[mask].mean()
            else:
                pred_count += 1
                continue

        ax.scatter(
            px, py, c=[color], s=500, marker="*", edgecolors="black", linewidth=2.5, alpha=0.9, zorder=11
        )

        label = f"Pred {pred_count + 1}"
        if pred_e is not None:
            label += f"\nE={pred_e[i]:.0f} MeV"
        if conf is not None:
            label += f"\nconf={conf[i]:.2f}"

        offset_y = 25 if pred_count % 2 == 0 else -35
        ax.annotate(
            label,
            xy=(px, py),
            xytext=(10, offset_y),
            textcoords="offset points",
            fontsize=10,
            color="darkred",
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9, edgecolor="red"),
        )
        pred_count += 1

    # Legend
    legend_elements = [
        Line2D([0], [0], marker="s", color="w", markerfacecolor="gray", markersize=12, label=f"Ground Truth ({gt_count})"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor="red", markersize=15, label=f"Prediction ({pred_count})"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="lightgray", markersize=8, label=f"Hits ({num_cells})"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=10)

    ax.set_xlabel("X (mm)", fontsize=14)
    ax.set_ylabel("Y (mm)", fontsize=14)
    title = f"Event {event_idx}: Predictions vs Ground Truth\n"
    title += f"Hits: {num_cells}, Predicted: {pred_count}, True: {gt_count}"
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_aspect("equal")
    ax.set_xlim(x_min - 100, x_max + 100)
    ax.set_ylim(y_min - 100, y_max + 100)

    plt.tight_layout()

    output_path = Path(output_dir) / f"event_{event_idx:03d}_pred_vs_gt.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"  Saved {output_path}")
    plt.close()


def plot_summary(all_true_counts, all_pred_counts, all_true_e, all_pred_e, all_effs, all_purs, output_dir):
    """Generate summary comparison plots."""
    output_dir = Path(output_dir)

    # --- Cluster count ---
    if len(all_true_counts) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        max_count = max(max(all_true_counts), max(all_pred_counts)) + 2
        bins = np.arange(-0.5, max_count + 0.5, 1)

        axes[0].hist(all_true_counts, bins=bins, alpha=0.7, label="True", color="steelblue")
        axes[0].hist(all_pred_counts, bins=bins, alpha=0.7, label="Predicted", color="coral")
        axes[0].set_xlabel("Number of clusters")
        axes[0].set_ylabel("Events")
        axes[0].set_title("Cluster Count Distribution")
        axes[0].legend()
        axes[0].grid(alpha=0.2)

        axes[1].scatter(all_true_counts, all_pred_counts, alpha=0.5, s=30)
        axes[1].plot([0, max_count], [0, max_count], "k--", alpha=0.5)
        axes[1].set_xlabel("True clusters")
        axes[1].set_ylabel("Predicted clusters")
        axes[1].set_title("True vs Predicted")
        axes[1].set_aspect("equal")
        axes[1].grid(alpha=0.2)

        plt.tight_layout()
        fig.savefig(output_dir / "cluster_count.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved cluster_count.png")

    # --- Energy comparison ---
    if len(all_true_e) > 0 and len(all_pred_e) > 0:
        true_e = np.array(all_true_e)
        pred_e = np.array(all_pred_e)
        residual = (pred_e - true_e) / np.clip(true_e, 1e-3, None)

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        axes[0].scatter(true_e, pred_e, alpha=0.3, s=15)
        emax = max(true_e.max(), pred_e.max()) * 1.1
        axes[0].plot([0, emax], [0, emax], "k--", alpha=0.5)
        axes[0].set_xlabel("True Energy [MeV]")
        axes[0].set_ylabel("Predicted Energy [MeV]")
        axes[0].set_title("Energy Correlation")
        axes[0].grid(alpha=0.2)

        axes[1].hist(residual, bins=50, range=(-2, 2), color="steelblue", alpha=0.7)
        axes[1].axvline(0, color="k", linestyle="--", alpha=0.5)
        axes[1].set_xlabel("(Pred - True) / True")
        axes[1].set_ylabel("Count")
        axes[1].set_title(f"Energy Residual\nmean={residual.mean():.3f}, std={residual.std():.3f}")
        axes[1].grid(alpha=0.2)

        axes[2].scatter(true_e, residual, alpha=0.3, s=15)
        axes[2].axhline(0, color="k", linestyle="--", alpha=0.5)
        axes[2].set_xlabel("True Energy [MeV]")
        axes[2].set_ylabel("(Pred - True) / True")
        axes[2].set_title("Residual vs Energy")
        axes[2].set_ylim(-2, 2)
        axes[2].grid(alpha=0.2)

        plt.tight_layout()
        fig.savefig(output_dir / "energy_comparison.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved energy_comparison.png")

    # --- Cell assignment efficiency / purity ---
    if len(all_effs) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        axes[0].hist(all_effs, bins=20, range=(0, 1), alpha=0.7, color="steelblue")
        axes[0].set_xlabel("Cell Assignment Efficiency")
        axes[0].set_ylabel("Count")
        axes[0].set_title(f"Efficiency (mean={np.mean(all_effs):.3f})")
        axes[0].grid(alpha=0.2)

        axes[1].hist(all_purs, bins=20, range=(0, 1), alpha=0.7, color="coral")
        axes[1].set_xlabel("Cell Assignment Purity")
        axes[1].set_ylabel("Count")
        axes[1].set_title(f"Purity (mean={np.mean(all_purs):.3f})")
        axes[1].grid(alpha=0.2)

        plt.tight_layout()
        fig.savefig(output_dir / "cell_assignment.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved cell_assignment.png")


def match_clusters(true_assign, pred_assign, true_valid, pred_valid):
    """Match predicted clusters to true clusters by IoU of cell assignments."""
    matches = []
    used_preds = set()

    for ti in range(len(true_valid)):
        if not true_valid[ti]:
            continue
        best_iou = 0
        best_pi = -1
        for pi in range(len(pred_valid)):
            if not pred_valid[pi] or pi in used_preds:
                continue
            intersection = (true_assign[ti] & pred_assign[pi]).sum()
            union = (true_assign[ti] | pred_assign[pi]).sum()
            if union == 0:
                continue
            iou = intersection / union
            if iou > best_iou:
                best_iou = iou
                best_pi = pi
        if best_pi >= 0 and best_iou > 0.1:
            matches.append((ti, best_pi, best_iou))
            used_preds.add(best_pi)

    return matches


def main():
    parser = argparse.ArgumentParser(description="ECAL MaskFormer Evaluation and Visualization")
    parser.add_argument("--eval_path", required=True, help="Path to evaluation h5 file from test step")
    parser.add_argument("--data_dir", required=True, help="Path to NPZ data directory used for test")
    parser.add_argument("--output_dir", default="eval_plots", help="Output directory for plots")
    parser.add_argument("--num_events", type=int, default=10, help="Number of event displays to generate")
    parser.add_argument("--existence_threshold", type=float, default=0.5, help="Threshold for cluster existence")
    parser.add_argument("--geometry_pickle", default=None, help="Path to geometry pickle file (optional)")
    args = parser.parse_args()

    print("=" * 70)
    print("ECAL MaskFormer Evaluation")
    print("=" * 70)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)

    # Load detector geometry if available
    module_info = None
    if args.geometry_pickle and Path(args.geometry_pickle).exists():
        module_info = load_detector_geometry(args.geometry_pickle)

    # Load eval h5
    eval_file = h5py.File(args.eval_path, "r")
    sample_ids = list(eval_file.keys())
    print(f"Found {len(sample_ids)} events in {args.eval_path}")

    # Build sample_id -> npz path mapping
    npz_files = sorted(data_dir.rglob("reco_ecal_*.npz"))
    id_to_npz = {}
    for f in npz_files:
        parts = f.stem.replace("reco_ecal_", "").split("_")
        sid = int(parts[0]) * 100000 + int(parts[1])
        id_to_npz[str(sid)] = f
    print(f"Found {len(id_to_npz)} NPZ files in {data_dir}")

    # Accumulators for summary plots
    all_true_counts, all_pred_counts = [], []
    all_true_e, all_pred_e = [], []
    all_effs, all_purs = [], []

    print(f"\nProcessing events...")
    print("=" * 70)

    for idx, sample_id in enumerate(sample_ids):
        if sample_id not in id_to_npz:
            print(f"  Warning: sample {sample_id} not found in data dir, skipping")
            continue

        truth = load_truth(id_to_npz[sample_id])
        num_cells = len(truth["cell.x"])

        # Extract predictions from h5
        final_preds = eval_file[sample_id]["preds"]["final"]

        # Cluster validity (already thresholded bool + probability)
        pred_valid = final_preds["cluster_valid"]["cluster_valid"][0][:]
        pred_probs = final_preds["cluster_valid"]["cluster_valid_prob"][0][:]

        # Cell assignment (already thresholded bool)
        pred_assign = final_preds["cluster_cell_assignment"]["cluster_cell_valid"][0][:]

        true_valid = truth["cluster_valid"]
        true_assign = truth["cluster_cell_valid"]

        # Count clusters
        all_true_counts.append(int(true_valid.sum()))
        all_pred_counts.append(int(pred_valid.sum()))

        # Build preds dict for visualization
        preds_dict = {
            "cluster_valid": pred_valid,
            "cluster_cell_valid": pred_assign,
            "confidence": pred_probs,
        }

        # Add regression predictions if available. New eval files are already
        # inverse-transformed by the ECAL prediction writer; older files are
        # detected and converted here for backward compatibility.
        if "cluster_regression" in final_preds:
            preds_dict.update(read_cluster_regression_predictions(final_preds["cluster_regression"]))

        # Match clusters by cell assignment IoU
        matches = match_clusters(true_assign, pred_assign[:, :num_cells], true_valid, pred_valid)

        for ti, pi, iou in matches:
            all_true_e.append(truth["cluster.e"][ti])
            if "pred_e" in preds_dict:
                all_pred_e.append(preds_dict["pred_e"][pi])

            tp = (true_assign[ti] & pred_assign[pi, :num_cells]).sum()
            all_effs.append(tp / max(true_assign[ti].sum(), 1))
            all_purs.append(tp / max(pred_assign[pi, :num_cells].sum(), 1))

        # Event display for first N events
        if idx < args.num_events:
            print(f"\nEvent {idx} (sample {sample_id}): {num_cells} cells, "
                  f"{true_valid.sum()} true clusters, {pred_valid.sum()} pred clusters, "
                  f"{len(matches)} matched")
            visualize_event(truth, preds_dict, idx, output_dir, module_info)

    eval_file.close()

    # Summary plots
    print(f"\n{'=' * 70}")
    print("Generating summary plots...")
    plot_summary(all_true_counts, all_pred_counts, all_true_e, all_pred_e, all_effs, all_purs, output_dir)

    # Print summary statistics
    print(f"\n{'=' * 70}")
    print(f"Evaluation Summary ({len(sample_ids)} events)")
    print(f"{'=' * 70}")
    print(f"  True clusters/event:  {np.mean(all_true_counts):.2f} +/- {np.std(all_true_counts):.2f}")
    print(f"  Pred clusters/event:  {np.mean(all_pred_counts):.2f} +/- {np.std(all_pred_counts):.2f}")
    if all_effs:
        print(f"  Cell efficiency:      {np.mean(all_effs):.3f} +/- {np.std(all_effs):.3f}")
        print(f"  Cell purity:          {np.mean(all_purs):.3f} +/- {np.std(all_purs):.3f}")
    if all_true_e and all_pred_e:
        residual = (np.array(all_pred_e) - np.array(all_true_e)) / np.clip(all_true_e, 1e-3, None)
        print(f"  Energy residual:      {np.mean(residual):.3f} +/- {np.std(residual):.3f}")
    print(f"\nAll plots saved to: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
