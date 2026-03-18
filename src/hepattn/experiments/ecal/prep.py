"""
Preprocessing script for ECAL data: ROOT → NPZ

Reads OutTrigd_*.root files using TCellReco/TGeometry from the reconstruction
framework and produces per-event NPZ files compatible with LRSMDataset.

Each NPZ file contains:
  Cell features (input):
    cell.eF         - Front section calibrated energy [MeV]
    cell.eB         - Back section calibrated energy [MeV]
    cell.log_eF     - log(1 + eF) for large dynamic range
    cell.log_eB     - log(1 + eB)
    cell.x          - Cell x position [mm]
    cell.y          - Cell y position [mm]
    cell.z          - Cell z position (constant 12620 mm)
    cell.tF         - Front section timing [ns]
    cell.tB         - Back section timing [ns]
    cell.dx         - Cell x size [mm]
    cell.dy         - Cell y size [mm]
    cell.region     - Module region ID (1-7)

  Cluster targets (from traditional reconstruction):
    cluster.e       - Total cluster energy [MeV]
    cluster.eF      - Front cluster energy [MeV]
    cluster.eB      - Back cluster energy [MeV]
    cluster.x       - Cluster x position [mm]
    cluster.y       - Cluster y position [mm]
    cluster.z       - Cluster z position [mm]
    cluster.t       - Cluster time [ns]
    cluster.tF      - Front cluster time [ns]
    cluster.tB      - Back cluster time [ns]
    cluster_valid   - Boolean mask of valid clusters

  Cell-to-cluster assignment:
    cluster_cell_valid - [num_clusters, num_cells] boolean assignment matrix

Usage:
    # Source CVMFS environment first:
    # source /cvmfs/sft.cern.ch/lcg/views/LCG_105_LHCB_7/x86_64-el9-gcc12-opt/setup.sh

    python prep.py --input-dir /path/to/root/files \\
                   --output-dir /path/to/output/npz \\
                   --lumi-condition Run5_rotated_2023_baseline \\
                   --geometry-file /path/to/ModuleInfo_Run5_rotated_2023_baseline.root \\
                   --seeding 0 \\
                   --max-events 1000
"""

import argparse
import glob
import math
import os
import sys

import numpy as np


def prep_event(tree, event_idx, geometry, TCalorimeter, TCellReco, seeding=0):
    """Process a single event and return a dict of numpy arrays."""
    tree.GetEntry(event_idx)

    # Build cell data
    cell_reco = TCellReco(tree, event_idx, geometry=geometry)
    hit_cells = cell_reco.getHitCells()

    if len(hit_cells) == 0:
        return None

    # Get clusters from traditional reconstruction
    calo = TCalorimeter(tree, event_idx, seeding=seeding, geometry=geometry)
    clusters = calo.getClusters(2)

    if len(clusters) == 0:
        return None

    num_cells = len(hit_cells)
    num_clusters = len(clusters)

    # Extract cell features
    cell_eF = np.zeros(num_cells, dtype=np.float32)
    cell_eB = np.zeros(num_cells, dtype=np.float32)
    cell_x = np.zeros(num_cells, dtype=np.float32)
    cell_y = np.zeros(num_cells, dtype=np.float32)
    cell_z = np.zeros(num_cells, dtype=np.float32)
    cell_tF = np.zeros(num_cells, dtype=np.float32)
    cell_tB = np.zeros(num_cells, dtype=np.float32)
    cell_dx = np.zeros(num_cells, dtype=np.float32)
    cell_dy = np.zeros(num_cells, dtype=np.float32)
    cell_region = np.zeros(num_cells, dtype=np.float32)
    cell_ids = np.zeros(num_cells, dtype=np.int32)

    for i, cell in enumerate(hit_cells):
        cell_eF[i] = max(cell.getEF(), 0.0)
        cell_eB[i] = max(cell.getEB(), 0.0)
        cell_x[i] = cell.getX()
        cell_y[i] = cell.getY()
        cell_z[i] = 12620.0
        cell_tF[i] = max(cell.getTF(), 0.0) if cell.getTF() > 0 else 0.0
        cell_tB[i] = max(cell.getTB(), 0.0) if cell.getTB() > 0 else 0.0
        cell_dx[i] = cell.getDx()
        cell_dy[i] = cell.getDy()
        module = cell.getModule()
        cell_region[i] = float(module.getRegion()) if module else 1.0
        cell_ids[i] = cell.getID()

    # Log-scale energies
    cell_log_eF = np.log1p(cell_eF)
    cell_log_eB = np.log1p(cell_eB)

    # Extract cluster properties
    cluster_e = np.zeros(num_clusters, dtype=np.float32)
    cluster_eF = np.zeros(num_clusters, dtype=np.float32)
    cluster_eB = np.zeros(num_clusters, dtype=np.float32)
    cluster_x = np.zeros(num_clusters, dtype=np.float32)
    cluster_y = np.zeros(num_clusters, dtype=np.float32)
    cluster_z = np.zeros(num_clusters, dtype=np.float32)
    cluster_t = np.zeros(num_clusters, dtype=np.float32)
    cluster_tF = np.zeros(num_clusters, dtype=np.float32)
    cluster_tB = np.zeros(num_clusters, dtype=np.float32)
    cluster_valid = np.ones(num_clusters, dtype=bool)

    # Build cell-to-cluster assignment matrix
    # Map cell ID -> index for fast lookup
    cell_id_to_idx = {cell_ids[i]: i for i in range(num_cells)}

    cluster_cell_valid = np.zeros((num_clusters, num_cells), dtype=bool)

    for j, cluster in enumerate(clusters):
        cluster_e[j] = cluster.getE()
        cluster_eF[j] = cluster.getEF()
        cluster_eB[j] = cluster.getEB()
        cluster_x[j] = cluster.getX()
        cluster_y[j] = cluster.getY()
        cluster_z[j] = cluster.getZ()
        cluster_t[j] = cluster.getT()
        cluster_tF[j] = cluster.getTF()
        cluster_tB[j] = cluster.getTB()

        # Mark which cells belong to this cluster
        for cell in cluster.getCells():
            cid = cell.getID()
            if cid in cell_id_to_idx:
                cluster_cell_valid[j, cell_id_to_idx[cid]] = True

    # Build the event dict
    event = {
        # Cell features
        "cell.eF": cell_eF,
        "cell.eB": cell_eB,
        "cell.log_eF": cell_log_eF,
        "cell.log_eB": cell_log_eB,
        "cell.x": cell_x,
        "cell.y": cell_y,
        "cell.z": cell_z,
        "cell.tF": cell_tF,
        "cell.tB": cell_tB,
        "cell.dx": cell_dx,
        "cell.dy": cell_dy,
        "cell.region": cell_region,
        # Cluster targets
        "cluster.e": cluster_e,
        "cluster.eF": cluster_eF,
        "cluster.eB": cluster_eB,
        "cluster.x": cluster_x,
        "cluster.y": cluster_y,
        "cluster.z": cluster_z,
        "cluster.t": cluster_t,
        "cluster.tF": cluster_tF,
        "cluster.tB": cluster_tB,
        "cluster_valid": cluster_valid,
        # Assignment matrix
        "cluster_cell_valid": cluster_cell_valid,
    }

    return event


def main():
    parser = argparse.ArgumentParser(description="Preprocess ECAL ROOT files to NPZ")
    parser.add_argument("--input-dir", required=True, help="Directory containing OutTrigd_*.root files")
    parser.add_argument("--output-dir", required=True, help="Output directory for NPZ files")
    parser.add_argument("--lumi-condition", default="Run5_rotated_2023_baseline", help="Luminosity condition")
    parser.add_argument("--geometry-file", default=None, help="Path to ModuleInfo_*.root geometry file")
    parser.add_argument("--reco-path", default=None, help="Path to reconstruction/modules/ directory")
    parser.add_argument("--seeding", type=int, default=0, help="Seeding mode for clustering")
    parser.add_argument("--min-cell-energy", type=float, default=1e-6, help="Minimum cell energy threshold")
    parser.add_argument("--min-seed-energy", type=float, default=50.0, help="Minimum seed energy for clustering")
    parser.add_argument("--max-events", type=int, default=None, help="Maximum number of events to process")
    parser.add_argument("--start-event", type=int, default=0, help="Starting event index")
    args = parser.parse_args()

    # Add reconstruction modules to path
    if args.reco_path:
        sys.path.insert(0, args.reco_path)
    else:
        # Try common relative paths
        for candidate in [
            os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..", "reconstruction", "reconstruction"),
            os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..", "reconstruction"),
        ]:
            candidate = os.path.abspath(candidate)
            if os.path.isdir(os.path.join(candidate, "modules")):
                sys.path.insert(0, candidate)
                break

    import ROOT
    from modules.Calorimeter import TCalorimeter
    from modules.CellReco import TCellReco
    from modules.Geometry import TGeometry

    # Setup geometry
    if args.geometry_file:
        geo_file = args.geometry_file
    else:
        geo_file = f"./modules/ModuleInfo_{args.lumi_condition}.root"

    geometry = TGeometry(moduleinfo=geo_file, LumiCondition=args.lumi_condition)

    # Set global thresholds
    TCellReco.global_minimum_energy = args.min_cell_energy
    TCellReco.global_minimum_seed_energy = args.min_seed_energy
    TCellReco.global_minimum_seed_ly = 1

    os.makedirs(args.output_dir, exist_ok=True)

    # Find ROOT files
    root_files = sorted(glob.glob(os.path.join(args.input_dir, "OutTrigd_*.root")))
    if not root_files:
        print(f"No OutTrigd_*.root files found in {args.input_dir}")
        return

    print(f"Found {len(root_files)} ROOT files")

    event_count = 0
    skipped_count = 0

    for file_idx, filepath in enumerate(root_files):
        try:
            f = ROOT.TFile.Open(filepath)
            if not f or f.IsZombie():
                print(f"Skipping invalid file: {filepath}")
                continue

            tree = f.Get("tree")
            if not tree:
                print(f"No tree found in: {filepath}")
                f.Close()
                continue

            n_entries = tree.GetEntries()
            print(f"Processing {filepath} ({n_entries} entries)")

            for event_idx in range(n_entries):
                global_idx = event_count + skipped_count
                if global_idx < args.start_event:
                    skipped_count += 1
                    continue

                if args.max_events and event_count >= args.max_events:
                    f.Close()
                    print(f"Reached max events ({args.max_events}), stopping")
                    print(f"Total: {event_count} events saved, {skipped_count} skipped")
                    return

                try:
                    event = prep_event(tree, event_idx, geometry, TCalorimeter, TCellReco, seeding=args.seeding)
                except Exception as e:
                    print(f"  Error processing event {event_idx}: {e}")
                    skipped_count += 1
                    continue

                if event is None:
                    skipped_count += 1
                    continue

                # Save as NPZ
                outpath = os.path.join(args.output_dir, f"reco_ecal_{file_idx}_{event_idx}.npz")
                np.savez_compressed(outpath, **event)
                event_count += 1

                if event_count % 100 == 0:
                    print(f"  Saved {event_count} events ({skipped_count} skipped)")

            f.Close()

        except Exception as e:
            print(f"Error processing file {filepath}: {e}")
            continue

    print(f"Done! Total: {event_count} events saved, {skipped_count} skipped")


if __name__ == "__main__":
    main()
