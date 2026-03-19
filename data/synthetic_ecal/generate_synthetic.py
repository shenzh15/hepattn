#!/usr/bin/env python3
"""
Generate realistic synthetic ECAL data using actual detector geometry.

Uses geometry extracted from ModuleInfo_Run5_rotated_2023_baseline.root
"""

import numpy as np
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def load_geometry(pickle_path):
    """Load detector geometry from pickle file."""
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
    return data


def get_module_color(module_type):
    """Get distinct color for each module type."""
    colors = {
        1: '#e41a1c', 2: '#377eb8', 3: '#4daf4a', 4: '#984ea3',
        5: '#ff7f00', 6: '#ffff33', 7: '#a65628', 8: '#f781bf',
        9: '#999999', 10: '#66c2a5', 11: '#fc8d62', 12: '#8da0cb',
        13: '#e78ac3', 14: '#a6d854', 15: '#ffd92f', 16: '#e5c494',
        17: '#b3b3b3', 18: '#e41a1c', 19: '#377eb8', 20: '#4daf4a'
    }
    return colors.get(module_type, 'gray')


def generate_realistic_event(geometry_data, event_idx, num_clusters=3, seed=None):
    """Generate a single realistic ECAL event using actual geometry."""
    if seed is not None:
        np.random.seed(seed + event_idx)

    modules = geometry_data['modules']
    cells = geometry_data['cells']

    # Generate clusters at random positions
    cluster_positions = []
    cluster_energies = []

    for c in range(num_clusters):
        # Pick a random module for this cluster
        mod_idx = np.random.randint(len(modules))
        mod = modules[mod_idx]
        mx, my = mod['x'], mod['y']

        # Cluster center near module center with small offset
        cx = mx + np.random.uniform(-30, 30)
        cy = my + np.random.uniform(-30, 30)

        cluster_positions.append([cx, cy])
        cluster_energies.append(np.random.uniform(100, 800))  # MeV

    if len(cluster_positions) == 0:
        return None

    cluster_positions = np.array(cluster_positions)
    cluster_energies = np.array(cluster_energies)
    actual_num_clusters = len(cluster_positions)

    # Select cells near clusters (within 150mm of any cluster center)
    hit_cells = []
    cell_selection_radius = 150.0

    for cell in cells:
        cell_x, cell_y = cell['x'], cell['y']
        # Check distance to all clusters
        for cx, cy in cluster_positions:
            dist = np.sqrt((cell_x - cx)**2 + (cell_y - cy)**2)
            if dist < cell_selection_radius:
                # Find which module this cell belongs to
                for mod in modules:
                    if (mod['x_bl'] <= cell['x'] <= mod['x_bl'] + mod['dx'] and
                        mod['y_bl'] <= cell['y'] <= mod['y_bl'] + mod['dy']):
                        cell_copy = cell.copy()
                        cell_copy['module_region'] = mod['type']
                        hit_cells.append(cell_copy)
                        break
                break  # Cell assigned, move to next cell

    num_cells = len(hit_cells)
    if num_cells < 5:
        return None

    # Assign cells to clusters based on proximity
    cluster_cell_valid = np.zeros((actual_num_clusters, num_cells), dtype=bool)
    cell_eF = np.zeros(num_cells, dtype=np.float32)
    cell_eB = np.zeros(num_cells, dtype=np.float32)

    for i, cell in enumerate(hit_cells):
        cell_x, cell_y = cell['x'], cell['y']

        # Find distance to each cluster
        distances = np.sqrt((cell_x - cluster_positions[:, 0])**2 +
                           (cell_y - cluster_positions[:, 1])**2)

        # Assign to closest cluster if within reasonable radius
        closest = np.argmin(distances)
        assignment_radius = 150 if hit_cells[i]['module_region'] > 2 else 100

        if distances[closest] < assignment_radius:
            cluster_cell_valid[closest, i] = True

            # Cell energy based on cluster energy and distance
            distance_factor = np.exp(-distances[closest] / 60)
            base_energy = cluster_energies[closest] * distance_factor * np.random.uniform(0.5, 1.5)

            # Split between front and back
            eF = base_energy * np.random.uniform(0.4, 0.6)
            eB = base_energy - eF
            cell_eF[i] = max(eF, 0)
            cell_eB[i] = max(eB, 0)

    # Remove empty clusters
    valid_clusters = cluster_cell_valid.sum(axis=1) > 0
    if valid_clusters.sum() == 0:
        return None

    cluster_cell_valid = cluster_cell_valid[valid_clusters]
    cluster_positions = cluster_positions[valid_clusters]
    cluster_energies = cluster_energies[valid_clusters]
    num_clusters = len(cluster_positions)

    if num_clusters == 0:
        return None

    cluster_eF = cluster_energies * np.random.uniform(0.4, 0.6, num_clusters)
    cluster_eB = cluster_energies - cluster_eF

    # Build arrays
    cell_x = np.array([c['x'] for c in hit_cells], dtype=np.float32)
    cell_y = np.array([c['y'] for c in hit_cells], dtype=np.float32)
    cell_z = np.full(num_cells, 12620.0, dtype=np.float32)
    cell_dx = np.array([c['dx'] for c in hit_cells], dtype=np.float32)
    cell_dy = np.array([c['dy'] for c in hit_cells], dtype=np.float32)
    cell_region = np.array([c['module_region'] for c in hit_cells], dtype=np.int32)
    cell_tF = np.random.exponential(2.0, num_cells).astype(np.float32)
    cell_tB = np.random.exponential(2.0, num_cells).astype(np.float32)

    cluster_x = cluster_positions[:, 0].astype(np.float32)
    cluster_y = cluster_positions[:, 1].astype(np.float32)
    cluster_z = np.full(num_clusters, 12620.0, dtype=np.float32)
    cluster_t = np.random.uniform(0, 10, num_clusters).astype(np.float32)
    cluster_tF = cluster_t + np.random.normal(0, 0.5, num_clusters).astype(np.float32)
    cluster_tB = cluster_t + np.random.normal(0, 0.5, num_clusters).astype(np.float32)
    cluster_valid = np.ones(num_clusters, dtype=bool)

    event = {
        "cell.x": cell_x,
        "cell.y": cell_y,
        "cell.z": cell_z,
        "cell.eF": cell_eF,
        "cell.eB": cell_eB,
        "cell.log_eF": np.log1p(cell_eF),
        "cell.log_eB": np.log1p(cell_eB),
        "cell.tF": cell_tF,
        "cell.tB": cell_tB,
        "cell.dx": cell_dx,
        "cell.dy": cell_dy,
        "cell.region": cell_region,
        "cluster.e": cluster_energies.astype(np.float32),
        "cluster.eF": cluster_eF.astype(np.float32),
        "cluster.eB": cluster_eB.astype(np.float32),
        "cluster.x": cluster_x,
        "cluster.y": cluster_y,
        "cluster.z": cluster_z,
        "cluster.t": cluster_t,
        "cluster.tF": cluster_tF,
        "cluster.tB": cluster_tB,
        "cluster_valid": cluster_valid,
        "cluster_cell_valid": cluster_cell_valid,
    }

    return event


def plot_event_with_geometry(event, geometry_data, output_path=None, event_idx=0):
    """Plot a single event with full detector geometry background."""
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))

    # Draw detector geometry (all modules)
    for mod in geometry_data['modules']:
        rect = Rectangle((mod['x_bl'], mod['y_bl']), mod['dx'], mod['dy'],
                        linewidth=0.2, edgecolor='lightgray',
                        facecolor=get_module_color(mod['type']), alpha=0.3)
        ax.add_patch(rect)

    # Extract event data
    cell_x = event['cell.x']
    cell_y = event['cell.y']
    cell_e = event['cell.eF'] + event['cell.eB']

    cluster_x = event['cluster.x']
    cluster_y = event['cluster.y']
    cluster_e = event['cluster.e']
    cluster_valid = event['cluster_valid']
    cluster_cell_valid = event['cluster_cell_valid']

    num_clusters = len(cluster_x)

    # Plot cells colored by cluster assignment
    colors = plt.cm.tab10(np.linspace(0, 1, max(num_clusters, 3)))

    for c in range(num_clusters):
        if not cluster_valid[c]:
            continue
        mask = cluster_cell_valid[c]
        if mask.sum() > 0:
            # Size based on cell energy
            sizes = 50 + 200 * (cell_e[mask] / (cell_e[mask].max() + 1e-6))
            ax.scatter(cell_x[mask], cell_y[mask], c=[colors[c]], s=sizes,
                      alpha=0.9, edgecolors='black', linewidth=0.5,
                      label=f'Cluster {c} ({cluster_e[c]:.0f} MeV)', zorder=10)

    # Show unassigned cells
    assigned = cluster_cell_valid.any(axis=0)
    unassigned = ~assigned
    if unassigned.sum() > 0:
        ax.scatter(cell_x[unassigned], cell_y[unassigned], c='lightgray',
                  s=20, alpha=0.5, marker='.', label='Unassigned', zorder=5)

    # Plot cluster centers as stars
    for c in range(num_clusters):
        if cluster_valid[c]:
            ax.scatter(cluster_x[c], cluster_y[c], c=[colors[c]], s=500,
                      marker='*', edgecolors='black', linewidths=2, zorder=15)

    ax.set_xlabel('X [mm]', fontsize=12)
    ax.set_ylabel('Y [mm]', fontsize=12)
    ax.set_title(f'Event {event_idx}: ECAL Event on Detector Geometry\n'
                f'{len(cell_x)} cells, {num_clusters} clusters', fontsize=14)
    ax.set_xlim(-3500, 3500)
    ax.set_ylim(-2500, 2500)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2)
    ax.legend(loc='upper right', fontsize=9)

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f'Saved: {output_path}')
        plt.close()

    return fig, ax


def main():
    data_dir = Path(__file__).parent
    geometry_path = data_dir / 'geometry_run5.pkl'

    if not geometry_path.exists():
        print(f"Error: Geometry file not found: {geometry_path}")
        return

    print("Loading detector geometry...")
    geometry_data = load_geometry(geometry_path)
    print(f"  Modules: {len(geometry_data['modules'])}")
    print(f"  Cells: {len(geometry_data['cells'])}")

    # Generate 20 synthetic events
    num_events = 20
    output_dir = data_dir / 'event_plots'
    output_dir.mkdir(exist_ok=True)

    for i in range(num_events):
        num_clusters = np.random.randint(2, 5)
        event = generate_realistic_event(geometry_data, i, num_clusters, seed=42)

        if event is None:
            continue

        # Save as NPZ (overwrite old ones)
        output_file = data_dir / f"reco_ecal_000_{i:04d}.npz"
        np.savez(output_file, **event)

        # Save plot
        plot_path = output_dir / f"event_{i:04d}.png"
        plot_event_with_geometry(event, geometry_data, plot_path, event_idx=i)

        print(f"Created event {i}: {len(event['cell.x'])} cells, {len(event['cluster.e'])} clusters")

    print(f"\nGenerated {num_events} realistic events with proper geometry")


if __name__ == "__main__":
    main()
