#!/usr/bin/env python3
"""Plot synthetic ECAL data in xy plane."""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.patches as mpatches


def plot_ecal_event(npz_path, output_dir=None, event_idx=0):
    """Plot a single ECAL event showing cells and clusters."""
    data = np.load(npz_path, allow_pickle=True)

    cell_x = data['cell.x']
    cell_y = data['cell.y']
    cell_e = data['cell.eF'] + data['cell.eB']  # Total energy

    cluster_x = data['cluster.x']
    cluster_y = data['cluster.y']
    cluster_e = data['cluster.e']
    cluster_valid = data['cluster_valid']
    cluster_cell_valid = data['cluster_cell_valid']

    num_clusters = len(cluster_x)

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Cells colored by energy
    ax = axes[0]
    scatter = ax.scatter(cell_x, cell_y, c=cell_e, s=50, cmap='viridis',
                         alpha=0.6, edgecolors='black', linewidth=0.5)
    ax.scatter(cluster_x, cluster_y, c='red', s=200, marker='x',
               linewidths=3, label='Cluster centers')

    ax.set_xlabel('X [mm]', fontsize=12)
    ax.set_ylabel('Y [mm]', fontsize=12)
    ax.set_title(f'Event {event_idx}: ECAL Cells (colored by energy)', fontsize=14)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend()

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Cell Energy [MeV]', fontsize=11)

    # Plot 2: Cells colored by cluster assignment
    ax = axes[1]
    colors = plt.cm.tab10(np.linspace(0, 1, num_clusters))

    for c in range(num_clusters):
        if not cluster_valid[c]:
            continue
        mask = cluster_cell_valid[c]
        if mask.sum() > 0:
            ax.scatter(cell_x[mask], cell_y[mask], c=[colors[c]], s=50,
                      alpha=0.7, label=f'Cluster {c} ({cluster_e[c]:.1f} MeV)')

    # Show cluster centers
    for c in range(num_clusters):
        if cluster_valid[c]:
            ax.scatter(cluster_x[c], cluster_y[c], c=[colors[c]], s=300,
                      marker='*', edgecolors='black', linewidths=1.5,
                      zorder=5)

    # Show unassigned cells
    assigned = cluster_cell_valid.any(axis=0)
    unassigned = ~assigned
    if unassigned.sum() > 0:
        ax.scatter(cell_x[unassigned], cell_y[unassigned], c='lightgray',
                  s=30, alpha=0.5, marker='.', label='Unassigned')

    ax.set_xlabel('X [mm]', fontsize=12)
    ax.set_ylabel('Y [mm]', fontsize=12)
    ax.set_title(f'Event {event_idx}: Cell-to-Cluster Assignment', fontsize=14)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=9)

    plt.tight_layout()

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f'ecal_event_{event_idx:04d}.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f'Saved: {output_path}')

    return fig, axes


def main():
    import glob

    data_dir = Path(__file__).parent
    npz_files = sorted(data_dir.glob('reco_ecal_*.npz'))

    if len(npz_files) == 0:
        print(f'No NPZ files found in {data_dir}')
        return

    print(f'Found {len(npz_files)} events')

    # Plot first 5 events
    output_dir = data_dir / 'plots'

    for i, npz_file in enumerate(npz_files[:5]):
        print(f'Plotting {npz_file.name}...')
        plot_ecal_event(npz_file, output_dir=output_dir, event_idx=i)

    plt.show()
    print(f'\nPlots saved to: {output_dir}')


if __name__ == '__main__':
    main()
