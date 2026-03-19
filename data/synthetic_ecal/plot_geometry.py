#!/usr/bin/env python3
"""
Plot ECAL detector geometry from pickle file.

Usage:
    python plot_geometry.py [--output geometry.png]
"""

import argparse
import pickle
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pathlib import Path


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


def plot_geometry(geometry_path, output_path=None, title=None):
    """Plot detector geometry from pickle file."""

    # Load geometry
    with open(geometry_path, 'rb') as f:
        data = pickle.load(f)

    modules = data['modules']

    print(f"Loaded geometry: {len(modules)} modules")

    # Count by type
    type_counts = {}
    for mod in modules:
        t = mod['type']
        type_counts[t] = type_counts.get(t, 0) + 1
    print(f"Module types: {sorted(type_counts.keys())}")

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))

    # Plot all modules
    for mod in modules:
        rect = Rectangle(
            (mod['x_bl'], mod['y_bl']), mod['dx'], mod['dy'],
            linewidth=0.3, edgecolor='black',
            facecolor=get_module_color(mod['type']), alpha=0.6
        )
        ax.add_patch(rect)

    # Configure plot
    ax.set_xlabel('X [mm]', fontsize=12)
    ax.set_ylabel('Y [mm]', fontsize=12)

    if title:
        ax.set_title(title, fontsize=14)
    else:
        ax.set_title(f'Run5 ECAL Detector Geometry\n{len(modules)} modules, {len(data["cells"])} cells',
                    fontsize=14)

    ax.set_xlim(-3500, 3500)
    ax.set_ylim(-2500, 2500)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=get_module_color(t), edgecolor='black',
              label=f'Type {t} ({type_counts[t]} mods)', alpha=0.7)
        for t in sorted(type_counts.keys())
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=8,
             title='Module Types', ncol=2)

    plt.tight_layout()

    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
    else:
        plt.show()

    plt.close()

    # Print summary
    print(f"\nGeometry Summary:")
    print(f"  LumiCondition: {data['LumiCondition']}")
    print(f"  Total modules: {len(modules)}")
    print(f"  Total cells: {len(data['cells'])}")
    print(f"  X range: {min(m['x'] for m in modules):.1f} to {max(m['x'] for m in modules):.1f} mm")
    print(f"  Y range: {min(m['y'] for m in modules):.1f} to {max(m['y'] for m in modules):.1f} mm")


def main():
    parser = argparse.ArgumentParser(description='Plot ECAL detector geometry')
    parser.add_argument('--geometry', '-g', default='geometry_run5.pkl',
                       help='Path to geometry pickle file')
    parser.add_argument('--output', '-o', default=None,
                       help='Output PNG file path')
    parser.add_argument('--title', '-t', default=None,
                       help='Plot title')

    args = parser.parse_args()

    # Default output name
    if args.output is None:
        args.output = args.geometry.replace('.pkl', '.png')

    plot_geometry(args.geometry, args.output, args.title)


if __name__ == '__main__':
    main()
