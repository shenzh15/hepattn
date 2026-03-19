# Synthetic ECAL Data Generator

This folder contains tools for generating realistic synthetic ECAL (Electromagnetic Calorimeter) data based on the actual Run5 detector geometry.

## Folder Structure

```
data/synthetic_ecal/
├── README.md                      # This file
├── geometry_run5.pkl              # Extracted detector geometry (1 MB)
│                                  # Contains 3312 modules, 15184 cells
├── detector_geometry.png          # Visualization of detector geometry
├── plot_geometry.py               # Script to plot detector geometry
├── generate_synthetic.py          # Script to generate synthetic events
├── plot_synthetic.py              # Script to plot events
├── reco_ecal_000_*.npz           # Generated synthetic events (20 files)
│                                  # Format compatible with hepattn ECAL experiment
└── event_plots/                   # Event visualization plots
    ├── event_0000.png
    ├── event_0001.png
    └── ...
```

## Prerequisites

Activate the conda environment:

```bash
source /home/zhaomr/workdir/PicoCal/miniconda3/bin/activate picocal
```

Required packages:
- numpy
- matplotlib
- pickle (built-in)

## Scripts Usage

### 1. Plot Detector Geometry

Visualize the Run5 ECAL detector geometry (3312 modules, 20 types):

```bash
# Default: saves to detector_geometry.png
python plot_geometry.py

# Custom output file
python plot_geometry.py --output my_geometry.png

# Custom title
python plot_geometry.py --title "Run5 ECAL" -o run5.png

# Use different geometry file
python plot_geometry.py --geometry other_geometry.pkl -o other.png
```

**Output:** PNG file showing all 3312 modules colored by type, with central beam hole.

### 2. Generate Synthetic Events

Generate realistic ECAL events using the actual detector geometry:

```bash
python generate_synthetic.py
```

**What it does:**
- Loads geometry from `geometry_run5.pkl`
- Generates 20 synthetic events
- Each event has 2-4 clusters with realistic cell distributions
- Clusters are localized to actual module positions
- Saves events as `reco_ecal_000_*.npz` (compatible with hepattn)
- Saves plots to `event_plots/event_*.png`

**Event format:**
- `cell.x`, `cell.y`, `cell.z`: Cell positions [mm]
- `cell.eF`, `cell.eB`: Front/back energy deposits [MeV]
- `cell.log_eF`, `cell.log_eB`: Log-scaled energies
- `cell.tF`, `cell.tB`: Front/back timing [ns]
- `cell.dx`, `cell.dy`: Cell sizes [mm]
- `cell.region`: Module region ID
- `cluster.e`, `cluster.x`, `cluster.y`, `cluster.z`: Cluster properties
- `cluster_valid`: Boolean mask for valid clusters
- `cluster_cell_valid`: [clusters, cells] assignment matrix

### 3. Plot Events

Plot individual events (useful for inspection):

```bash
python plot_synthetic.py
```

This will plot the first 5 events and save to `event_plots/`.

## Regenerating Data

To regenerate all synthetic data from scratch:

```bash
# Clean old events
rm reco_ecal_000_*.npz
rm -rf event_plots/

# Regenerate
python generate_synthetic.py
```

## Geometry Source

The `geometry_run5.pkl` file was extracted from the official detector description:

```bash
# Run from reconstruction/reconstruction directory
source /cvmfs/sft.cern.ch/lcg/views/LCG_105_LHCB_7/x86_64-el9-gcc12-opt/setup.sh

python /path/to/extract_geometry.py \
    --lumi Run5_rotated_2023_baseline \
    --moduleinfo ./modules/ModuleInfo_Run5_rotated_2023_baseline.root \
    --output geometry_run5.pkl
```

## Detector Specifications

| Property | Value |
|----------|-------|
| LumiCondition | Run5_rotated_2023_baseline |
| Total modules | 3312 |
| Total cells | 15184 |
| Module types | 20 |
| X range | -3840 to +3840 mm |
| Y range | -3108 to +3108 mm |
| Z position | 12620 mm |
| Central hole | Yes (beam pipe) |

### Module Type Distribution

| Type | Count | Position |
|------|-------|----------|
| 1 | 1344 | Outer ring (red) |
| 2 | 1344 | Middle ring (blue) |
| 3 | 176 | Inner ring (green) |
| 4 | 272 | Inner ring (purple) |
| 5-20 | 176 | Special regions (various colors) |

## Using with hepattn

The generated NPZ files can be used directly with the hepattn ECAL experiment:

```bash
cd /data5/lhcb/zhaomr/PicoCal/hepattn

# Run training with synthetic data
python -m hepattn.experiments.ecal.main fit \
    -c src/hepattn/experiments/ecal/configs/test_synthetic.yaml
```

Update the config to point to this directory:

```yaml
data:
  train_dir: data/synthetic_ecal
  val_dir: data/synthetic_ecal
  test_dir: data/synthetic_ecal
```

## Notes

- Synthetic events use realistic cluster energies (100-800 MeV)
- Cell energies have exponential falloff with distance from cluster center
- Events have 2-4 clusters randomly distributed across detector
- Cell-to-cluster assignment based on proximity (100-150 mm radius)
- All positions match actual detector geometry
