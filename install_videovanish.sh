#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="videovanish"
INSTALL_GUI=1
INSTALL_SAM2=1
INSTALL_DiffuEraser=1

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Options:
  --no-gui           Install without GUI packages (default is GUI on)
  --no-sam2          Install without SAM2 packages (default is SAM2 on)
  --no-diffu-eraser  Install without DiffuEraser packages (default is DiffuEraser on)
  -h, --help         Show this help
EOF
}

# Parse args
while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-gui) INSTALL_GUI=0; shift ;;
    --no-sam2) INSTALL_SAM2=0; shift ;;
    --no-diffu-eraser) INSTALL_DiffuEraser=0; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1"; usage; exit 1 ;;
  esac
done

# Ensure conda is available
if ! command -v conda >/dev/null 2>&1; then
  echo "Error: conda not found in PATH. Install Miniconda/Anaconda and try again."
  exit 1
fi

# Load conda into this shell
CONDA_BASE="$(conda info --base)"
# shellcheck source=/dev/null
source "${CONDA_BASE}/etc/profile.d/conda.sh"

# Create env if missing
if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "Conda env '${ENV_NAME}' already exists. Using it."
else
  echo "Creating conda env '${ENV_NAME}' with Python 3.11..."
  if command -v mamba >/dev/null 2>&1; then
    mamba create -y -n "${ENV_NAME}" python=3.11
  else
    conda create -y -n "${ENV_NAME}" python=3.11
  fi
fi

# Activate env
conda activate "${ENV_NAME}"

# Upgrade pip and install base deps
python -m pip install --upgrade pip
python -m pip install numpy opencv-python torch

if [[ $INSTALL_SAM2 -eq 1 ]]; then
	git clone https://github.com/calledit/sam2_numpy_frames
	cd sam2_numpy_frames
	pip install -e .
	cd checkpoints
	if [[ "$OSTYPE" == "linux-gnu"* ]]; then
	    wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
	elif [[ "$OSTYPE" == "darwin"* ]]; then
	    curl -L -O https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
	fi
	
	cd ..
	cd ..
fi

if [[ $INSTALL_DiffuEraser -eq 1 ]]; then
	git clone https://github.com/calledit/DiffuEraser_np_array
	pip install einops diffusers==0.29.2 transformers scipy matplotlib accelerate peft
fi

# Optional GUI
if [[ $INSTALL_GUI -eq 1 ]]; then
  echo "Installing GUI dependencies (PySide6)..."
  python -m pip install PySide6
else
  echo "Skipping GUI dependencies (requested --no-gui)."
fi

echo
echo "✅ Done."
echo "Environment: ${ENV_NAME}"
if [[ $INSTALL_GUI -eq 1 ]]; then
  echo "GUI: enabled (PySide6 installed)"
else
  echo "GUI: disabled"
fi

