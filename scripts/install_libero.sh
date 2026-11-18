#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "\${BASH_SOURCE[0]}")/.." && pwd)"
LIBERO_DIR="\${ROOT_DIR}/third_party/LIBERO"
PYTHON_BIN="\${PYTHON_BIN:-python}"
PYPI_MIRROR="\${PYPI_MIRROR:-https://pypi.tuna.tsinghua.edu.cn/simple}"

"\${PYTHON_BIN}" - <<'PY'
import sys
if sys.version_info[:2] != (3, 8):
    raise SystemExit(
        "This reproducible setup expects Python 3.8.x; current interpreter is "
        + sys.version.split()[0]
        + ". Create the environment with: conda env create -f environment.yml"
    )
print("Python:", sys.version.split()[0])
PY

if [[ ! -d "\${LIBERO_DIR}/.git" ]]; then
  mkdir -p "$(dirname "\${LIBERO_DIR}")"
  git clone --depth 1 --branch master \
    https://github.com/Lifelong-Robot-Learning/LIBERO.git \
    "\${LIBERO_DIR}"
else
  echo "Using existing LIBERO checkout: \${LIBERO_DIR}"
fi

"\${PYTHON_BIN}" -m pip install --upgrade "pip<25" -i "\${PYPI_MIRROR}"
"\${PYTHON_BIN}" -m pip install -r "\${LIBERO_DIR}/requirements.txt" -i "\${PYPI_MIRROR}"

# robosuite 1.4.0 is an older LIBERO dependency. Pin MuJoCo explicitly rather
# than allowing pip to silently select a newer release.
"\${PYTHON_BIN}" -m pip install "mujoco==2.3.7" -i "\${PYPI_MIRROR}"

# Match the upstream LIBERO installation recipe. The simulator itself uses
# EGL; PyTorch is also required to load the benchmark's fixed init-state files.
"\${PYTHON_BIN}" -m pip install \
  "torch==1.11.0+cu113" \
  "torchvision==0.12.0+cu113" \
  "torchaudio==0.11.0" \
  --index-url "\${PYPI_MIRROR}" \
  --extra-index-url https://download.pytorch.org/whl/cu113

"\${PYTHON_BIN}" -m pip install \
  "imageio==2.35.1" \
  "imageio-ffmpeg==0.5.1" \
  -i "\${PYPI_MIRROR}"

"\${PYTHON_BIN}" -m pip install -e "\${LIBERO_DIR}" --no-deps
"\${PYTHON_BIN}" -m pip install -e "\${ROOT_DIR}" --no-deps

"\${PYTHON_BIN}" "\${ROOT_DIR}/scripts/configure_libero.py"

echo
echo "LIBERO installation completed."
echo "Run: python scripts/check_install.py"
