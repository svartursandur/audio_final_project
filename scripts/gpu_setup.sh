#!/usr/bin/env bash
# Set up the rented GPU server for fine-tuning. Run once, from the project folder on the server:
#     bash scripts/gpu_setup.sh
# The server image already has PyTorch with CUDA. This adds the rest.
set -euo pipefail

apt-get update -qq
apt-get install -y -qq ffmpeg tmux > /dev/null           # ffmpeg decodes the audio
# The image's Python is marked as system-managed; on a throwaway server it is fine to override that.
pip install -q --break-system-packages transformers scikit-learn joblib   # AST, the probe, saving models

python - <<'EOF'
import torch, torchaudio, transformers
print("GPU:", torch.cuda.get_device_name(0))
print("torch", torch.__version__, "| torchaudio", torchaudio.__version__, "| transformers", transformers.__version__)
EOF
