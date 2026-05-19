#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Build the 10eros-runpod-worker Docker image on a build host (Hetzner,
# RunPod GPU Pod, anywhere with Docker + buildx) and push to Docker Hub.
#
# This script is self-contained: it clones the repo into /tmp itself, so the
# typical usage is `curl | bash` directly from GitHub (no local checkout needed).
#
# Usage (curl from GitHub — recommended):
#
#   export DOCKERHUB_USERNAME="your-dockerhub-user"
#   export DOCKERHUB_TOKEN="your-dockerhub-access-token"
#   export IMAGE_TAG="your-user/10eros-runpod-worker:latest"
#   export MODEL_VARIANT=fp8   # or bf16
#   curl -fsSL https://raw.githubusercontent.com/Jmendapara/10eros-runpod-worker/main/scripts/build-on-pod.sh | bash
#
# Usage (local checkout):
#
#   bash scripts/build-on-pod.sh
#
# Optional:
CUDA_LEVEL=12.8                # Force older CUDA 12.6 base (default is 12.8;
#                                  #   12.8 covers Ampere/Ada/Hopper/Blackwell, so
#                                  #   you only need 12.6 for hosts pinned to an
#                                  #   older driver)
PYTORCH_VERSION=2.5.0          # Pin PyTorch; default is "latest" on the index
#   BRANCH=main                    # Git branch to build from (the script always
#                                  #   clones from REPO_URL, never the cwd)
#   REPO_URL=...                   # Override for forks
#   HUGGINGFACE_ACCESS_TOKEN=...   # Pass through if Lightricks/LTX-2.3 is gated
#
# Variant footprints:
#   fp8  → ~37 GB image, fits A100 80 GB / H100 80 GB
#   bf16 → ~54 GB image, recommend H100 80 GB or RTX PRO 6000 96 GB
#
# Prerequisites:
#   - Docker + buildx
#   - ~120 GB free disk for fp8 builds, ~200 GB for bf16.
# =============================================================================

REPO_URL="${REPO_URL:-https://github.com/dxxx12bit/10eros-runpod-worker.git}"
BRANCH="${BRANCH:-main}"
COMFYUI_VERSION="${COMFYUI_VERSION:-latest}"

: "${DOCKERHUB_USERNAME:?Set DOCKERHUB_USERNAME}"
: "${DOCKERHUB_TOKEN:?Set DOCKERHUB_TOKEN}"
: "${IMAGE_TAG:?Set IMAGE_TAG (e.g. yourdockerhubuser/10eros-runpod-worker:latest)}"

MODEL_VARIANT="${MODEL_VARIANT:-fp8}"
if [ "${MODEL_VARIANT}" != "fp8" ] && [ "${MODEL_VARIANT}" != "bf16" ]; then
    echo "ERROR: MODEL_VARIANT must be 'fp8' or 'bf16', got '${MODEL_VARIANT}'" >&2
    exit 1
fi

# Auto-suffix the IMAGE_TAG with the variant if the user didn't already include
# it. This makes it safe to alternate fp8/bf16 builds with a single IMAGE_TAG.
if ! echo "${IMAGE_TAG}" | grep -qE "(:|\-)(fp8|bf16)$"; then
    IMAGE_TAG="${IMAGE_TAG}-${MODEL_VARIANT}"
    echo "       (auto-suffixed IMAGE_TAG → ${IMAGE_TAG})"
fi

echo "============================================="
echo " 10eros-runpod-worker builder"
echo "============================================="
echo "  Repo:         ${REPO_URL}"
echo "  Branch:       ${BRANCH}"
echo "  ComfyUI ver:  ${COMFYUI_VERSION}"
echo "  Image tag:    ${IMAGE_TAG}"
echo "  CUDA level:   ${CUDA_LEVEL:-12.8}"
echo "  Variant:      ${MODEL_VARIANT}"
echo "============================================="

# ---- Step 1: Docker + buildx ----
if ! command -v docker &>/dev/null || ! docker buildx version &>/dev/null 2>&1; then
    echo "[1/5] Installing Docker..."
    curl -fsSL https://get.docker.com | sh
else
    echo "[1/5] Docker already available: $(docker --version)"
fi

if ! docker info &>/dev/null 2>&1; then
    echo "[1/5] Starting Docker daemon..."
    if ! systemctl start docker 2>/dev/null; then
        dockerd &>/dev/null &
        sleep 5
    fi
fi
docker info >/dev/null 2>&1 || { echo "ERROR: Docker daemon is not running."; exit 1; }

# ---- Step 2: Docker Hub login ----
echo "[2/5] Logging into Docker Hub..."
echo "${DOCKERHUB_TOKEN}" | docker login --username "${DOCKERHUB_USERNAME}" --password-stdin

# ---- Step 3: Clone repo ----
WORK_DIR="/tmp/10eros-build-workspace"
if [ -d "${WORK_DIR}" ]; then
    rm -rf "${WORK_DIR}"
fi
echo "[3/5] Cloning ${REPO_URL} (branch ${BRANCH})..."
git clone --depth 1 --branch "${BRANCH}" "${REPO_URL}" "${WORK_DIR}"
cd "${WORK_DIR}"

# ---- Step 4: Build ----
if [ "${MODEL_VARIANT}" = "bf16" ]; then
    echo "[4/5] Building Docker image (bf16 — ~54 GB final image, expect 30+ minutes)..."
else
    echo "[4/5] Building Docker image (fp8 — ~37 GB final image, expect 20+ minutes)..."
fi

BUILD_ARGS=(
    --platform linux/amd64
    --target final
    --build-arg "COMFYUI_VERSION=${COMFYUI_VERSION}"
    --build-arg "MODEL_VARIANT=${MODEL_VARIANT}"
)

if [ -n "${HUGGINGFACE_ACCESS_TOKEN:-}" ]; then
    BUILD_ARGS+=(--build-arg "HUGGINGFACE_ACCESS_TOKEN=${HUGGINGFACE_ACCESS_TOKEN}")
fi

CUDA_LEVEL="${CUDA_LEVEL:-12.8}"
PYTORCH_VERSION="${PYTORCH_VERSION:-}"

if [ "${CUDA_LEVEL}" = "12.6" ]; then
    BUILD_ARGS+=(
        --build-arg "BASE_IMAGE=nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04"
        --build-arg "CUDA_VERSION_FOR_COMFY=12.6"
        --build-arg "ENABLE_PYTORCH_UPGRADE=true"
        --build-arg "PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu126"
        --build-arg "PYTORCH_VERSION=${PYTORCH_VERSION}"
    )
    echo "       Using CUDA 12.6 base + PyTorch cu126 (A100/H100 only; driver >= 560)"
else
    BUILD_ARGS+=(
        --build-arg "BASE_IMAGE=nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04"
        --build-arg "CUDA_VERSION_FOR_COMFY=12.8"
        --build-arg "ENABLE_PYTORCH_UPGRADE=true"
        --build-arg "PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu128"
        --build-arg "PYTORCH_VERSION=${PYTORCH_VERSION}"
    )
    echo "       Using CUDA 12.8 base + PyTorch cu128 (Ampere/Ada/Hopper/Blackwell; driver >= 570)"
fi

# Free disk before build
docker system prune -af --volumes 2>/dev/null || true
docker builder prune -af 2>/dev/null || true
echo "       Disk free: $(df -h /var/lib/docker 2>/dev/null | tail -1 | awk '{print $4}')"

docker buildx use default 2>/dev/null || true
docker buildx build "${BUILD_ARGS[@]}" -t "${IMAGE_TAG}" .

echo "[4/5] Build complete: $(docker images "${IMAGE_TAG}" --format '{{.Size}}')"

# ---- Step 5: Push ----
echo "[5/5] Pushing ${IMAGE_TAG} to Docker Hub..."
docker push "${IMAGE_TAG}"

echo ""
echo "============================================="
echo " SUCCESS"
echo "============================================="
echo "  Image pushed: ${IMAGE_TAG}"
echo "  Variant:      ${MODEL_VARIANT}"
echo ""
echo "  Next steps:"
echo "    1. https://www.runpod.io/console/serverless"
echo "    2. Create endpoint with container image: ${IMAGE_TAG}"
if [ "${CUDA_LEVEL}" = "12.6" ]; then
    echo "    3. Pick GPU: H100 80 GB or A100 80 GB (Blackwell needs CUDA 12.8 build)"
elif [ "${MODEL_VARIANT}" = "bf16" ]; then
    echo "    3. Pick GPU: H100 80 GB or RTX PRO 6000 96 GB (bf16 is tight on A100 80 GB)"
else
    echo "    3. Pick GPU: H100 80 GB, A100 80 GB, or RTX PRO 6000 96 GB"
fi
echo "    4. Container disk: $([ "${MODEL_VARIANT}" = "bf16" ] && echo 120 || echo 80) GB"
echo "    5. Set Min Workers=0, Max Workers=1"
echo "    6. Add env vars: BUCKET_ENDPOINT_URL, BUCKET_ACCESS_KEY_ID,"
echo "         BUCKET_SECRET_ACCESS_KEY, R2_BUCKET_NAME"
echo "    7. Destroy this build server to stop charges!"
echo "============================================="
