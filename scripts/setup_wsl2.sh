#!/usr/bin/env bash
# ===========================================================================
# FastTracker — WSL2 Ubuntu 22.04 環境構築スクリプト
#
# 使い方:
#   1. Windows側から FastTracker リポジトリを WSL2 にコピー
#      cp -r /mnt/c/Users/<user>/projects/FastTracker ~/FastTracker
#
#   2. スクリプトを実行
#      cd ~/FastTracker
#      bash scripts/setup_wsl2.sh
#
# 前提: Ubuntu 22.04 (WSL2), NVIDIA GPU ドライバが Windows側にインストール済み
# ===========================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# 色付き出力
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

info()  { echo -e "${BLUE}[INFO]${NC}  $*"; }
ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; }

# ===========================================================================
# 1. システムパッケージ
# ===========================================================================
info "=== Step 1: システムパッケージのインストール ==="

sudo apt-get update -qq
sudo apt-get install -y -qq \
    build-essential \
    cmake \
    git \
    libeigen3-dev \
    python3 \
    python3-pip \
    python3-venv \
    wget \
    gnupg \
    software-properties-common

ok "システムパッケージ インストール完了"

# ===========================================================================
# 2. CUDA Toolkit
# ===========================================================================
info "=== Step 2: CUDA Toolkit ==="

if command -v nvcc &>/dev/null; then
    NVCC_VER=$(nvcc --version | grep "release" | sed 's/.*release //' | sed 's/,.*//')
    ok "CUDA Toolkit 検出済み: ${NVCC_VER}"
else
    info "CUDA Toolkit をインストールします..."

    # NVIDIA CUDA リポジトリキー追加 (Ubuntu 22.04 / WSL2)
    CUDA_KEYRING="cuda-keyring_1.1-1_all.deb"
    if [ ! -f "/tmp/${CUDA_KEYRING}" ]; then
        wget -q "https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/${CUDA_KEYRING}" \
             -O "/tmp/${CUDA_KEYRING}"
    fi
    sudo dpkg -i "/tmp/${CUDA_KEYRING}"
    sudo apt-get update -qq

    # CUDA Toolkit インストール (最新安定版)
    sudo apt-get install -y -qq cuda-toolkit

    # PATH 設定
    CUDA_PATH_LINE='export PATH=/usr/local/cuda/bin:$PATH'
    CUDA_LD_LINE='export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}'

    if ! grep -qF "cuda/bin" ~/.bashrc 2>/dev/null; then
        echo "" >> ~/.bashrc
        echo "# CUDA Toolkit" >> ~/.bashrc
        echo "${CUDA_PATH_LINE}" >> ~/.bashrc
        echo "${CUDA_LD_LINE}" >> ~/.bashrc
    fi

    export PATH=/usr/local/cuda/bin:$PATH
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}

    if command -v nvcc &>/dev/null; then
        NVCC_VER=$(nvcc --version | grep "release" | sed 's/.*release //' | sed 's/,.*//')
        ok "CUDA Toolkit インストール完了: ${NVCC_VER}"
    else
        error "CUDA Toolkit のインストールに失敗しました"
        error "手動で https://developer.nvidia.com/cuda-downloads からインストールしてください"
        error "  OS: Linux → Architecture: x86_64 → Distribution: WSL-Ubuntu → Version: 2.0"
        exit 1
    fi
fi

# GPU検出確認
if command -v nvidia-smi &>/dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    ok "GPU 検出: ${GPU_NAME}"
else
    warn "nvidia-smi が見つかりません。Windows側のNVIDIAドライバが最新か確認してください"
fi

# ===========================================================================
# 3. Python 依存パッケージ
# ===========================================================================
info "=== Step 3: Python 依存パッケージ ==="

cd "$PROJECT_ROOT"

if [ -f "requirements.txt" ]; then
    pip3 install --quiet -r requirements.txt
    ok "Python パッケージ インストール完了"
else
    warn "requirements.txt が見つかりません。手動でインストールしてください"
    pip3 install --quiet flask numpy matplotlib
fi

# Playwright ブラウザ (ドキュメント生成用、オプション)
if python3 -c "import playwright" 2>/dev/null; then
    python3 -m playwright install chromium 2>/dev/null || true
    ok "Playwright Chromium インストール完了"
fi

# ===========================================================================
# 4. CMake ビルド
# ===========================================================================
info "=== Step 4: FastTracker ビルド ==="

cd "$PROJECT_ROOT"

# 古いビルドディレクトリ削除
if [ -d "build" ]; then
    warn "既存の build/ ディレクトリを削除します"
    rm -rf build
fi

mkdir -p build
cd build

# GPU アーキテクチャの自動検出
CUDA_ARCH=""
if command -v nvidia-smi &>/dev/null; then
    # Compute Capability を取得
    CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.')
    if [ -n "$CC" ]; then
        CUDA_ARCH="-DCMAKE_CUDA_ARCHITECTURES=${CC}"
        info "GPU Compute Capability 検出: ${CC}"
    fi
fi

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    ${CUDA_ARCH}

NPROC=$(nproc)
info "ビルド中... (${NPROC} 並列)"
make -j"${NPROC}"

ok "ビルド完了"

# ===========================================================================
# 5. 動作確認
# ===========================================================================
info "=== Step 5: 動作確認 ==="

cd "$PROJECT_ROOT"

# 実行ファイル確認
if [ -x "build/fasttracker" ]; then
    ok "実行ファイル: build/fasttracker"
    ./build/fasttracker --help 2>&1 | head -3
else
    error "build/fasttracker が見つかりません"
    exit 1
fi

# テスト実行
info "テスト実行中..."
cd build
if ctest --output-on-failure --timeout 60 2>&1; then
    ok "全テスト合格"
else
    warn "一部テストが失敗しました（上記の出力を確認してください）"
fi

cd "$PROJECT_ROOT"

# ===========================================================================
# 完了
# ===========================================================================
echo ""
echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}  FastTracker WSL2 環境構築完了!${NC}"
echo -e "${GREEN}============================================${NC}"
echo ""
echo "使い方:"
echo "  CLI:     ./build/fasttracker --help"
echo "  Web GUI: python3 python/webapp/app.py"
echo "           → http://localhost:5000"
echo ""
echo "ドキュメント再生成:"
echo "  python3 docs/generate_algorithm_doc.py"
echo "  python3 docs/generate_perf_report.py"
echo ""
