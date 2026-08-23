#!/usr/bin/env bash
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
DEST="$ROOT/04_AI模型/知识工作台/HY-MT1.5-1.8B"
REMOTE="https://huggingface.co/tencent/HY-MT1.5-1.8B"

if ! git lfs version >/dev/null 2>&1; then
  echo "ERROR: Git LFS 未安装。请先安装 Git LFS 后重新运行。" >&2
  exit 2
fi

mkdir -p "$(dirname "$DEST")"

if [[ ! -d "$DEST/.git" ]]; then
  rm -rf "$DEST"
  echo "[1/3] 克隆 HY-MT1.5-1.8B 元数据..."
  GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 "$REMOTE" "$DEST"
else
  echo "[1/3] 检测到已有仓库，继续下载..."
fi

cd "$DEST"
git lfs install --local >/dev/null 2>&1 || true

# Keep the model repository independent from the Phoenix source repository.
# Re-running this script resumes/repairs an interrupted LFS download.
echo "[2/3] 同步模型仓库..."
git fetch --depth 1 origin main
git checkout -f main

echo "[3/3] 下载 HY-MT 权重（Git LFS，可断点续传）..."
git lfs pull origin main

python_config="$DEST/config.json"
if [[ ! -s "$python_config" ]]; then
  echo "ERROR: config.json 缺失，模型下载不完整。" >&2
  exit 3
fi

if ! find "$DEST" -maxdepth 1 \( -name '*.safetensors' -o -name '*.bin' \) -type f -size +1k | grep -q .; then
  echo "ERROR: 未发现有效模型权重，Git LFS 下载可能未完成。" >&2
  exit 4
fi

echo "HY-MT1.5-1.8B 下载完成："
echo "$DEST"
echo "Phoenix 会自动把它作为第二级本地医学翻译模型。"
