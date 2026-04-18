#!/usr/bin/env bash
set -euo pipefail

RUNNER_OS="${RUNNER_OS:-$(uname -s)}"

echo "CCACHE_BASEDIR=${GITHUB_WORKSPACE:-$PWD}" >> "${GITHUB_ENV}"
echo "CCACHE_COMPILERCHECK=content" >> "${GITHUB_ENV}"
echo "CCACHE_CPP2=yes" >> "${GITHUB_ENV}"
echo "CCACHE_SLOPPINESS=pch_defines,time_macros,include_file_mtime" >> "${GITHUB_ENV}"

ccache --zero-stats || true
ccache --set-config=max_size=2G || true
ccache --set-config=compiler_check=content || true
ccache --set-config=base_dir="${GITHUB_WORKSPACE:-$PWD}" || true
ccache --set-config=sloppiness=pch_defines,time_macros,include_file_mtime || true

if [[ "${RUNNER_OS}" == "Linux" ]]; then
    echo "/usr/lib/ccache" >> "${GITHUB_PATH}"
    sudo mkdir -p /usr/local/bin
    for tool in gcc g++ cc c++; do
        sudo ln -sf "$(command -v ccache)" "/usr/local/bin/${tool}"
    done
    command -v gcc || true
    gcc --version | head -1 || true
fi

if [[ "${RUNNER_OS}" == "macOS" ]]; then
    sudo mkdir -p /usr/local/bin
    sudo ln -sf "$(command -v ccache)" /usr/local/bin/clang
    sudo ln -sf "$(command -v ccache)" /usr/local/bin/clang++
    echo "/usr/local/bin" >> "${GITHUB_PATH}"
    command -v clang || true
    clang --version | head -1 || true
fi

ccache -p || true
