#!/usr/bin/env bash
set -euo pipefail

echo "=== CCache Statistics ==="
ccache -s --verbose || ccache -s || true
echo "========================="
echo ""
echo "Cache directory: $(ccache -k cache_dir 2>/dev/null || echo 'N/A')"
echo "Cache size: $(du -sh "$(ccache -k cache_dir 2>/dev/null || echo .)" 2>/dev/null | cut -f1 || echo 'N/A')"
