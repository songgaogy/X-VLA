#!/usr/bin/env bash
# Create repo-root symlinks for data/ and checkpoints/ on this machine.
# Usage:
#   DATA_ROOT=/path/to/data CHECKPOINTS_ROOT=/path/to/checkpoints ./pipeline/setup_local_paths.sh
# Or pass paths as arguments:
#   ./pipeline/setup_local_paths.sh /path/to/data /path/to/checkpoints
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

DATA_ROOT="${1:-${DATA_ROOT:-}}"
CHECKPOINTS_ROOT="${2:-${CHECKPOINTS_ROOT:-}}"

if [[ -z "${DATA_ROOT}" || -z "${CHECKPOINTS_ROOT}" ]]; then
    echo "Usage: DATA_ROOT=... CHECKPOINTS_ROOT=... $0" >&2
    echo "   or: $0 <data_root> <checkpoints_root>" >&2
    exit 1
fi

link_one() {
    local name="$1"
    local src="$2"
    local target="${REPO_ROOT}/${name}"

    if [[ -e "${target}" || -L "${target}" ]]; then
        echo "Skip ${name}: already exists (${target})" >&2
        return 0
    fi
    if [[ ! -d "${src}" ]]; then
        echo "Error: ${src} is not a directory (for ${name})" >&2
        exit 1
    fi
    ln -s "${src}" "${target}"
    echo "Created ${target} -> ${src}"
}

link_one data "${DATA_ROOT}"
link_one checkpoints "${CHECKPOINTS_ROOT}"
