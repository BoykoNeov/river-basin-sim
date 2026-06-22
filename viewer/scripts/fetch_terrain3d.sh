#!/usr/bin/env bash
# Fetch the Terrain3D GDExtension into viewer/addons/ (gitignored: ~50 MB of
# third-party binaries we don't vendor). Pinned for reproducibility.
#
# Usage:  bash viewer/scripts/fetch_terrain3d.sh
# Requires: curl, unzip.
set -euo pipefail

VERSION="v1.0.2-stable"  # verified to load on Godot 4.7 (compatibility_minimum 4.4)
URL="https://github.com/TokisanGames/Terrain3D/releases/download/${VERSION}/Terrain3D_${VERSION}.zip"

here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # viewer/
addons_dir="${here}/addons"

if [ -f "${addons_dir}/terrain_3d/plugin.cfg" ]; then
	echo "Terrain3D already present at ${addons_dir}/terrain_3d -- nothing to do."
	exit 0
fi

tmp="$(mktemp -d)"
trap 'rm -rf "${tmp}"' EXIT
echo "Downloading Terrain3D ${VERSION} ..."
curl -fsSL -o "${tmp}/t3d.zip" "${URL}"
echo "Extracting into ${addons_dir} ..."
mkdir -p "${addons_dir}"
unzip -oq "${tmp}/t3d.zip" -d "${tmp}/extracted"
cp -r "${tmp}/extracted/addons/." "${addons_dir}/"
echo "Done. Terrain3D ${VERSION} installed at ${addons_dir}/terrain_3d"
