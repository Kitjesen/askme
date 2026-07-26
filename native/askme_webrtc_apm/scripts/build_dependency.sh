#!/usr/bin/env sh
set -eu

COMMIT=846fe90a289f58b7c9303a635142aa2c7caa93e5
REPOSITORY=https://gitlab.freedesktop.org/pulseaudio/webrtc-audio-processing.git
PYTHON=${PYTHON:-python3}
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
BUILD_ROOT=${BUILD_ROOT:-"$SCRIPT_DIR/../.build"}
SOURCE="$BUILD_ROOT/source"
BUILD="$BUILD_ROOT/meson-build"
PREFIX="$BUILD_ROOT/prefix"

mkdir -p "$BUILD_ROOT"
if [ ! -d "$SOURCE/.git" ]; then
  mkdir -p "$SOURCE"
  git -C "$SOURCE" init
  git -C "$SOURCE" remote add origin "$REPOSITORY"
fi
# Fetch the pinned tree directly. A blobless clone of the default branch can
# make Git request this older commit one blob at a time from GitLab.
git -C "$SOURCE" config --unset-all remote.origin.promisor 2>/dev/null || {
  code=$?
  [ "$code" -eq 5 ] || exit "$code"
}
git -C "$SOURCE" config --unset-all remote.origin.partialclonefilter 2>/dev/null || {
  code=$?
  [ "$code" -eq 5 ] || exit "$code"
}
git -C "$SOURCE" fetch --depth 1 origin "$COMMIT"
git -C "$SOURCE" checkout --detach "$COMMIT"
ACTUAL_COMMIT=$(git -C "$SOURCE" rev-parse HEAD)
if [ "$ACTUAL_COMMIT" != "$COMMIT" ]; then
  echo "WebRTC APM checkout mismatch: expected $COMMIT, got $ACTUAL_COMMIT" >&2
  exit 1
fi

if [ -f "$BUILD/meson-private/coredata.dat" ]; then
  WIPE=--wipe
else
  WIPE=
fi
"$PYTHON" -m mesonbuild.mesonmain setup $WIPE \
  --prefix "$PREFIX" \
  --buildtype release \
  --wrap-mode forcefallback \
  -Ddefault_library=shared \
  "$BUILD" "$SOURCE"
"$PYTHON" -m mesonbuild.mesonmain compile -C "$BUILD"
"$PYTHON" -m mesonbuild.mesonmain install -C "$BUILD"

mkdir -p "$PREFIX/share/askme-webrtc-apm"
printf '%s' "$COMMIT" > "$PREFIX/share/askme-webrtc-apm/SOURCE_COMMIT.txt"
cp "$SOURCE/COPYING" "$PREFIX/share/askme-webrtc-apm/SOURCE_LICENSE.txt"
printf 'WEBRTC_APM_ROOT=%s\n' "$PREFIX"
