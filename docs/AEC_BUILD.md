# AskMe WebRTC AEC build gate

AskMe only reports acoustic echo cancellation as active when the optional
`askme-webrtc-apm` native wheel loads and initializes successfully. The wheel
uses Freedesktop's WebRTC Audio Processing **v2.1** source at commit
`846fe90a289f58b7c9303a635142aa2c7caa93e5`. An amplitude gate, VAD threshold,
or pass-through adapter is not AEC.

The build is intentionally two-stage. First build the pinned C++ dependency;
then build the Python wheel against that exact prefix. CMake verifies
`share/askme-webrtc-apm/SOURCE_COMMIT.txt` and refuses an unverified or
different source tree.

The wrapper is MIT-licensed and the bundled upstream library is BSD-licensed;
the build copies Freedesktop's `COPYING` file into the wheel as
`WEBRTC_APM_LICENSE.txt`.

## Prerequisites

- Python 3.11 or newer
- a modern C++ compiler, CMake 3.21+, Git, and Ninja; the Windows dependency
  build uses MSVC in C++20 mode because WebRTC APM v2.1 contains a designated
  initializer even though its Meson project default still says C++17
- Meson 0.63+ (`python -m pip install "meson>=0.63,<2" ninja`)
- platform development tools required by WebRTC APM

The wheel build itself pins `scikit-build-core==1.0.3` and `pybind11==3.0.4`
in `native/askme_webrtc_apm/pyproject.toml`.

## Windows (PowerShell)

```powershell
cd native/askme_webrtc_apm
./scripts/build_dependency.ps1
$env:WEBRTC_APM_ROOT = (Resolve-Path ./.build/prefix)
python -m pip wheel . --no-deps --wheel-dir dist
python -m pip install ./dist/askme_webrtc_apm-*.whl
```

The script activates the Visual Studio environment and forces the dependency
to use MSVC, matching the wheel's compiler and Python architecture. Repair a
release wheel with `delvewheel` and test it in a clean environment before
deployment.

### Match the project interpreter ABI

The extension wheel must match the Python interpreter that starts AskMe. For
this repository's current `uv` environment (CPython 3.11), build and install
the CPython 3.11 wheel from the repository root:

```powershell
$env:WEBRTC_APM_ROOT = (Resolve-Path native/askme_webrtc_apm/.build/prefix).Path
uv build --wheel --python .\.venv\Scripts\python.exe --out-dir native/askme_webrtc_apm/dist native/askme_webrtc_apm
$wheel = Get-ChildItem native/askme_webrtc_apm/dist/askme_webrtc_apm-*-cp311-cp311-win_amd64.whl | Select-Object -First 1
uv pip install --python .\.venv\Scripts\python.exe --reinstall $wheel.FullName
```

A CPython 3.13 (`cp313`) wheel cannot load in CPython 3.11, and the reverse is
also true. If the ABI does not match, AskMe treats native AEC as unavailable
and fails closed to half-duplex instead of advertising unsafe full-duplex.

Run the project's base `uv sync` before installing this platform wheel. A later
plain `uv sync` performs an exact sync and removes packages that are not in the
cross-platform lockfile; after provisioning, use `uv sync --inexact` when the
native wheel must be retained. Ordinary `uv run` also uses inexact syncing by
default. Do not start a prepared full-duplex environment with `uv run --exact`
unless the native wheel has been installed again afterwards. See uv's
[exact/inexact synchronization contract](https://docs.astral.sh/uv/concepts/projects/sync/#handling-of-extraneous-packages).

## Linux/macOS

```sh
cd native/askme_webrtc_apm
python -m pip install "meson>=0.63,<2" ninja
./scripts/build_dependency.sh
export WEBRTC_APM_ROOT="$PWD/.build/prefix"
python -m pip wheel . --no-deps --wheel-dir dist
```

Run `auditwheel repair` on Linux or `delocate-wheel` on macOS before publishing
a release wheel. The CMake install step places dependency libraries beside the
extension, but platform repair is still required to validate loader paths and
manylinux/macOS compatibility.

## Runtime verification

```python
from askme.voice.input.aec_processor import create_aec_processor

aec = create_aec_processor(sample_rate_hz=48_000, channels=1, required=True)
assert aec.stats().active
assert aec.stats().backend == "webrtc-apm-v2.1"
```

`required=True` is the production fail-closed setting for true full duplex. If
the extension is absent, has an ABI/load error, or cannot initialize, it raises
`AecUnavailableError`. With `required=False`, the factory returns a degraded
10 ms PCM pass-through adapter whose status has `active=False`; the application
must then select a platform echo-cancelled path or half-duplex mode.

The processor accepts only interleaved signed 16-bit PCM at 8, 16, 32, or 48
kHz, with one or two channels. Every render and capture call is exactly 10 ms.
Feed the final PCM sent to the speaker into `process_render`, and pass the
measured non-negative render/capture delay to `process_capture` as close to the
audio device boundary as possible.
