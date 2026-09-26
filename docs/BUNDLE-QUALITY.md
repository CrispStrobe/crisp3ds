# Board-anchored bundle adjustment quality experiment

The existing unmasked MVE L1 baseline has 2.130 mm matched point-to-plane MAE;
the known-pose diagnostic ceiling has 0.660 mm on the same rendered fixture.
This isolated experiment tests whether a production-available input, observed
board corners, can help close that gap. It does not modify the production core,
the MVE source, the fixture, or the original six-run ablation.

One candidate was frozen before looking at dense scores: jointly optimize the
three camera object-to-camera poses and all 1,189 object tracks from the 2,843
existing ORB pixel observations, plus 48 default-detector ArUco board-corner
observations. The board's 3D corners are fixed in millimetres, anchoring gauge
and scale. Intrinsics remain fixed. Marker and track residuals have equal
weight; each 2D residual uses Huber loss with 1 px scale. Ceres runs at most
30 iterations using `DENSE_SCHUR`. No truth pose, plane, object point, or
ground-truth pixel enters the optimizer. The same unmasked images and selected
MVE engine/options from [QUALITY-ABLATION.md](QUALITY-ABLATION.md) provide
the dense replay.

## Local dependencies and reproducibility

This test uses local source archives, with no global install:

| Source | Pin | Archive SHA-256 |
| --- | --- | --- |
| [Ceres Solver](https://github.com/ceres-solver/ceres-solver/tree/2.2.0) | 2.2.0 | `12efacfadbfdc1bbfa203c236e96f4d3c210bed96994288b3ff0c8e7c6f350d4` |
| [Eigen](https://gitlab.com/libeigen/eigen/-/tree/3.4.0) | 3.4.0 | `8586084f71f9bde545ee7fa6d00288b264a2b7ac3607b974e54d13e7162c1c72` |

The Ceres source has a BSD-style license plus bundled notices; its `LICENSE`
SHA-256 is `765fd9eac4cb827fa5de4304d1f81f4f2c5c332a3f1d6f0f3f7a69a33156729b`.
Eigen 3.4.0 is primarily MPL-2.0 but its archive includes optional LGPL files
(`COPYING.README` SHA-256 `c83230b770f17ef1386ea1fd3681271dd98aa93646bdbfb5bff3a1b7050fff9d`).
The selected guarded build adds `EIGEN_MPL2_ONLY`, which makes Eigen fail
compilation if an LGPL header is included. SuiteSparse, LAPACK, Accelerate
sparse, glog, and gflags are disabled. This local research build and record do
not constitute transitive-license review or shipping approval.

Recreate the guarded build from the repository root, using fresh local
directories and the archive URLs matching the pinned version links:

```sh
mkdir -p .local-tools/bundle-quality
mkdir -p .local-tools/tmp
export TMPDIR="$PWD/.local-tools/tmp"
df -h .  # verify at least 12 GiB free before the build (10 GiB reserve plus headroom)
curl -L --fail https://github.com/ceres-solver/ceres-solver/archive/refs/tags/2.2.0.tar.gz -o .local-tools/bundle-quality/ceres-2.2.0.tar.gz
curl -L --fail https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz -o .local-tools/bundle-quality/eigen-3.4.0.tar.gz
shasum -a 256 .local-tools/bundle-quality/*.tar.gz
tar -xzf .local-tools/bundle-quality/ceres-2.2.0.tar.gz -C .local-tools/bundle-quality
tar -xzf .local-tools/bundle-quality/eigen-3.4.0.tar.gz -C .local-tools/bundle-quality
cmake -S .local-tools/bundle-quality/eigen-3.4.0 -B .local-tools/bundle-quality/eigen-build -DCMAKE_INSTALL_PREFIX="$PWD/.local-tools/bundle-quality/eigen-install" -DBUILD_TESTING=OFF -DEIGEN_BUILD_DOC=OFF
cmake --install .local-tools/bundle-quality/eigen-build
cmake -S .local-tools/bundle-quality/ceres-solver-2.2.0 -B .local-tools/bundle-quality/ceres-build-mpl -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS=-DEIGEN_MPL2_ONLY -DCMAKE_INSTALL_PREFIX="$PWD/.local-tools/bundle-quality/ceres-install-mpl" -DCMAKE_PREFIX_PATH="$PWD/.local-tools/bundle-quality/eigen-install" -DMINIGLOG=ON -DGFLAGS=OFF -DSUITESPARSE=OFF -DACCELERATESPARSE=OFF -DLAPACK=OFF -DEIGENSPARSE=ON -DUSE_CUDA=OFF -DBUILD_TESTING=OFF -DBUILD_EXAMPLES=OFF -DBUILD_DOCUMENTATION=OFF -DBUILD_SHARED_LIBS=OFF -DCMAKE_EXPORT_PACKAGE_REGISTRY=OFF
cmake --build .local-tools/bundle-quality/ceres-build-mpl --target ceres -j2
cmake --install .local-tools/bundle-quality/ceres-build-mpl --prefix "$PWD/.local-tools/bundle-quality/ceres-install-mpl"
cmake -S scripts/bundle_quality -B .local-tools/bundle-quality/quality-build-mpl -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS=-DEIGEN_MPL2_ONLY -DCMAKE_PREFIX_PATH="$PWD/.local-tools/bundle-quality/ceres-install-mpl;$PWD/.local-tools/bundle-quality/eigen-install"
cmake --build .local-tools/bundle-quality/quality-build-mpl -j2
.local-tools/bundle-quality/quality-build-mpl/bundle_quality_optimize --self-test
python3 scripts/bundle_quality/run.py --output build-opencv/bundle-quality-mpl --optimizer .local-tools/bundle-quality/quality-build-mpl/bundle_quality_optimize
python3 -m unittest scripts.bundle_quality.test_input -v
```

The runner deliberately returns failure if board RMS increases. It preserves
the optimizer output and initial/final raw ORB and marker RMS, robust cost,
positive-depth checks, convergence, binary/source/input checksums, `/usr/bin/time`
peak resident bytes, and logs. Replaying a rejected saved solution for diagnostic
scoring is explicit:

```sh
python3 scripts/bundle_quality/score_rejected.py --output build-opencv/bundle-quality-mpl
```

## Fixed candidate: rejected

The first local build solved the frozen problem in five iterations.
Robust cost fell 201.281 to 134.104; ORB RMS fell 0.3504 to 0.2733 px; all
observed points stayed in front of their cameras (minimum 601.29 mm). The
board-marker RMS rose from 1.10791 to 1.11717 px. This violated the declared
board safeguard by 0.00925 px, so the candidate is **rejected**, despite
convergence. The failure and values are preserved in
[`optimized.json`](../build-opencv/bundle-quality/optimized.json) and
[`status.json`](../build-opencv/bundle-quality/status.json).

The saved rejected solution was passed through unmasked MVE L1 solely to
measure what this optimization did. Its matched point-to-plane MAE was 1.580 mm
versus 2.130 mm for the estimated baseline; P95 was 3.151 versus 3.919 mm;
coverage was 98.62% versus 98.51%; missing-inclusive bad-5 was 1.77% versus
2.54%. Peak MVE resident memory was 45.9 MB and wall time about 7.06 s for
this replay. The result is in
[`rejected-diagnostic.json`](../build-opencv/bundle-quality/rejected-diagnostic.json).
An independent fixed-truth pixel comparison found finite-rectangle bad-2
51.74% to 27.23% and 2-pixel boundary bad-2 62.67% to 50.00%; see
[`world-fixed-truth-L1.json`](../build-opencv/bundle-quality/world-fixed-truth-L1.json).
These diagnostic gains do not override the marker safeguard. No alternative
weight, loss, or iteration count was tried after seeing these scores.

The guarded `EIGEN_MPL2_ONLY` build repeated the same frozen candidate in a
fresh directory. Initial and final robust costs agree with the first build to
about 1e-12; final ORB and marker RMS agree to machine precision. Its rejected
L1 point-to-plane MAE is 1.579472 mm, differing from the first replay by less
than 1e-12 mm. The original failed status remains intact. The guarded run's
[`rejected-diagnostic.json`](../build-opencv/bundle-quality-mpl/rejected-diagnostic.json)
and [`rejected-manifest.json`](../build-opencv/bundle-quality-mpl/rejected-manifest.json)
record the scored output, verified original input hashes, replay source hash,
scorer hash, and baseline result hash. Both numerical Jacobian checks and a
synthetic anchored camera-and-point fit pass; malformed rotation, observation
index, and nonpositive-depth inputs are rejected by the guarded executable.
