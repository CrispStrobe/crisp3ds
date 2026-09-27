# OpenMVG v2.1 GeometricFilter: separate fifth-target oracle gate

The successful four-target LiGT-off build is sealed by manifest SHA-256
`b9dc7aeb98403acde7377e2b635312b138604cee0d3d74def652803399ed83c6`,
build log SHA-256
`fe58812988f1ac98b9df1359f37d69b792e310eac6a00d8ec78be576a527466d`,
and license receipt SHA-256
`9aec9a7ac01c76f683416cd8eb47c57bbcbfa006a642348568a04298570078c1`.
The fifth target is needed only for a separately approved standard OpenMVG
photo pipeline; **this plan does not run photos or SfM**. The fixed external
root remains `openmvg-v21-ligt-off-oracle-001`, with no new checkout,
configuration, install, or network fetch.

## Exact read-only target delta

Pinned source `main_GeometricFilter.cpp` SHA-256 is
`087875a3122f64c328c365e7e73db752830ddb709ed7ecbda1a075c241ff40fa`.
It declares MPL-2.0 and directly includes the same unresolved LGPLv3+
`third_party/cmdLine/cmdLine.h` as the first four CLI mains. Its other direct
includes are OpenMVG and already-vendored stlplus headers; this is a source
inspection, not a complete transitive-header license clearance.

The generated `ninja -t commands openMVG_main_GeometricFilter` closure has
367 commands, SHA-256
`a680fe2c358c97c6b6a4e56592034c9d6a587871bb05a26b06b909d284928e6d`.
Compared with the sealed four-target command set, **exactly two commands are
new**: one compile of that main (line SHA-256
`08be7d878fe108784cc87f07fff21b56eb203ab8e0090295380e0a5dbfa87d19`)
and one link of `Darwin-arm64-Release/openMVG_main_GeometricFilter` (line
SHA-256 `43cea7df03550a35ad5803edc531a8026cb7304d8213f1aee6ca707061d82d1f`).
No new dependency C/C++ source, static/dynamic library, or framework is in
the link closure. It reuses existing OpenMVG, Ceres, Clp/Osi/CoinUtils,
JPEG/PNG/TIFF and Accelerate inputs. No LiGT file/object or
`USE_PATENTED_LIGT` appears. A dry-run against the sealed frozen Ninja graph
shows exactly two steps and no pending CMake regeneration.

## One-shot execution contract (not yet approved)

The [fifth-target supervisor](../scripts/classical_backend/openmvg_geometric_filter_build.py)
defaults to read-only preflight. It checks all prior manifest/log/receipt,
source/Eigen/patch, four binaries/archives, frozen/generated graph hashes,
main and command-closure seals. A new attempt is blocked if its dedicated
manifest, log, receipt, or binary already exists; a failure is preserved with
no cleanup/retry. The only proposed build command is
`ninja -C <fixed-build> -f frozen-build.ninja -j 2 openMVG_main_GeometricFilter`.
The build log is capped at 16 MiB, process-tree RSS at 4 GiB, the entire fork
tree at 1 GiB, and wall time at 900 seconds. Before launch, external free
must cover the remaining full tree allocation plus 11 GiB; external and
internal free must remain above 11 GiB. Scratch/cache paths stay external.

After a successful build, the supervisor rechecks the four-target/source/
graph seals, hashes the fifth executable, records `otool -L`, and rejects any
new dynamic dependency. It writes distinct
`geometric-filter-build-manifest.json` and
`geometric-filter-license-receipt.json`. The receipt remains explicitly
**evaluation-oracle-only**: cmdLine LGPL-versus-MPL, EPL Clp/Osi/CoinUtils,
Bison exception, Ceres/Eigen, VLFeat/nonFree, and Homebrew dynamic-library
questions are unresolved. No App Store/commercial shipping claim follows
from this fifth target.

Current status: **read-only audit and synthetic tests only; fifth target not
built**. Separate approval and a fresh preflight are required after any
concurrent external/CPU producer is quiet.
