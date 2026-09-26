# Dependency and release policy

## Public AGPL project update (2026-09-27)

Original project code is now AGPL-3.0-only. The approved
[quality roadmap](SOTA-ROADMAP.md) permits evaluating COLMAP/OpenMVS integration
for an AGPL desktop/server distribution. The earlier blanket exclusion below is
retained as the existing conservative third-party shipping/App Store build policy,
not a statement that the original project is permissively licensed. Experimental
backend binaries remain isolated and unapproved for bundling. No automated
allowlist is relaxed by this documentation update. Promotion needs exact source,
dependency, notice and distribution review; App Store compatibility is not implied.

## Existing shipped third-party dependency gate

The shipped application may use permissive licenses and MPL-2.0. GPL and AGPL components are excluded. LGPL, custom, missing, or unclear terms require an explicit review before inclusion. This is a project policy, not a legal conclusion about an app store.

The user additionally permits separate GPL **development/test oracles**, not shipped dependencies. Keep their pins, notices and data provenance separate from the app's dependency inventory; no linking, source copying, bundling or runtime requirement is authorized by that exception. See [test strategy](TESTING.md). Downloaded benchmark data likewise stays outside distribution and has its own usage terms.

`dependencies/inventory.json` is the exact, reviewed metadata snapshot for npm and Cargo lockfiles plus `dependencies/native.json`. `scripts/check_dependencies.py` (Python 3.11+) compares every locked package identity, integrity checksum, source, version, and reported license against that snapshot and the allowlist. New, changed, missing, or rejected licenses fail. A dual-license entry records the permitted branch selected for this project. To update it after deliberately changing dependencies: install the locked npm tree, generate the Cargo lockfile, run `python3 scripts/check_dependencies.py --update-inventory`, review the diff and upstream license files, then run the checker again. The native list records the optional selected OpenCV source build; the dependency-free C++ foundation still uses only the standard library and platform toolchain. An external native library must be added with a specific version or commit, source URL, SHA-256 integrity, source/notice paths, license, and `status: approved` after review.

The checker examines all npm lock entries, including optional packages for other target platforms. It uses installed package metadata where available, lockfile license metadata where present, and exact-version npm registry metadata for omitted optional packages. Cargo metadata covers all packages selected by the lockfile, including target-specific crates. This is an automated metadata and change-control gate. It does **not** inspect every source file, packaged binary, codec, model, asset, framework, or platform runtime; release review must do that against the actual bundles. Toolchain components and macOS/Windows/Linux webview runtimes are recorded separately at packaging time. npm build tools are included in the inventory because they are part of the reproducible build, even if they are not shipped.

For MPL-covered shipped code, preserve notices and provide the exact covered source and patches for each release through an in-app licenses/source link and a versioned archive. Keep that source accessible for the release. The project's application code does not become MPL merely by linking an MPL library, but copied or modified MPL files retain their obligations. The release gate must verify the actual archive and user-facing link.

The optional, pinned OpenCV selected build and its reviewed native metadata are
documented in [OpenCV](OPENCV.md). Its approval covers only that configuration;
it is not a whole-repository or release-binary legal audit. Other reconstruction
engines listed in [backend evaluation](BACKEND-EVALUATION.md) remain candidates.
No generic `make all` from a candidate repository is accepted as a license-safe build.

Primary references: [Mozilla MPL FAQ](https://www.mozilla.org/en-US/MPL/2.0/FAQ/), [MPL 2.0 text](https://www.mozilla.org/en-US/MPL/2.0/), [Tauri architecture](https://github.com/tauri-apps/tauri/blob/dev/ARCHITECTURE.md), [OpenMVG third-party list](https://openmvg.readthedocs.io/en/latest/third_party/third_party/).
