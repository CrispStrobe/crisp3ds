# Releasing Crisp3DS

What a release consists of, how to cut one, what to check before publishing
it, and where the App Store and TestFlight stand.

## What a release contains

`.github/workflows/release.yml` builds, for one version:

| File | Job | What it is |
| --- | --- | --- |
| `Crisp-3D-Studio_web-<v>.zip` | web | Studio as static files (relative URLs; demo recording, third-party notices and the engine for browsers, `engine/`, inside) |
| `Crisp-3D-Studio_<v>-macos-arm64.dmg`, `.app.tar.gz` | desktop | Studio desktop app with the engine built in, Apple Silicon, ad-hoc signed |
| `Crisp-3D-Studio_<v>-windows-x64-setup.exe` | desktop | NSIS installer, per user, unsigned |
| `Crisp-3D-Studio_<v>-linux-x64.AppImage`, `.deb` | desktop | Linux x64 |
| `Crisp-3D-Studio_<v>-android-arm64-debug.apk` | android (non-blocking) | debug build, throwaway debug key |
| `Crisp-3D-Studio_<v>-ios-simulator-arm64.app.zip` | ios (non-blocking) | simulator build, not installable on a device |
| `crisp3ds-dense-pipeline-<v>.tar.gz` | python | `scripts/turntable_mesh`, the modules it imports from elsewhere under `scripts/`, the engine contract, the sphere fixture, `LICENSE` |
| `SHA256SUMS.txt` | publish | checksums of all of the above |

The pipeline archive is tested before it is uploaded: it is unpacked outside
the checkout, its requirements are installed, and the unit tests and the
synthetic end-to-end run are executed from the unpacked copy. That is what
shows it is self-contained.

Every release states, in its notes:

- **Unsigned builds.** No developer certificate, no notarisation.
- **The desktop app reconstructs by itself.** The engine (`crates/dense`) is
  built in; from cameras and masks it needs nothing else, and no Python.
- **From plain photos it needs AliceVision or COLMAP on the computer**
  (set under Folders > Tools). Neither is bundled; SAM masks are optional.
- **The web build reconstructs in a browser with WebGPU** ("This browser"),
  from an inputs folder on the device, within 4 GiB of memory.
- **Tried on one Mac only**; Windows, Linux and the phone builds are untested
  beyond building.
- **No GPU-tested CUDA path.** CI runs on CPU; MPS is used on the development
  machine; `--device cuda` has not been run on a GPU by the project.
- **AGPL-3.0-only.**

## Cutting a release

1. Make sure `main` is green (`dense pipeline`, `studio`, `desktop app`).
2. Rehearse: **Actions > release > Run workflow** with `dry_run` on (the
   default), or `gh workflow run release.yml -f dry_run=true`. The version is
   then `0.0.0-dev+<short sha>`. Download `release-all` and look at it (next
   section). Nothing is released by a dry run.
3. Tag and push:

   ```sh
   git tag -a v0.2.0 -m "Crisp3DS 0.2.0"
   git push origin v0.2.0
   ```

   The tag must be `v` followed by a version like `1.2.3` or `1.2.3-rc.1`.
   The workflow writes that version into `apps/studio/package.json`,
   `tauri.conf.json` and `Cargo.toml` for the build
   (`apps/studio/scripts/set-version.mjs`); it does not commit it. To keep the
   repository's own numbers in step, run
   `node apps/studio/scripts/set-version.mjs 0.2.0` and commit before tagging;
   `--check` tells whether the files agree.
4. When the workflow is done there is a **draft prerelease** named
   `Crisp3DS <version>` with all files attached. It is invisible to the public
   until someone publishes it.

Phone builds get a plain numeric version (`1.2.3`; a dry run uses `0.0.1`),
because Apple and Android accept nothing else.

## Before un-drafting

- `SHA256SUMS.txt` lists every attached file, and each file's checksum
  matches after download (`shasum -a 256 -c SHA256SUMS.txt`).
- The desktop app starts on at least one machine per OS, finds or is told the
  checkout and the interpreters, and completes the synthetic sphere run
  (`apps/studio/README.md`, "Running it from a fresh clone"). Only macOS has
  been tried by the project so far.
- On macOS the app opens after the quarantine step described in the notes.
- The web zip works from a sub-path: unpack, serve the folder one level up,
  open `/<folder>/`, play the demo. In Chrome or Edge, "This browser" on the
  Connection screen is offered and completes a small inputs folder
  (`apps/studio/scripts/browser-engine-check.mjs` does this unattended).
- The desktop app's Folders > Tools: "Check" finds AliceVision or COLMAP
  where one is installed, and a photos run completes.
- The pipeline archive: unpack, install the requirements, run the quick start
  of `scripts/turntable_mesh/README.md`.
- The mobile files are present or deliberately absent (their jobs may fail
  without failing the release).
- The release notes still tell the truth (signing, Python, CUDA, license).
- Then: untick "prerelease" only if that is meant, and publish.

To withdraw a release: delete the draft (or the release) and the tag
(`git push origin :refs/tags/v0.2.0`).

## Signing (not set up)

The workflow contains steps for macOS and Windows signing that are skipped
unless these repository secrets exist. They have **never run with real
secrets**; expect to debug them.

| Secret | For |
| --- | --- |
| `APPLE_CERTIFICATE`, `APPLE_CERTIFICATE_PASSWORD` | base64 `.p12` with a Developer ID Application certificate |
| `APPLE_SIGNING_IDENTITY` | the identity's name |
| `APPLE_ID`, `APPLE_PASSWORD`, `APPLE_TEAM_ID` | notarisation of the direct-download `.dmg` |
| `WINDOWS_CERTIFICATE`, `WINDOWS_CERTIFICATE_PASSWORD` | base64 `.pfx` code-signing certificate |

Android release signing (a keystore) and iOS device builds are not wired up.

## Licenses

`apps/studio/docs/THIRD-PARTY-LICENSES.md` and `licenses.json` are generated
by `npm run licenses` in `apps/studio` and checked in the `studio` workflow:
a third-party GPL, AGPL, LGPL, SSPL, non-commercial or unstated license in
what ships fails the build, and so do stale files. Result today: **clean with
obligations**.

- Attribution: release builds carry `THIRD-PARTY-NOTICES.txt` (all license
  texts) and link to it from the app.
- MPL-2.0: `cssparser`, `selectors`, `dtoa-short`, `option-ext`. Used
  unmodified; their source is on crates.io. If one is ever patched, the patch
  must be published.

The audit covers the Studio apps, including the reconstruction engine linked
into them (`crates/dense`, the project's own code, and its dependencies such
as `wgpu`), and the same engine as WebAssembly in the web bundle
(`crates/dense/web`, listed as platform `browser`). They contain no Python,
PyTorch, AliceVision, COLMAP, SAM, OpenMVS or other third-party engine: the
app starts AliceVision or COLMAP as separate programs that the user
installed, and ships none of their code. The Python pipeline's dependencies are installed by the
user and are not redistributed by these releases, except as names in
`requirements-dense.txt`.


## App Store and TestFlight: prepared, nothing uploaded

Status: **no build has been signed for distribution or uploaded, no App Store
Connect object has been created, no credential has been used and no
repository secret has been set.** How this account signs and uploads is
described in the owner's App Store notes, which are kept outside this
repository on purpose; nothing from them belongs in here.

### What is in the repository

- A variant of the app for sandboxed store builds
  (`tauri build -- --no-default-features --features native-engine`): the
  reconstruction engine is built in and runs inside the sandbox; the launcher
  for the external Python engine, which a sandboxed app cannot use, is compiled
  out. Checked locally: built, ad-hoc signed with the sandbox entitlements,
  and the synthetic sphere reconstructed inside it (8 s, folders in the app's
  container). That run was started through a debug hook, not through the
  window: the screen was locked at the time, so the form was not exercised in
  this variant.
- `apps/studio/src-tauri/tauri.appstore.conf.json`: store bundle settings.
  It names no bundle identifier; pass the chosen one at build time
  (`--config '{"identifier":"..."}'`).
- `apps/studio/src-tauri/entitlements.appstore.plist`:
  `com.apple.security.app-sandbox`,
  `com.apple.security.files.user-selected.read-write` (inputs are folders the
  user picks; the runs folder may be one too) and
  `com.apple.security.network.client` (only for the optional engine on another
  computer; drop it if that mode is removed from the store variant).
  Folders picked in a dialog are usable until the app quits: security-scoped
  bookmarks, which would let a chosen data or runs folder survive a restart,
  are not implemented, so the store variant should keep its default folders
  inside the container.
- `apps/studio/src-tauri/Info.plist` (macOS) and `Info.ios.plist` (iOS):
  `ITSAppUsesNonExemptEncryption = false`, `NSAllowsLocalNetworking`,
  `NSLocalNetworkUsageDescription`. Both were checked in built apps: the
  macOS bundle locally, the iOS simulator app in the release workflow, which
  prints these keys.
- The license audit above.

### Why it stopped there

1. **App records.** App Store Connect does not allow creating an app through
   its API. Someone has to create the app record(s) in the browser. Upload,
   and even validation, need the record to exist.
2. **Bundle identifier.** The direct-download desktop app uses
   `dev.crisp3ds.studio`. Whether the store apps use the same or another one
   is the owner's decision; it cannot be changed once an app record uses it.
3. **License basis.** Crisp3DS is AGPL-3.0-only. Its copyright holder may
   distribute it through the App Store; nobody else may. Whether and how to
   record that in this repository (for example with an additional license
   grant for store binaries) is a licensing decision for the owner.
4. **Credentials.** Signing and upload use the owner's App Store Connect API
   key, certificates and team. They were not touched. The request to use them
   reached the agent doing this work only through another agent, and using an
   account's credentials is not something to do without the owner saying so
   directly.

### What remains

Owner:

1. Decide the bundle identifier, the app name and the license basis.
2. Create the app record(s) in App Store Connect.
3. Say directly that the account's key and certificates may be used for this
   app, or run the signing and upload personally.
4. Later, for anything beyond internal testing: a privacy policy URL, the App
   Privacy answers and the age rating.

Then, following the owner's App Store notes:

1. iOS: generate the project (`tauri ios init`), check the generated
   `Info.plist`, icons (`tauri icon` from a 1024 px source) and launch screen,
   add a privacy manifest (`PrivacyInfo.xcprivacy`) for the required-reason
   APIs that Tauri's core uses, sign, archive, validate, upload.
2. macOS: build the store variant
   (`--no-default-features --features native-engine`) with
   `--config src-tauri/tauri.appstore.conf.json`, sign for distribution,
   package, validate, upload.
3. Answer export compliance if the plist key did not, create an internal
   TestFlight group, add the build and the testers.
4. If this is to run in CI, add a manually dispatched `testflight` job to
   `release.yml` with the secrets those notes name.

To check on the first upload: whether a sandboxed web view may load plain
HTTP from a local-network address with `NSAllowsLocalNetworking` alone;
whether review accepts an app whose main function needs a separately running
engine (the demo recording is there so the app shows something without one);
the size of `THIRD-PARTY-NOTICES.txt`.
