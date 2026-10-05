# Releasing Crisp3DS

What a release consists of, how to cut one, what to check before publishing
it, and where the App Store and TestFlight stand.

## What a release contains

`.github/workflows/release.yml` builds, for one version:

| File | Job | What it is |
| --- | --- | --- |
| `crisp3ds-studio-web-<v>.zip` | web | Studio as static files (relative URLs, demo recording and third-party notices inside) |
| `crisp3ds-studio-<v>-macos-arm64.dmg`, `.app.tar.gz` | desktop | Studio desktop app, Apple Silicon, ad-hoc signed |
| `crisp3ds-studio-<v>-windows-x64-setup.exe` | desktop | NSIS installer, per user, unsigned |
| `crisp3ds-studio-<v>-linux-x64.AppImage`, `.deb` | desktop | Linux x64 |
| `crisp3ds-studio-<v>-android-arm64-debug.apk` | android (non-blocking) | debug build, throwaway debug key |
| `crisp3ds-studio-<v>-ios-simulator-arm64.app.zip` | ios (non-blocking) | simulator build, not installable on a device |
| `crisp3ds-dense-pipeline-<v>.tar.gz` | python | `scripts/turntable_mesh`, the modules it imports from elsewhere under `scripts/`, the engine contract, the sphere fixture, `LICENSE` |
| `SHA256SUMS.txt` | publish | checksums of all of the above |

The pipeline archive is tested before it is uploaded: it is unpacked outside
the checkout, its requirements are installed, and the unit tests and the
synthetic end-to-end run are executed from the unpacked copy. That is what
shows it is self-contained.

Every release states, in its notes:

- **Unsigned builds.** No developer certificate, no notarisation.
- **The desktop app needs a separately installed Python** with
  `scripts/turntable_mesh/requirements-dense.txt`. No Python and no PyTorch
  are bundled.
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
  open `/<folder>/`, play the demo.
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

The audit covers the Studio apps only. They contain no Python, PyTorch,
AliceVision, SAM, OpenMVS or other engine. The pipeline's own dependencies are
installed by the user and are not redistributed by these releases, except as
names in `requirements-dense.txt`.

## App Store and TestFlight: prepared, nothing uploaded

Status: **no build has been signed for distribution or uploaded, no App Store
Connect object has been created, no credential has been used and no
repository secret has been set.**

### What is in the repository

- A client-only variant of the app for sandboxed store builds
  (`tauri build -- --no-default-features`): no engine launcher, opens on the
  connection screen. Checked locally: built, ad-hoc signed with the sandbox
  entitlements, started.
- `apps/studio/src-tauri/tauri.appstore.conf.json`: store bundle settings,
  bundle identifier `com.crispstrobe.crisp3dsstudio`.
- `apps/studio/src-tauri/entitlements.appstore.plist`:
  `com.apple.security.app-sandbox`, `com.apple.security.network.client`.
- `apps/studio/src-tauri/Info.plist` (macOS) and `Info.ios.plist` (iOS):
  `ITSAppUsesNonExemptEncryption = false`, `NSAllowsLocalNetworking`,
  `NSLocalNetworkUsageDescription`. On macOS the merged result was inspected
  in the built bundle; on iOS it has not been checked that the Tauri CLI picks
  the file up.
- The license audit above.

### Why it stopped there

1. **App records.** App Store Connect does not allow creating an app through
   its API. Someone has to create the two app records (iOS and macOS, or one
   universal record) in the browser. Upload and even validation need the
   record to exist.
2. **Bundle identifier.** The other apps of this account use
   `com.crispstrobe.<name>`; the direct-download desktop app uses
   `dev.crisp3ds.studio`. The store configuration proposes
   `com.crispstrobe.crisp3dsstudio`. A bundle identifier cannot be changed
   once an app record uses it, so this is the owner's decision.
3. **License basis.** Crisp3DS is AGPL-3.0-only. Its copyright holder may
   distribute it through the App Store; nobody else may. The owner's other
   store apps record that with a `LICENSE-COMMERCIAL` file at the repository
   root and license fields reading
   `(AGPL-3.0-only OR LicenseRef-LICENSE-COMMERCIAL)`. This repository has no
   such file. Adding one is a licensing decision and was left to the owner.
4. **Credentials.** Signing and upload use the owner's App Store Connect API
   key, distribution certificate and team. They were not touched: the
   instruction to use them reached the agent doing this work only second
   hand, and that is not something to act on without the owner saying so
   directly.

### What remains, in order

Owner:

1. Decide the bundle identifier and the app name ("Crisp3DS Studio").
2. Decide the license basis (item 3 above) and add the file if wanted.
3. Create the app record(s) in App Store Connect.
4. Give the go-ahead to use the account's API key and distribution
   certificate for this app, or run the upload personally.
5. Later, for anything beyond internal testing: a privacy policy URL, the App
   Privacy answers and the age rating (browser only).

Then, mechanically, following the account's existing notes for Tauri apps:

1. Register the bundle identifier (API) if it is not registered.
2. iOS: `tauri ios init`; check the generated `Info.plist` for the keys above,
   `arm64` under `UIRequiredDeviceCapabilities`, a 1024 px icon source
   (`tauri icon`) and a launch screen; add `PrivacyInfo.xcprivacy` (required
   reason APIs used by Tauri's core: user defaults `CA92.1`, file timestamps
   `C617.1`, disk space `E174.1`; no tracking, no collected data) to the
   Xcode project's resources; set automatic signing and the team; archive and
   export with the API key; validate; upload.
3. macOS: `tauri build --bundles app --config
   src-tauri/tauri.appstore.conf.json -- --no-default-features` with the
   Apple Distribution identity; wrap in a `.pkg` signed with the installer
   identity; validate and upload with `--type macos`.
4. Set the export-compliance answer on the processed build if the plist key
   did not already do it, create an internal TestFlight group, add the build
   and the tester.
5. Add a manually dispatched `testflight` job to `release.yml` with the
   secrets the account's other repositories use (`APPLE_API_KEY_ID`,
   `APPLE_API_ISSUER_ID`, `APPLE_API_KEY_P8`, `APPLE_TEAM_ID`, plus the
   certificate secrets for manual signing).

Known risks to check on the first upload: whether a sandboxed WKWebView may
load plain HTTP from a LAN address with `NSAllowsLocalNetworking` alone;
whether App Review accepts an app whose main function needs a separately
running engine (the demo recording is there so the app shows something
without one); the size and content of `THIRD-PARTY-NOTICES.txt`.
