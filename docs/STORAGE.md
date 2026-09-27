# Storage locations

User-confirmed Mac convention (2026-09-27):

- Source checkout and small active build files: internal disk.
- Large reconstruction artifacts: `/Volumes/backups/code/crisp3ds-data`.
- AI model weights: `/Volumes/backups/ai` (use dedicated, versioned subdirectories).
- VPS large datasets, models, and outputs: `/mnt/storage`; use `/mnt/volume1`
  only after checking its available capacity.

The external Mac volumes `backups` and `BKP_MAC` share one APFS container's
free space; their free-space figures must not be added together.
Keep at least 10 GiB free on the Mac internal disk. Avoid `/tmp` for large
jobs. Verify that the external volume is actually mounted before writing;
an ordinary directory at its mount path is not sufficient.

Completed local reconstruction runs may be relocated with complete file
checksum verification and links at their original paths. Never discard the
internal copy until the destination and the replacement link have been
verified. Such runs become unavailable while the external disk is detached.
Do not move an active run or build directory.

## Verified relocation helper

`python -m scripts.storage.relocate_runs --run NAME` is restricted to four
reviewed completed runs. It refuses existing destinations rather than
overwriting them. Per-run checksums and completion flags live in
`/Volumes/backups/code/crisp3ds-data/build-opencv/.relocation-audits/`.
Only `backup_removed: true` together with `link_verified: true` records a
completed relocation. A partial destination is not a completed move; keep its
internal original until the copy has been verified. The external drive must
remain attached throughout a relocation.
