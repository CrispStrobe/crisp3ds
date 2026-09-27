# Storage locations

User-confirmed Mac convention (2026-09-27):

- Source checkout and small active build files: internal disk.
- Large reconstruction artifacts: `/Volumes/backups/code/crisp3ds-data`.
- AI model weights: `/Volumes/backups/ai` (use dedicated, versioned subdirectories).
- VPS large datasets, models, and outputs: `/mnt/storage`; use `/mnt/volume1`
  only after checking its available capacity.

Compute policy clarified by the user: run inference on the Mac M1 or Kaggle;
reserve the VPS for smaller CPU tasks and dataset storage. Historical VPS SAM
experiments remain evidence, not the target for further inference. Read
`../kaggle_usage.md` before using Kaggle. On the Mac, put inference environments,
weights and caches under `/Volumes/backups/ai` and experiment outputs under
`/Volumes/backups/code/crisp3ds-data`, with both-volume reserve checks.

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

The four allowlisted runs completed relocation on 2026-09-27: both audit flags
are true for every run, totaling 5,824,950,121 file bytes (5.42 GiB). Their
original paths now resolve through symlinks to the external drive. The verified
internal duplicates were removed; all data remains on the external drive.
