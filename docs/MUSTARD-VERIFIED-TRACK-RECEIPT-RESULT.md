# Sealed TRAIN verified-track receipt attempts

No receipt or extracted tracks were produced. The fresh destination
`/Volumes/backups/code/crisp3ds-data/mustard-verified-tracks-001.json`
remains absent. The adapter, graph caps, and gates were not changed in response
to these attempts; no fixed-board-pose sparse or dense integration ran.

The first root-approved attempt used receipt runner SHA-256
`6cd6442e25dc96e8d84af5dcc7e138a0468e84b04bb9d5f2c16c64f0537c46cc`
and exited in 0.77 seconds before writing output. Its supervisor suppressed
the child error, so no exact cause was recorded. After a synthetic-tested
bounded error-reporting patch, the one approved retry used runner SHA-256
`db53413fbd46dc419059df1025f4ed602cc106fd706feedf3f573fef8c2c938a`
and exited in 1.83 seconds with the sanitized diagnostic
`TrackAdapterError: invalid verified geometry row`. No third attempt was run.

The exact offending row/field is unknown. The failure means at least one
`two_view_geometries` row violated the adapter's frozen count/column/blob/
configuration contract; it does **not** establish whether the verified
correspondences are usable or whether any object-only track exists. A
separately authorized scalar-only row-schema audit is needed before any
parser decision. The accepted coarse masks may contain rotating board or
support; mask-supported candidates would not, by themselves, be object-only
geometry.

Preflight and postflight source SHA-256 remained:

- Stage `stage-report.json`: `bbcba624f6a51f82e0ebdebf6e7e4cda10804bc8df34cfd0d5b3242983a225d0`
- Producer `result.json`: `5a83c3e05e2b66887411799ecc8c5f3f41e3b8af3bbd775d308055ef7ee0041e`
- Sealed database: `5858be16dfa3f0fc50303d0eb4a001e21ee14036f67755a9997d5dac374b3075`

Both internal and external volumes remained above the 10 GiB reserve after
the second attempt (approximately 22.2 GiB and 14.2 GiB free, respectively).
