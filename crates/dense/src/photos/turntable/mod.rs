//! Camera provider `turntable`: our own solver for ordered turntable photos
//! with a known lens (`docs/TURNTABLE-SOLVER.md`).
//!
//! | Module | Content |
//! | --- | --- |
//! | `features`, `matching` | SIFT-class features inside the masks and their matches between neighbouring photos |
//! | `geometry` | Sampson distance and triangulation; an essential-matrix estimator with RANSAC that the solver does not use (kept for captures that are not turntables) |
//! | `solver` | the turntable starting point, tracks, the rounds of adjustment |
//! | `adjust` | bundle adjustment with a Schur complement |
//! | `provider` | the provider as the `photos` command runs it (not in a browser build) |
//!
//! Plain Rust without external programs; everything but `provider` builds for every target.

// Numeric code over small matrices and pixel grids reads best with indices.
#![allow(clippy::needless_range_loop)]

pub mod adjust;
pub mod features;
pub mod geometry;
pub mod matching;
pub mod solver;

pub mod provider;
