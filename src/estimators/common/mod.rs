//! Shared helper components used across estimator implementations.
//!
//! The modules here intentionally mix:
//! - production helpers (`basis`, `linalg`, `logit`), and
//! - crate-internal test helpers (`testing`, only compiled under `cfg(test)`).
//!
//! These helpers are lower-level building blocks and generally avoid embedding
//! estimator-specific policy decisions.

pub mod basis;
pub mod linalg;
pub mod logit;
#[cfg(test)]
pub mod testing;
