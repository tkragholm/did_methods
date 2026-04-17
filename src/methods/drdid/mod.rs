//! Doubly robust difference-in-differences estimators.
//!
//! Current ownership:
//! - [`improved`] contains the explicit locally efficient / improved entrypoints.
//! - [`panel`] contains the panel-data implementation backing the improved panel estimator.
//! - [`repeated`] contains the traditional repeated-cross-section estimator.
//!
//! Historical note:
//! - `estimate_drdid_panel` is retained as a compatibility name, but it currently
//!   implements the improved / locally efficient panel estimator matching
//!   `DRDID::drdid_panel`.

pub mod improved;
pub mod moments;
pub mod panel;
pub mod repeated;
