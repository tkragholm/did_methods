//! Improved / locally efficient DR-DiD estimators.
//!
//! This module exposes the Sant'Anna-Zhao improved estimators:
//! - panel improved DR-DiD, which in this crate is implemented by the
//!   `panel` path and re-exported here as the improved panel entrypoint
//! - repeated-cross-section improved DR-DiD, implemented in
//!   [`repeated`]
//!
//! References:
//! - Sant'Anna, P. H. C. and Zhao, J. (2020). "Doubly Robust Difference-in-
//!   Differences Estimators". *Journal of Econometrics*.
//! - `DRDID::drdid_imp_panel`
//! - `DRDID::drdid_imp_rc`

mod repeated;

use crate::types::{DrDidConfig, DrDidError, DrDidEstimate, DrDidObservation};

pub use repeated::estimate_drdid_improved_repeated_cross_section;

/// Estimate the improved / locally efficient panel DR-DiD estimator.
///
/// This is the Sant'Anna-Zhao improved panel estimator used by the official
/// `DRDID::drdid_imp_panel` implementation. In this crate the panel
/// implementation already targets that improved score, so this entrypoint
/// delegates to [`super::panel::estimate_drdid_panel`].
///
/// # Errors
///
/// Returns [`DrDidError`] when inputs or nuisance fits are invalid.
pub fn estimate_drdid_improved_panel(
    observations: &[DrDidObservation],
    config: DrDidConfig,
) -> Result<DrDidEstimate, DrDidError> {
    super::panel::estimate_drdid_panel(observations, config)
}
