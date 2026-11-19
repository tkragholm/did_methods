//! Improved / locally efficient DR-DiD estimators.
//!
//! This module exposes the Sant'Anna-Zhao improved estimators:
//! - panel improved DR-DiD, implemented in [`panel`]
//! - repeated-cross-section improved DR-DiD, implemented in [`repeated`]
//!
//! Both fit the propensity score by calibration, see [`pscore`].
//!
//! References:
//! - Sant'Anna, P. H. C. and Zhao, J. (2020). "Doubly Robust Difference-in-
//!   Differences Estimators". *Journal of Econometrics*.
//! - `DRDID::drdid_imp_panel`
//! - `DRDID::drdid_imp_rc`

mod panel;
mod pscore;
mod repeated;

pub use panel::estimate_drdid_improved_panel;
pub use repeated::estimate_drdid_improved_repeated_cross_section;
