//! Counts of the conic programs that needed a second attempt or that Clarabel
//! reports as `AlmostSolved`.
//!
//! Each public computation creates a [`SolverCounter`] and passes it down to the
//! functions that solve programs, which cannot be called without one. The
//! counter is a pair of atomic integers behind an `Arc`, so rayon closures and
//! workspaces share it by reference or by clone, and the totals do not depend on
//! the order threads finish in. Two computations use two counters and do not
//! see each other's programs.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use clarabel::solver::SolverStatus;

#[derive(Debug, Default)]
struct Tally {
    retried: AtomicU64,
    almost_solved: AtomicU64,
}

/// How many Clarabel programs a computation solved in a degraded way.
///
/// A program is retried when the first attempt ends in `NumericalError` or
/// `InsufficientProgress`. The retry uses a larger static regularisation, more
/// iterations and tolerances of `1e-6` (smoothness cones) or `1e-7`
/// (relative-magnitude programs), so a retried program is solved to that
/// accuracy rather than Clarabel's default `1e-8`. A program is
/// `AlmostSolved` when Clarabel stops at its reduced tolerances without
/// reaching the full ones, and its solution is used as it is. A retried program
/// that then ends `AlmostSolved` is counted in both fields.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SolverCounts {
    /// Programs that needed the retry.
    pub retried: usize,
    /// Programs whose final status was `AlmostSolved`.
    pub almost_solved: usize,
}

/// The tally of one computation. Clones share the same counts.
#[derive(Debug, Clone, Default)]
pub(crate) struct SolverCounter(Arc<Tally>);

impl SolverCounter {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    pub(crate) fn note_retry(&self) {
        self.0.retried.fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn note_status(&self, status: SolverStatus) {
        if matches!(status, SolverStatus::AlmostSolved) {
            self.0.almost_solved.fetch_add(1, Ordering::Relaxed);
        }
    }

    pub(crate) fn counts(&self) -> SolverCounts {
        SolverCounts {
            retried: usize::try_from(self.0.retried.load(Ordering::Relaxed)).unwrap_or(usize::MAX),
            almost_solved: usize::try_from(self.0.almost_solved.load(Ordering::Relaxed))
                .unwrap_or(usize::MAX),
        }
    }
}
