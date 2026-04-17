use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InternalDidError {
    Estimation(String),
    Convergence(String),
}

impl fmt::Display for InternalDidError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Estimation(message) => write!(f, "Estimation failed: {message}"),
            Self::Convergence(message) => write!(f, "Convergence error: {message}"),
        }
    }
}

impl std::error::Error for InternalDidError {}

#[derive(Debug, thiserror::Error, Clone, PartialEq, Eq)]
pub enum DatasetError {
    #[error("invalid field '{0}': {1}")]
    InvalidField(String, String),

    #[error("unbalanced panel: units missing observations in periods {missing:?}")]
    UnbalancedPanel { missing: Vec<i32> },

    #[error("invalid treatment timing for unit {unit}: {reason}")]
    InvalidTreatmentTiming { unit: String, reason: String },

    #[error("no baseline period found for cohort {cohort}")]
    NoBaselinePeriod { cohort: i32 },

    #[error("missing outcome for unit {unit} in period {period}")]
    MissingOutcome { unit: String, period: i32 },

    #[error("cluster label mismatch: expected {expected} labels, got {actual}")]
    ClusterMismatch { expected: usize, actual: usize },
}
