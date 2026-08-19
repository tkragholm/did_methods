#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TreatmentGroup {
    Control,
    Treated,
}

impl TreatmentGroup {
    #[must_use]
    pub const fn from_bool(treated: bool) -> Self {
        if treated {
            Self::Treated
        } else {
            Self::Control
        }
    }

    #[must_use]
    pub const fn is_treated(self) -> bool {
        matches!(self, Self::Treated)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TimePeriod {
    Pre,
    Post,
}

impl TimePeriod {
    #[must_use]
    pub const fn from_bool(post_period: bool) -> Self {
        if post_period { Self::Post } else { Self::Pre }
    }

    #[must_use]
    pub const fn is_post(self) -> bool {
        matches!(self, Self::Post)
    }
}

/// One 2x2 `DiD` panel observation.
#[derive(bon::Builder, Debug, Clone, Copy, PartialEq)]
pub struct PanelObservation {
    pub treated: bool,
    pub post_period: bool,
    pub outcome: f64,
    #[builder(default = 1.0)]
    pub weight: f64,
}

impl PanelObservation {
    #[must_use]
    pub const fn new(cell: DidCell, outcome: f64) -> Self {
        let (treated, post_period) = cell.flags();
        Self {
            treated,
            post_period,
            outcome,
            weight: 1.0,
        }
    }
}

/// Configuration for 2x2 `DiD` estimation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InferenceConfig {
    pub confidence_level: f64,
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            confidence_level: 0.95,
        }
    }
}

impl InferenceConfig {
    #[must_use]
    pub const fn new(confidence_level: f64) -> Self {
        Self { confidence_level }
    }
}

/// Shared bootstrap configuration for inference procedures.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BootstrapConfig {
    pub reps: usize,
    pub seed: u64,
}

impl Default for BootstrapConfig {
    fn default() -> Self {
        Self {
            reps: 999,
            seed: 17_431,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DidConfig {
    pub confidence_level: f64,
}

impl Default for DidConfig {
    fn default() -> Self {
        Self {
            confidence_level: InferenceConfig::default().confidence_level,
        }
    }
}

impl DidConfig {
    #[must_use]
    pub const fn inference(self) -> InferenceConfig {
        InferenceConfig {
            confidence_level: self.confidence_level,
        }
    }
}

/// 2x2 ATT estimate with a normal-approximation confidence interval.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DidEstimate {
    pub att: f64,
    pub se: f64,
    pub ci_low: f64,
    pub ci_high: f64,
}

/// Canonical 2x2 cell identifiers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DidCell {
    TreatedPre,
    TreatedPost,
    ControlPre,
    ControlPost,
}

impl DidCell {
    #[must_use]
    pub const fn from_parts(group: TreatmentGroup, period: TimePeriod) -> Self {
        match (group, period) {
            (TreatmentGroup::Treated, TimePeriod::Pre) => Self::TreatedPre,
            (TreatmentGroup::Treated, TimePeriod::Post) => Self::TreatedPost,
            (TreatmentGroup::Control, TimePeriod::Pre) => Self::ControlPre,
            (TreatmentGroup::Control, TimePeriod::Post) => Self::ControlPost,
        }
    }

    #[must_use]
    pub const fn as_label(self) -> &'static str {
        match self {
            Self::TreatedPre => "treated_pre",
            Self::TreatedPost => "treated_post",
            Self::ControlPre => "control_pre",
            Self::ControlPost => "control_post",
        }
    }

    #[must_use]
    pub const fn flags(self) -> (bool, bool) {
        match self {
            Self::TreatedPre => (true, false),
            Self::TreatedPost => (true, true),
            Self::ControlPre => (false, false),
            Self::ControlPost => (false, true),
        }
    }

    #[must_use]
    pub const fn treatment_group(self) -> TreatmentGroup {
        match self {
            Self::TreatedPre | Self::TreatedPost => TreatmentGroup::Treated,
            Self::ControlPre | Self::ControlPost => TreatmentGroup::Control,
        }
    }

    #[must_use]
    pub const fn time_period(self) -> TimePeriod {
        match self {
            Self::TreatedPre | Self::ControlPre => TimePeriod::Pre,
            Self::TreatedPost | Self::ControlPost => TimePeriod::Post,
        }
    }
}

/// Summary statistics for one 2x2 cell.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CellSummary {
    pub cell: DidCell,
    pub observations: usize,
    pub weight_sum: f64,
    pub effective_n: f64,
    pub mean_outcome: f64,
    pub variance: f64,
}

/// Input diagnostics across all 2x2 cells.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DidInputSummary {
    pub treated_pre: CellSummary,
    pub treated_post: CellSummary,
    pub control_pre: CellSummary,
    pub control_post: CellSummary,
}

impl DidInputSummary {
    #[must_use]
    pub const fn total_observations(self) -> usize {
        self.treated_pre.observations
            + self.treated_post.observations
            + self.control_pre.observations
            + self.control_post.observations
    }

    #[must_use]
    pub const fn cell(self, cell: DidCell) -> CellSummary {
        match cell {
            DidCell::TreatedPre => self.treated_pre,
            DidCell::TreatedPost => self.treated_post,
            DidCell::ControlPre => self.control_pre,
            DidCell::ControlPost => self.control_post,
        }
    }
}

/// One event-time estimate point for dynamic `DiD` aggregation.
#[derive(bon::Builder, Debug, Clone, Copy, PartialEq)]
pub struct EventTimePoint {
    pub event_time: i32,
    pub estimate: f64,
    pub se: f64,
    #[builder(default = 1.0)]
    pub weight: f64,
}

impl EventTimePoint {
    #[must_use]
    pub const fn new(event_time: i32, estimate: f64, se: f64) -> Self {
        Self {
            event_time,
            estimate,
            se,
            weight: 1.0,
        }
    }
}

/// Weighting strategy used when aggregating multiple points at the same event time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EventTimeWeighting {
    Equal,
    ByWeight,
}

/// Aggregated event-time estimate with confidence interval.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EventTimeEstimate {
    pub event_time: i32,
    pub estimate: f64,
    pub se: f64,
    pub ci_low: f64,
    pub ci_high: f64,
    pub points: usize,
    pub total_weight: f64,
}

/// Errors returned by event-time aggregation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum EventTimeError {
    #[error("confidence_level must be finite and in (0, 1)")]
    InvalidConfidenceLevel,
    #[error("event-time point estimate must be finite")]
    InvalidPointEstimate,
    #[error("event-time point standard error must be finite and non-negative")]
    InvalidPointSe,
    #[error("event-time point weight must be finite and positive")]
    InvalidPointWeight,
}

/// Errors returned by `DiD` estimation.
#[derive(Debug, Clone, Copy, PartialEq, thiserror::Error)]
pub enum DidError {
    #[error("missing observations for cell={cell:?}")]
    EmptyCell { cell: DidCell },
    #[error("weights must be finite and positive, got {value}")]
    InvalidWeight { value: f64 },
    #[error("outcomes must be finite, got {value}")]
    InvalidOutcome { value: f64 },
    #[error("confidence_level must be finite and in (0, 1), got {value}")]
    InvalidConfidenceLevel { value: f64 },
}

/// One panel-level row for doubly robust `DiD` with outcome deltas.
#[derive(bon::Builder, Debug, Clone, PartialEq)]
pub struct DrDidObservation {
    pub treated: bool,
    pub delta_outcome: f64,
    #[builder(default = 1.0)]
    pub weight: f64,
    #[builder(default)]
    pub covariates: Vec<f64>,
}

impl DrDidObservation {
    #[must_use]
    pub const fn new(group: TreatmentGroup, delta_outcome: f64) -> Self {
        Self {
            treated: group.is_treated(),
            delta_outcome,
            weight: 1.0,
            covariates: Vec::new(),
        }
    }
}

/// One repeated cross-section row for doubly robust `DiD`.
#[derive(bon::Builder, Debug, Clone, PartialEq)]
pub struct DrDidRepeatedObservation {
    pub treated: bool,
    pub post_period: bool,
    pub outcome: f64,
    #[builder(default = 1.0)]
    pub weight: f64,
    #[builder(default)]
    pub covariates: Vec<f64>,
}

impl DrDidRepeatedObservation {
    #[must_use]
    pub const fn new(cell: DidCell, outcome: f64) -> Self {
        let (treated, post_period) = cell.flags();
        Self {
            treated,
            post_period,
            outcome,
            weight: 1.0,
            covariates: Vec::new(),
        }
    }
}

/// Configuration for panel `DR-DiD` estimation.
#[derive(bon::Builder, Debug, Clone, Copy, PartialEq)]
pub struct DrDidConfig {
    #[builder(default = 0.95)]
    pub confidence_level: f64,
    #[builder(default = 1e-6)]
    pub propensity_clip: f64,
    #[builder(default = 1e-8)]
    pub ridge: f64,
    #[builder(default = 200_usize)]
    pub max_iter: usize,
    #[builder(default = 1e-8)]
    pub tol: f64,
    #[builder(default = 999_usize)]
    pub bootstrap_reps: usize,
    #[builder(default = 17_431_u64)]
    pub bootstrap_seed: u64,
}

impl Default for DrDidConfig {
    fn default() -> Self {
        Self {
            confidence_level: InferenceConfig::default().confidence_level,
            propensity_clip: 1e-6,
            ridge: 1e-8,
            max_iter: 200,
            tol: 1e-8,
            bootstrap_reps: BootstrapConfig::default().reps,
            bootstrap_seed: BootstrapConfig::default().seed,
        }
    }
}

impl DrDidConfig {
    #[must_use]
    pub const fn inference(self) -> InferenceConfig {
        InferenceConfig {
            confidence_level: self.confidence_level,
        }
    }

    #[must_use]
    pub const fn bootstrap(self) -> BootstrapConfig {
        BootstrapConfig {
            reps: self.bootstrap_reps,
            seed: self.bootstrap_seed,
        }
    }
}

/// `DR-DiD` ATT estimate and uncertainty summary.
#[derive(Debug, Clone, PartialEq)]
pub struct DrDidEstimate {
    pub att: f64,
    pub se: f64,
    pub ci_low: f64,
    pub ci_high: f64,
    pub treated_n: usize,
    pub control_n: usize,
    pub total_weight: f64,
    pub influence_function: Vec<f64>,
}

/// Configuration for `DiD_CC` estimators and the stationarity Hausman test.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DidCcConfig {
    pub drdid: DrDidConfig,
    pub hausman_alpha: f64,
    pub bootstrap_confidence_intervals: bool,
    pub cross_fit_folds: usize,
    pub cross_fit_seed: u64,
}

impl Default for DidCcConfig {
    fn default() -> Self {
        Self {
            drdid: DrDidConfig::default(),
            hausman_alpha: 0.05,
            bootstrap_confidence_intervals: true,
            cross_fit_folds: 1,
            cross_fit_seed: BootstrapConfig::default().seed,
        }
    }
}

/// ATT estimate from the compositional-change `DiD_CC` family.
#[derive(Debug, Clone, PartialEq)]
pub struct DidCcEstimate {
    pub att: f64,
    pub se: f64,
    pub ci_low: f64,
    pub ci_high: f64,
    pub treated_post_n: usize,
    pub treated_pre_n: usize,
    pub control_post_n: usize,
    pub control_pre_n: usize,
    pub total_weight: f64,
    pub influence_function: Vec<f64>,
}

/// Hausman-style comparison between robust and stationarity-based `DiD_CC` ATT
/// estimators.
#[derive(Debug, Clone, PartialEq)]
pub struct DidCcHausmanTest {
    pub robust: DidCcEstimate,
    pub stationary: DidCcEstimate,
    pub difference: f64,
    pub difference_se: f64,
    pub statistic: f64,
    pub p_value: f64,
    pub reject_null: bool,
}

/// Errors returned by `DiD_CC` estimation and testing.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum DidCcError {
    #[error("DiD_CC requires at least one observation")]
    EmptyInput,
    #[error("DiD_CC sample has no treated units")]
    NoTreated,
    #[error("DiD_CC sample has no control units")]
    NoControl,
    #[error("DiD_CC repeated cross-section sample is missing cell={cell:?}")]
    MissingCell { cell: DidCell },
    #[error("weights must be finite and positive, got {value}")]
    InvalidWeight { value: f64 },
    #[error("outcomes must be finite, got {value}")]
    InvalidOutcome { value: f64 },
    #[error("covariates must be finite, got {value}")]
    InvalidCovariate { value: f64 },
    #[error(
        "all DiD_CC rows must have the same covariate count; expected {expected}, got {actual}"
    )]
    InconsistentCovariateCount { expected: usize, actual: usize },
    #[error("confidence_level must be finite and in (0, 1)")]
    InvalidConfidenceLevel,
    #[error("hausman_alpha must be finite and in (0, 1)")]
    InvalidHausmanAlpha,
    #[error("invalid DiD_CC config: {0}")]
    InvalidConfig(String),
    #[error("cross_fit_folds must be at least 1")]
    InvalidCrossFitFolds,
    #[error(
        "cross-fitting with {folds} folds requires at least {folds} observations in cell={cell:?}"
    )]
    InsufficientCellForCrossFit { folds: usize, cell: DidCell },
    #[error("could not solve linear system")]
    SingularSystem,
    #[error("Hausman difference variance must be strictly positive")]
    DegenerateHausmanVariance,
}

/// Errors returned by panel `DR-DiD` estimation.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum DrDidError {
    #[error("DR-DiD requires at least one observation")]
    EmptyInput,
    #[error("DR-DiD sample has no treated units")]
    NoTreated,
    #[error("DR-DiD sample has no control units")]
    NoControl,
    #[error("DR-DiD repeated cross-section sample is missing cell={cell:?}")]
    MissingCell { cell: DidCell },
    #[error("weights must be finite and positive, got {value}")]
    InvalidWeight { value: f64 },
    #[error("delta outcomes must be finite, got {value}")]
    InvalidOutcome { value: f64 },
    #[error("covariates must be finite, got {value}")]
    InvalidCovariate { value: f64 },
    #[error(
        "all DR-DiD rows must have the same covariate count; expected {expected}, got {actual}"
    )]
    InconsistentCovariateCount { expected: usize, actual: usize },
    #[error("invalid DR-DiD config: {0}")]
    InvalidConfig(String),
    #[error("could not solve linear system")]
    SingularSystem,
}

/// One row for staggered-adoption `ATT(g,t)` estimation.
#[derive(bon::Builder, Debug, Clone, Copy, PartialEq)]
pub struct AttGtObservation {
    pub unit_id: Option<i64>,
    pub first_treated_time: Option<i32>,
    pub time: i32,
    pub outcome: f64,
    #[builder(default = 1.0)]
    pub weight: f64,
}

impl AttGtObservation {
    #[must_use]
    pub const fn new(first_treated_time: Option<i32>, time: i32, outcome: f64) -> Self {
        Self {
            unit_id: None,
            first_treated_time,
            time,
            outcome,
            weight: 1.0,
        }
    }

    #[must_use]
    pub const fn with_unit_id(
        unit_id: i64,
        first_treated_time: Option<i32>,
        time: i32,
        outcome: f64,
    ) -> Self {
        Self {
            unit_id: Some(unit_id),
            first_treated_time,
            time,
            outcome,
            weight: 1.0,
        }
    }
}

/// Comparison-group strategy for staggered `ATT(g,t)` estimators.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComparisonGroup {
    NeverTreated,
    NotYetTreated,
}

/// Base-period convention for staggered `ATT(g,t)` estimators.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BasePeriod {
    Varying,
    Universal,
}

/// Configuration for `ATT(g,t)` estimation.
#[derive(bon::Builder, Debug, Clone, Copy, PartialEq)]
pub struct AttGtConfig {
    #[builder(default = InferenceConfig::default())]
    pub confidence_level: InferenceConfig,
    #[builder(default = ComparisonGroup::NeverTreated)]
    pub comparison_group: ComparisonGroup,
    #[builder(default = BasePeriod::Varying)]
    pub base_period: BasePeriod,
    #[builder(default = 0_i32)]
    pub anticipation_periods: i32,
    #[builder(default = true)]
    pub skip_incomplete_pairs: bool,
}

impl Default for AttGtConfig {
    fn default() -> Self {
        Self {
            confidence_level: InferenceConfig::default(),
            comparison_group: ComparisonGroup::NeverTreated,
            base_period: BasePeriod::Varying,
            anticipation_periods: 0,
            skip_incomplete_pairs: true,
        }
    }
}

/// One identified group-time effect estimate.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct AttGtEstimate {
    pub group: i32,
    pub time: i32,
    pub event_time: i32,
    pub att: f64,
    pub se: f64,
    pub ci_low: f64,
    pub ci_high: f64,
    pub treated_n: usize,
    pub control_n: usize,
    pub total_weight: f64,
}

/// ATT(g,t) estimates with aligned influence-function vectors.
#[derive(Debug, Clone, PartialEq)]
pub struct AttGtInfluenceOutput {
    pub estimates: Vec<AttGtEstimate>,
    /// Influence vectors aligned to the full input sample index.
    pub influence_functions: Vec<Vec<f64>>,
}

/// Event-time aggregated ATT estimates with aligned influence-function vectors.
#[derive(Debug, Clone, PartialEq)]
pub struct AttGtEventTimeInfluenceOutput {
    pub estimates: Vec<AttGtEventTimeEstimate>,
    /// Influence vectors aligned to the full input sample index.
    pub influence_functions: Vec<Vec<f64>>,
}

/// Errors returned by `ATT(g,t)` estimation.
#[derive(Debug, Clone, Copy, PartialEq, thiserror::Error)]
pub enum AttGtError {
    #[error("ATT(g,t) requires at least one observation")]
    EmptyInput,
    #[error("confidence_level must be finite and in (0, 1)")]
    InvalidConfidenceLevel,
    #[error("anticipation_periods must be >= 0")]
    InvalidAnticipationPeriods,
    #[error("weights must be finite and positive, got {value}")]
    InvalidWeight { value: f64 },
    #[error("outcomes must be finite, got {value}")]
    InvalidOutcome { value: f64 },
    #[error("covariates must be finite, got {value}")]
    InvalidCovariate { value: f64 },
    #[error(
        "all ATT(g,t) rows must have the same covariate count; expected {expected}, got {actual}"
    )]
    InconsistentCovariateCount { expected: usize, actual: usize },
    #[error("never-treated comparison group is required but missing")]
    MissingNeverTreatedGroup,
    #[error(
        "panel ATT(g,t) requires a unit_id on every row: a unit cannot be \
         differenced across periods unless it can be named"
    )]
    MissingUnitId,
    #[error("no estimable ATT(g,t) pairs found")]
    NoEstimablePairs,
    #[error(
        "missing required cell for group={group}, time={time}, baseline={baseline_time}, cell={cell}"
    )]
    MissingCell {
        group: i32,
        time: i32,
        baseline_time: i32,
        cell: &'static str,
    },
    #[error("ATT(g,t) pair estimation failed for method={method}, group={group}, time={time}")]
    PairEstimationFailure {
        method: &'static str,
        group: i32,
        time: i32,
    },
    #[error(
        "ATT(g,t) pair influence length mismatch for method={method}, group={group}, time={time}: expected {expected}, got {actual}"
    )]
    InfluenceLengthMismatch {
        method: &'static str,
        group: i32,
        time: i32,
        expected: usize,
        actual: usize,
    },
}

/// One row for DR staggered-adoption `ATT(g,t)` estimation with covariates.
#[derive(bon::Builder, Debug, Clone, PartialEq)]
pub struct AttGtDrObservation {
    /// Panel identifier. Required by the panel route
    /// ([`estimate_att_gt_dr_panel`](crate::estimate_att_gt_dr_panel)), which
    /// differences a unit against itself; ignored by the repeated-cross-section
    /// route, where one row is one observation.
    pub unit_id: Option<i64>,
    pub first_treated_time: Option<i32>,
    pub time: i32,
    pub outcome: f64,
    #[builder(default = 1.0)]
    pub weight: f64,
    #[builder(default)]
    pub covariates: Vec<f64>,
}

impl AttGtDrObservation {
    #[must_use]
    pub const fn new(first_treated_time: Option<i32>, time: i32, outcome: f64) -> Self {
        Self {
            unit_id: None,
            first_treated_time,
            time,
            outcome,
            weight: 1.0,
            covariates: Vec::new(),
        }
    }

    /// The same row carrying its panel identifier.
    #[must_use]
    pub const fn with_unit_id(
        unit_id: i64,
        first_treated_time: Option<i32>,
        time: i32,
        outcome: f64,
    ) -> Self {
        Self {
            unit_id: Some(unit_id),
            first_treated_time,
            time,
            outcome,
            weight: 1.0,
            covariates: Vec::new(),
        }
    }
}

/// Unified configuration surface for event-study estimation and inference.
///
/// This groups settings for DR-DiD pair estimation, ATT(g,t) identification,
/// and subsequent aggregation/inference into a single hierarchical object.
#[derive(bon::Builder, Debug, Clone, PartialEq)]
pub struct EventStudyConfig {
    #[builder(default = 0.95)]
    pub confidence_level: f64,
    #[builder(default)]
    pub drdid: DrDidConfig,
    #[builder(default)]
    pub att_gt: AttGtConfig,
    #[builder(default)]
    pub aggregation: AttGtAggregationConfig,
}

impl Default for EventStudyConfig {
    fn default() -> Self {
        Self {
            confidence_level: 0.95,
            drdid: DrDidConfig::default(),
            att_gt: AttGtConfig::default(),
            aggregation: AttGtAggregationConfig::default(),
        }
    }
}

impl EventStudyConfig {
    /// Validate the configuration and ensure no internal conflicts.
    ///
    /// # Errors
    /// Returns an error string if sub-configs contradict top-level settings.
    pub fn validate(&self) -> Result<(), String> {
        // We enforce that if leaf configs differ from the top-level defaults,
        // it must be intentional. For now, we just check for consistency.
        if (self.drdid.confidence_level - self.confidence_level).abs() > 1e-9 {
            return Err(format!(
                "Conflicting confidence_level: top-level {} vs drdid {}",
                self.confidence_level, self.drdid.confidence_level
            ));
        }
        if (self.att_gt.confidence_level.confidence_level - self.confidence_level).abs() > 1e-9 {
            return Err(format!(
                "Conflicting confidence_level: top-level {} vs att_gt {}",
                self.confidence_level, self.att_gt.confidence_level.confidence_level
            ));
        }
        if (self.aggregation.confidence_level.confidence_level - self.confidence_level).abs() > 1e-9
        {
            return Err(format!(
                "Conflicting confidence_level: top-level {} vs aggregation {}",
                self.confidence_level, self.aggregation.confidence_level.confidence_level
            ));
        }
        Ok(())
    }

    /// Synchronize all leaf configurations to match the top-level settings.
    #[must_use]
    pub const fn sync(mut self) -> Self {
        self.drdid.confidence_level = self.confidence_level;
        self.att_gt.confidence_level = self.inference();
        self.aggregation.confidence_level = self.inference();
        self
    }

    #[must_use]
    pub const fn inference(&self) -> InferenceConfig {
        InferenceConfig {
            confidence_level: self.confidence_level,
        }
    }
}

/// Human-readable summary of an estimation result for reporting.
#[derive(serde::Serialize, Debug, Clone, PartialEq)]
pub struct ReportingSummary {
    /// Schema version for downstream pipeline compatibility.
    pub schema_version: u32,
    pub crate_version: String,
    pub config_hash: Option<String>,

    // Raw numeric fields
    pub estimate: f64,
    pub std_error: f64,
    pub ci_lower: f64,
    pub ci_upper: f64,
    pub p_value: Option<f64>,

    // Context
    pub outcome: String,
    pub subgroup: String,
    pub estimator: String,
    pub n_obs: usize,
    pub n_clusters: Option<usize>,
}

impl ReportingSummary {
    /// Display-formatted estimate string (4 decimal places).
    #[must_use]
    pub fn estimate_display(&self) -> String {
        format!("{:.4}", self.estimate)
    }

    /// Display-formatted CI string: "[lower, upper]".
    #[must_use]
    pub fn ci_display(&self) -> String {
        format!("[{:.4}, {:.4}]", self.ci_lower, self.ci_upper)
    }
}

/// Configuration for DR `ATT(g,t)` estimation.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct AttGtDrConfig {
    pub att_gt: AttGtConfig,
    pub drdid: DrDidConfig,
}

/// Weighting rule for ATT(g,t) aggregation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttGtAggregationWeighting {
    Equal,
    ByTreatedCount,
    ByTotalWeight,
}

/// Configuration for ATT(g,t) aggregation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AttGtAggregationConfig {
    pub confidence_level: InferenceConfig,
    pub weighting: AttGtAggregationWeighting,
}

impl Default for AttGtAggregationConfig {
    fn default() -> Self {
        Self {
            confidence_level: InferenceConfig::default(),
            weighting: AttGtAggregationWeighting::ByTreatedCount,
        }
    }
}

/// Aggregated ATT result for a keyed dimension.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AttGtAggregatedEstimate {
    pub estimate: f64,
    pub se: f64,
    pub ci_low: f64,
    pub ci_high: f64,
    pub components: usize,
    pub total_weight: f64,
}

/// Event-time aggregated ATT result.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AttGtEventTimeEstimate {
    pub event_time: i32,
    pub summary: AttGtAggregatedEstimate,
}

/// Cohort aggregated ATT result.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AttGtCohortEstimate {
    pub group: i32,
    pub summary: AttGtAggregatedEstimate,
}

/// Calendar-time aggregated ATT result.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AttGtCalendarEstimate {
    pub time: i32,
    pub summary: AttGtAggregatedEstimate,
}

/// Errors returned by ATT(g,t) aggregation helpers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum AttGtAggregationError {
    #[error("ATT(g,t) aggregation requires at least one estimate")]
    EmptyInput,
    #[error("confidence_level must be finite and in (0, 1)")]
    InvalidConfidenceLevel,
    #[error("ATT(g,t) estimate value must be finite")]
    InvalidEstimate,
    #[error("ATT(g,t) standard error must be finite and non-negative")]
    InvalidSe,
}

/// Configuration for simultaneous confidence bands over ATT(g,t) estimates.
///
/// Used by both:
/// - influence-based multiplier bootstrap bands (preferred), and
/// - Gaussian-max approximation bands (fallback).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AttGtBandConfig {
    pub confidence_level: InferenceConfig,
    pub reps: usize,
    pub seed: u64,
}

impl Default for AttGtBandConfig {
    fn default() -> Self {
        Self {
            confidence_level: InferenceConfig::default(),
            reps: 999,
            seed: 17_431,
        }
    }
}

/// One row for continuous treatment `DiD` (CGBS 2024).
#[derive(bon::Builder, Debug, Clone, Copy, PartialEq)]
pub struct ContinuousObservation {
    /// Treatment dose (D). D=0 means control.
    pub dose: f64,
    /// Change in outcome (ΔY).
    pub delta_outcome: f64,
    /// Sampling weight.
    #[builder(default = 1.0)]
    pub weight: f64,
}

impl ContinuousObservation {
    #[must_use]
    pub const fn new(dose: f64, delta_outcome: f64) -> Self {
        Self {
            dose,
            delta_outcome,
            weight: 1.0,
        }
    }
}

/// Result of an ACRT (Average Causal Response) Sieve estimation.
#[derive(Debug, Clone, PartialEq)]
pub struct ACRTResult {
    /// Average derivative over the treated units.
    pub acrt_glob: f64,
    /// Sieve coefficients.
    pub coefficients: Vec<f64>,
    /// Influence function for `acrt_glob`.
    pub influence_function: Vec<f64>,
}

/// Errors returned by continuous `DiD` estimation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum ContinuousDidError {
    #[error("continuous DiD requires at least one observation")]
    EmptyInput,
    #[error("no treated units found (dose > 0)")]
    NoTreatedUnits,
    #[error("basis matrix is singular")]
    SingularBasis,
}

/// One row for Triple-Difference `DiD` (OVS 2025).
#[derive(bon::Builder, Debug, Clone, PartialEq)]
pub struct TripleDidObservation {
    /// Treatment indicator (D).
    pub treated: bool,
    /// Group indicator (S). e.g., region or cohort enabled.
    pub group_s: bool,
    /// Partition indicator (Q). e.g., eligible age group.
    pub partition_q: bool,
    /// Change in outcome (ΔY).
    pub delta_outcome: f64,
    /// Sampling weight.
    #[builder(default = 1.0)]
    pub weight: f64,
    /// Covariates for Doubly Robust estimation.
    #[builder(default)]
    pub covariates: Vec<f64>,
}

/// Result of a Triple-Difference estimation.
#[derive(Debug, Clone, PartialEq)]
pub struct TripleDidResult {
    pub att_ddd: f64,
    pub se: f64,
    pub ci_low: f64,
    pub ci_high: f64,
    pub influence_function: Vec<f64>,
}

/// Simultaneous-band result for one ATT(g,t) entry.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AttGtBandEstimate {
    pub group: i32,
    pub time: i32,
    pub event_time: i32,
    pub att: f64,
    pub se: f64,
    pub band_low: f64,
    pub band_high: f64,
}

/// Errors returned by ATT(g,t) simultaneous-band helpers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum AttGtBandError {
    #[error("ATT(g,t) simultaneous bands require at least one estimate")]
    EmptyInput,
    #[error("confidence_level must be finite and in (0, 1)")]
    InvalidConfidenceLevel,
    #[error("reps must be > 0")]
    InvalidReps,
    #[error("ATT(g,t) estimate value must be finite")]
    InvalidEstimate,
    #[error("ATT(g,t) standard error must be finite and non-negative")]
    InvalidSe,
    #[error("number of influence vectors must match estimate count")]
    InfluenceCountMismatch,
    #[error("influence vectors must be non-empty")]
    EmptyInfluence,
    #[error("all influence vectors must share the same sample length")]
    InconsistentInfluenceLength,
    #[error("influence vectors must contain only finite values")]
    InvalidInfluence,
    #[error("influence vectors must have non-zero variance")]
    DegenerateInfluence,
}

/// Configuration for the baseline no-covariate `Efficient_DiD` panel path.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct EfficientDidConfig {
    /// Confidence interval level.
    pub confidence_level: InferenceConfig,
}

/// One baseline contribution to an efficient `ATT(g,t)` estimate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EfficientBaselineWeight {
    /// Pre-treatment baseline period used by the underlying `DiD`.
    pub baseline_time: i32,
    /// Baseline-specific `ATT(g,t)` estimate.
    pub att: f64,
    /// Efficient aggregation weight attached to that baseline.
    pub weight: f64,
    /// Treated units with complete observations for this baseline pair.
    pub treated_n: usize,
    /// Never-treated units with complete observations for this baseline pair.
    pub control_n: usize,
}

/// Diagnostics for the baseline no-covariate `Efficient_DiD` weighting step.
#[derive(Debug, Clone, PartialEq)]
pub struct EfficientDidDiagnostics {
    /// Baseline-by-baseline covariance matrix of the influence-function inputs.
    pub baseline_covariance: Vec<Vec<f64>>,
    /// Raw precision-system solution before normalization.
    pub raw_precision_solution: Vec<f64>,
    /// Ridge penalty used to stabilize the covariance inversion, if any.
    pub ridge_penalty: Option<f64>,
}

/// Efficient baseline-weighted `ATT(g,t)` estimate for panel data.
#[derive(Debug, Clone, PartialEq)]
pub struct EfficientDidEstimate {
    /// Treatment cohort.
    pub group: i32,
    /// Post-treatment calendar time.
    pub time: i32,
    /// Event time `time - group`.
    pub event_time: i32,
    /// Efficiently weighted `ATT(g,t)`.
    pub att: f64,
    /// Standard error from the combined efficient influence function.
    pub se: f64,
    /// Lower confidence bound.
    pub ci_low: f64,
    /// Upper confidence bound.
    pub ci_high: f64,
    /// Number of treated units used in the panel comparison.
    pub treated_n: usize,
    /// Number of never-treated units used in the panel comparison.
    pub control_n: usize,
    /// Baseline-specific contributions and efficient weights.
    pub baseline_weights: Vec<EfficientBaselineWeight>,
    /// Combined influence function for the efficient `ATT(g,t)` estimate.
    pub influence_function: Vec<f64>,
    /// Diagnostics for the efficient weighting step.
    pub diagnostics: EfficientDidDiagnostics,
}

/// Event-time aggregated efficient `DiD` estimate.
#[derive(Debug, Clone, PartialEq)]
pub struct EfficientDidEventTimeEstimate {
    /// Event time `e = t - g`.
    pub event_time: i32,
    /// Aggregated summary for this event time.
    pub summary: AttGtAggregatedEstimate,
}

/// Event-time aggregated efficient `DiD` estimates with aligned influence
/// vectors.
#[derive(Debug, Clone, PartialEq)]
pub struct EfficientDidEventTimeInfluenceOutput {
    pub estimates: Vec<EfficientDidEventTimeEstimate>,
    pub influence_functions: Vec<Vec<f64>>,
}

/// Errors returned by the baseline no-covariate `Efficient_DiD` path.
#[derive(Debug, Clone, Copy, PartialEq, thiserror::Error)]
pub enum EfficientDidError {
    #[error("efficient DiD requires at least one observation")]
    EmptyInput,
    #[error("confidence_level must be finite and in (0, 1)")]
    InvalidConfidenceLevel,
    #[error("efficient DiD requires panel unit_id values on every row")]
    MissingUnitId,
    #[error("weights must be finite and positive, got {value}")]
    InvalidWeight { value: f64 },
    #[error("outcomes must be finite, got {value}")]
    InvalidOutcome { value: f64 },
    #[error("never-treated comparison group is required but missing")]
    MissingNeverTreatedGroup,
    #[error("treated cohort {group} has no pre-treatment baseline periods")]
    NoPrePeriods { group: i32 },
    #[error("treated cohort {group} has no post-treatment periods")]
    NoPostPeriods { group: i32 },
    #[error("group {group} time {time} has no complete treated units across all baselines")]
    MissingTreatedPanel { group: i32, time: i32 },
    #[error("group {group} time {time} has no complete never-treated units across all baselines")]
    MissingControlPanel { group: i32, time: i32 },
    #[error("could not solve efficient baseline weighting system")]
    SingularWeightingSystem,
}
