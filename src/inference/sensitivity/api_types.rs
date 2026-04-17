use crate::types::InferenceConfig;

/// Sensitivity restrictions for `HonestDiD` robustness analysis.
///
/// `Smoothness(M)` corresponds to the `ΔSD(M)` class, which bounds changes in
/// slope across adjacent event-time violations.
///
/// `RelativeMagnitude(Mbar)` corresponds to the `ΔRM(Mbar)` class, which bounds
/// post-treatment violations by a multiple of the largest pre-treatment
/// violation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum HonestSensitivity {
    /// Smoothness restriction `ΔSD(M)`.
    Smoothness(f64),
    /// Relative-magnitude restriction `ΔRM(Mbar)`.
    RelativeMagnitude(f64),
}

/// Result of a scalar `HonestDiD` robustness calculation.
#[derive(Debug, Clone, PartialEq)]
pub struct HonestResult {
    /// Original point estimate.
    pub original_estimate: f64,
    /// Original standard error.
    pub original_se: f64,
    /// Robust confidence interval.
    pub robust_ci: (f64, f64),
    /// Identified set implied by the sensitivity restriction.
    pub identified_set: (f64, f64),
    /// Sensitivity parameter used in the calculation.
    pub m: f64,
}

/// `HonestDiD` result for one post-treatment event-study period.
#[derive(Debug, Clone, PartialEq)]
pub struct HonestEventStudyPoint {
    /// User-facing post-period label.
    pub post_period: i32,
    /// `HonestDiD` scalar result for that period.
    pub result: HonestResult,
}

/// Typed input for event-study `HonestDiD` analysis.
///
/// Coefficients are stored in pre-then-post order. The covariance matrix must
/// use the same ordering.
#[derive(Debug, Clone, PartialEq)]
pub struct HonestEventStudyInput {
    /// Event-study coefficient vector in pre-then-post order.
    pub betahat: Vec<f64>,
    /// Covariance matrix matching `betahat`.
    pub covariance: Vec<Vec<f64>>,
    /// Labels for pre-treatment coefficients.
    pub pre_periods: Vec<i32>,
    /// Labels for post-treatment coefficients.
    pub post_periods: Vec<i32>,
}

/// Original parallel-trends confidence set for a post-treatment linear
/// functional.
#[derive(Debug, Clone, PartialEq)]
pub struct HonestOriginalConfidenceSet {
    /// Point estimate of the target linear functional.
    pub estimate: f64,
    /// Standard error of the target functional.
    pub se: f64,
    /// Two-sided original confidence interval.
    pub ci: (f64, f64),
}

/// Exact `ΔRM(Mbar)` identified set.
#[derive(Debug, Clone, PartialEq)]
pub struct HonestIdentifiedSet {
    /// Lower bound.
    pub lb: f64,
    /// Upper bound.
    pub ub: f64,
}

/// Conditional confidence set under a relative-magnitude sensitivity
/// restriction.
#[derive(Debug, Clone, PartialEq)]
pub struct HonestConditionalConfidenceSet {
    /// Lower bound.
    pub lb: f64,
    /// Upper bound.
    pub ub: f64,
}

/// Hybrid mode for `DeltaRM` conditional confidence sets.
///
/// This matches the `hybrid_flag` argument used by the reference
/// `HonestDiD::computeConditionalCS_DeltaRM` implementation:
///
/// - `LeastFavorable` corresponds to `"LF"`
/// - `ArpOnly` corresponds to `"ARP"`
///
/// The least-favorable mode is the current default because it matches the
/// study-facing conditional confidence-set path used elsewhere in this crate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RelativeMagnitudeHybrid {
    /// Least-favorable hybrid conditional confidence set (`"LF"` in R).
    LeastFavorable,
    /// ARP-only conditional confidence set (`"ARP"` in R).
    ArpOnly,
}

/// Hybrid mode for `DeltaSD` conditional confidence sets.
///
/// This matches the `hybrid_flag` argument used by the reference
/// `HonestDiD::computeConditionalCS_DeltaSD` implementation:
///
/// - `Flci` corresponds to `"FLCI"`
/// - `LeastFavorable` corresponds to `"LF"`
/// - `ArpOnly` corresponds to `"ARP"`
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SmoothnessHybrid {
    /// Conditional FLCI hybrid (`"FLCI"` in R).
    Flci,
    /// Least-favorable hybrid conditional confidence set (`"LF"` in R).
    LeastFavorable,
    /// ARP-only conditional confidence set (`"ARP"` in R).
    ArpOnly,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HonestBiasDirection {
    Positive,
    Negative,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HonestMonotonicityDirection {
    Increasing,
    Decreasing,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HonestRelativeMagnitudeBound {
    ParallelTrendsDeviation,
    LinearTrendDeviation,
}

/// Configuration for `DeltaSD` conditional confidence-set construction.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SmoothnessConfidenceSetConfig {
    /// Hybrid conditional confidence-set mode.
    pub hybrid: SmoothnessHybrid,
    /// First-stage hybrid size parameter `κ`.
    pub hybrid_kappa: f64,
    /// Whether ARP should exclude moments involving only pre periods.
    pub post_period_moments_only: bool,
    /// Number of points in the theta inversion grid.
    pub grid_points: usize,
}

impl SmoothnessConfidenceSetConfig {
    /// Build the default conditional-CS configuration for a target confidence
    /// level.
    #[must_use]
    pub fn from_inference(inference: InferenceConfig) -> Self {
        Self {
            hybrid: SmoothnessHybrid::Flci,
            hybrid_kappa: (1.0 - inference.confidence_level) / 10.0,
            post_period_moments_only: true,
            grid_points: 1_000,
        }
    }

    /// Validate the configuration against the requested confidence level.
    ///
    /// # Errors
    /// Returns an error if `hybrid_kappa` is not finite, outside `[0, alpha]`,
    /// or if the requested grid size is too small.
    pub fn validate(self, inference: InferenceConfig) -> Result<(), String> {
        let alpha = 1.0 - inference.confidence_level;
        if !self.hybrid_kappa.is_finite() {
            return Err("smoothness hybrid_kappa must be finite".to_string());
        }
        if !(0.0..=alpha).contains(&self.hybrid_kappa) {
            return Err(format!(
                "smoothness hybrid_kappa must lie in [0, alpha]={alpha}, got {}",
                self.hybrid_kappa
            ));
        }
        if self.grid_points < 2 {
            return Err(format!(
                "smoothness conditional grid_points must be at least 2, got {}",
                self.grid_points
            ));
        }
        Ok(())
    }
}

/// Configuration for `DeltaRM` conditional confidence-set construction.
///
/// The reference `HonestDiD` implementation exposes this as the pair
/// `(hybrid_flag, hybrid_kappa)`. We keep the same semantics, but express them
/// as a typed Rust configuration object.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RelativeMagnitudeConfidenceSetConfig {
    /// Hybrid conditional confidence-set mode.
    pub hybrid: RelativeMagnitudeHybrid,
    /// First-stage hybrid size parameter `κ`.
    pub hybrid_kappa: f64,
}

impl RelativeMagnitudeConfidenceSetConfig {
    /// Build the default conditional-CS configuration for a target confidence
    /// level.
    #[must_use]
    pub fn from_inference(inference: InferenceConfig) -> Self {
        Self {
            hybrid: RelativeMagnitudeHybrid::LeastFavorable,
            hybrid_kappa: (1.0 - inference.confidence_level) / 20.0,
        }
    }

    /// Validate the configuration against the requested confidence level.
    ///
    /// # Errors
    /// Returns an error if `hybrid_kappa` is not finite or is outside
    /// `[0, alpha]`, where `alpha = 1 - confidence_level`.
    pub fn validate(self, inference: InferenceConfig) -> Result<(), String> {
        let alpha = 1.0 - inference.confidence_level;
        if !self.hybrid_kappa.is_finite() {
            return Err("relative-magnitude hybrid_kappa must be finite".to_string());
        }
        if !(0.0..=alpha).contains(&self.hybrid_kappa) {
            return Err(format!(
                "relative-magnitude hybrid_kappa must lie in [0, alpha]={alpha}, got {}",
                self.hybrid_kappa
            ));
        }
        Ok(())
    }
}

/// Study-facing assessment of whether a target functional remains
/// distinguishable from a null value under a given sensitivity restriction.
#[derive(Debug, Clone, PartialEq)]
pub struct HonestAssessment {
    /// Original parallel-trends confidence interval.
    pub original_ci: (f64, f64),
    /// Robust `HonestDiD` confidence interval.
    pub robust_ci: (f64, f64),
    /// Identified set implied by the sensitivity restriction.
    pub identified_set: (f64, f64),
    /// Point estimate of the target functional.
    pub estimate: f64,
    /// Standard error of the target functional.
    pub se: f64,
    /// Null value used for the significance assessment.
    pub null_value: f64,
    /// Sensitivity parameter value (`M` or `Mbar`).
    pub sensitivity_value: f64,
    /// Whether the robust confidence interval excludes `null_value`.
    pub robustly_significant: bool,
}

/// One post-treatment point inside a conservative joint `HonestDiD` path region.
///
/// The interval is computed from a scalar `HonestDiD` assessment at an adjusted
/// confidence level so that the collection of intervals forms a family-wise
/// confidence region for the whole post-treatment path.
#[derive(Debug, Clone, PartialEq)]
pub struct HonestJointPathPoint {
    /// User-facing post-period label.
    pub post_period: i32,
    /// Scalar assessment for this post period under the joint-region
    /// confidence level.
    pub assessment: HonestAssessment,
}

/// Method used to calibrate a simultaneous joint post-treatment path region.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HonestJointPathMethod {
    /// Bonferroni-adjusted per-period intervals.
    Bonferroni,
    /// Covariance-aware simulated max-`t` critical value over the
    /// post-treatment covariance block.
    GaussianSimulated,
}

/// Configuration for simultaneous joint post-treatment path regions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HonestJointPathConfig {
    /// Method used to control family-wise error.
    pub method: HonestJointPathMethod,
    /// Number of Gaussian simulation draws when using
    /// [`HonestJointPathMethod::GaussianSimulated`].
    pub simulation_draws: usize,
    /// RNG seed used for simulated critical values.
    pub simulation_seed: u64,
}

impl HonestJointPathConfig {
    /// Default joint-path configuration for study-facing use.
    #[must_use]
    pub const fn default_for_production() -> Self {
        Self {
            method: HonestJointPathMethod::GaussianSimulated,
            simulation_draws: 20_000,
            simulation_seed: 42,
        }
    }
}

/// Simultaneous confidence region for the full post-treatment path.
///
/// This is implemented as a simultaneous rectangle over the post-treatment
/// coefficients. Depending on configuration, the rectangle is calibrated by a
/// conservative Bonferroni correction or by a tighter covariance-aware
/// simulated max-`t` critical value. It provides simultaneous coverage of the
/// whole post-treatment path at the requested confidence level, but it is
/// still not the tightest joint region obtainable from the full `HonestDiD`
/// optimization surface.
#[derive(Debug, Clone, PartialEq)]
pub struct HonestJointPathRegion {
    /// Requested family-wise confidence level.
    pub confidence_level: f64,
    /// Per-period confidence level actually used for the scalar intervals.
    pub pointwise_confidence_level: f64,
    /// Two-sided max-`|t|` critical value implied by the calibration.
    pub calibrated_max_t_critical_value: f64,
    /// Method used to calibrate the simultaneous region.
    pub method: HonestJointPathMethod,
    /// Joint region points in post-period order.
    pub points: Vec<HonestJointPathPoint>,
}

/// One directional functional inside a simultaneous multi-functional
/// `HonestDiD` region.
///
/// The functional is represented by post-treatment weights `post_weights`, applied to
/// `τ_post`.
#[derive(Debug, Clone, PartialEq)]
pub struct HonestDirectionalFunctionalPoint {
    /// Post-treatment linear-functional weights in `τ_post` order.
    pub post_weights: Vec<f64>,
    /// Scalar assessment for this functional under the simultaneous
    /// confidence level.
    pub assessment: HonestAssessment,
}

/// Simultaneous region over a finite set of post-treatment linear functionals.
///
/// This extends path-wise period intervals by allowing arbitrary direction sets
/// over `τ_post`. It is a practical approximation to the full optimization
/// surface: tighter than per-period rectangles when informative directions are
/// supplied, but still finite-direction rather than continuum-tight.
#[derive(Debug, Clone, PartialEq)]
pub struct HonestDirectionalRegion {
    /// Requested family-wise confidence level.
    pub confidence_level: f64,
    /// Scalar confidence level used for each directional assessment.
    pub pointwise_confidence_level: f64,
    /// Two-sided max-`|t|` critical value implied by the calibration.
    pub calibrated_max_t_critical_value: f64,
    /// Method used to calibrate the simultaneous region.
    pub method: HonestJointPathMethod,
    /// Directional functional assessments.
    pub points: Vec<HonestDirectionalFunctionalPoint>,
    /// Calibration and convergence diagnostics for this region.
    pub diagnostics: HonestDirectionalRegionDiagnostics,
}

/// Diagnostics emitted for directional/surface joint-region calibration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HonestDirectionalRegionDiagnostics {
    /// Number of directions used in the final calibration.
    pub direction_count: usize,
    /// Number of random directions used in the final calibration.
    pub random_direction_count: usize,
    /// Number of enrichment iterations performed.
    pub iterations_run: usize,
    /// Whether adaptive enrichment reported convergence before hitting limits.
    pub converged: bool,
}

impl HonestDirectionalRegionDiagnostics {
    #[must_use]
    pub const fn fixed(direction_count: usize, random_direction_count: usize) -> Self {
        Self {
            direction_count,
            random_direction_count,
            iterations_run: 1,
            converged: true,
        }
    }

    #[must_use]
    pub const fn adaptive(
        direction_count: usize,
        random_direction_count: usize,
        iterations_run: usize,
        converged: bool,
    ) -> Self {
        Self {
            direction_count,
            random_direction_count,
            iterations_run,
            converged,
        }
    }
}

/// One functional inside a simultaneous least-favorable confidence-interval
/// result.
#[derive(Debug, Clone, PartialEq)]
pub struct HonestMultiFlciPoint {
    /// Functional weights in post-period order.
    pub post_weights: Vec<f64>,
    /// Simultaneous least-favorable confidence interval.
    pub flci: (f64, f64),
    /// Original confidence interval.
    pub original_ci: (f64, f64),
    /// Identified set.
    pub identified_set: (f64, f64),
    /// Null value tested for robust significance.
    pub null_value: f64,
    /// Whether the simultaneous FLCI excludes `null_value`.
    pub robustly_significant: bool,
}

/// Simultaneous least-favorable result over a functional set.
#[derive(Debug, Clone, PartialEq)]
pub struct HonestMultiFlciResult {
    /// Requested family-wise confidence level.
    pub confidence_level: f64,
    /// Pointwise confidence level used for each scalar FLCI.
    pub pointwise_confidence_level: f64,
    /// Two-sided max-`|t|` critical value implied by the calibration.
    pub calibrated_max_t_critical_value: f64,
    /// Joint calibration method.
    pub method: HonestJointPathMethod,
    /// Functional points.
    pub points: Vec<HonestMultiFlciPoint>,
}

pub type RelativeMagnitudeMultiFlciPoint = HonestMultiFlciPoint;
pub type RelativeMagnitudeMultiFlciResult = HonestMultiFlciResult;
pub type SmoothnessMultiFlciPoint = HonestMultiFlciPoint;
pub type SmoothnessMultiFlciResult = HonestMultiFlciResult;

/// Direction-set construction for optimization-surface joint regions.
///
/// This controls which linear functionals are included when approximating the
/// full multi-period `HonestDiD` optimization surface via a finite directional
/// envelope.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HonestOptimizationSurfaceConfig {
    /// Include canonical basis directions `e_t` for all post periods.
    pub include_basis: bool,
    /// Include pairwise contrast directions `(e_i - e_j) / sqrt(2)`.
    pub include_pairwise_contrasts: bool,
    /// Number of random unit directions to sample.
    pub random_unit_directions: usize,
    /// RNG seed for random direction generation.
    pub random_seed: u64,
}

impl HonestOptimizationSurfaceConfig {
    /// Default reduced direction-set configuration for study-facing production use.
    ///
    /// This keeps a deterministic backbone of basis and pairwise-contrast
    /// directions, then adds a moderate random tail. It is intended to give a
    /// practical directional approximation without making full pipeline reruns
    /// dominated by optimization-surface work.
    #[must_use]
    pub const fn default_for_production() -> Self {
        Self {
            include_basis: true,
            include_pairwise_contrasts: true,
            random_unit_directions: 250,
            random_seed: 1729,
        }
    }

    /// Default dense direction-set configuration for explicit slow/full runs.
    #[must_use]
    pub const fn default_for_full_surface() -> Self {
        Self {
            include_basis: true,
            include_pairwise_contrasts: true,
            random_unit_directions: 2_000,
            random_seed: 1729,
        }
    }
}

/// Adaptive enrichment settings for optimization-surface direction sets.
///
/// The adaptive routine incrementally increases random directions and stops
/// when the calibrated pointwise confidence level stabilizes within
/// `pointwise_tolerance`, or when limits are reached.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HonestOptimizationSurfaceAdaptiveConfig {
    /// Number of random directions added per iteration.
    pub random_batch_size: usize,
    /// Minimum number of random directions before convergence checks apply.
    pub min_random_for_convergence: usize,
    /// Absolute convergence tolerance on calibrated pointwise confidence level.
    pub pointwise_tolerance: f64,
    /// Maximum number of enrichment iterations.
    pub max_iterations: usize,
}

impl HonestOptimizationSurfaceAdaptiveConfig {
    /// Default adaptive settings for study-facing production use.
    #[must_use]
    pub const fn default_for_production() -> Self {
        Self {
            random_batch_size: 100,
            min_random_for_convergence: 250,
            pointwise_tolerance: 5e-4,
            max_iterations: 6,
        }
    }

    /// Default adaptive settings for explicit slow/full runs.
    #[must_use]
    pub const fn default_for_full_surface() -> Self {
        Self {
            random_batch_size: 500,
            min_random_for_convergence: 1_000,
            pointwise_tolerance: 5e-4,
            max_iterations: 20,
        }
    }

    /// Validate adaptive settings.
    ///
    /// # Errors
    /// Returns an error if the adaptive parameters are invalid.
    pub fn validate(self) -> Result<(), String> {
        if self.random_batch_size == 0 {
            return Err(
                "optimization surface adaptive random_batch_size must be positive".to_string(),
            );
        }
        if self.max_iterations == 0 {
            return Err(
                "optimization surface adaptive max_iterations must be positive".to_string(),
            );
        }
        if !self.pointwise_tolerance.is_finite() || self.pointwise_tolerance < 0.0 {
            return Err(
                "optimization surface adaptive pointwise_tolerance must be finite and non-negative"
                    .to_string(),
            );
        }
        Ok(())
    }
}

/// Bundled configuration for the adaptive optimization-surface region API.
///
/// This groups the calibration and direction-set controls that previously
/// traveled as separate arguments, keeping the public API estimand-first while
/// avoiding oversized function signatures.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HonestOptimizationSurfaceAdaptiveRunConfig {
    /// `DeltaRM` conditional confidence-set construction settings.
    pub relative_magnitude: RelativeMagnitudeConfidenceSetConfig,
    /// Simultaneous calibration settings for the final region.
    pub joint: HonestJointPathConfig,
    /// Direction-set construction settings.
    pub surface: HonestOptimizationSurfaceConfig,
    /// Adaptive enrichment settings.
    pub adaptive: HonestOptimizationSurfaceAdaptiveConfig,
}

impl HonestOptimizationSurfaceAdaptiveRunConfig {
    /// Build study-facing defaults for the supplied inference level.
    #[must_use]
    pub fn from_inference(inference: InferenceConfig) -> Self {
        Self {
            relative_magnitude: RelativeMagnitudeConfidenceSetConfig::from_inference(inference),
            joint: HonestJointPathConfig::default_for_production(),
            surface: HonestOptimizationSurfaceConfig::default_for_production(),
            adaptive: HonestOptimizationSurfaceAdaptiveConfig::default_for_production(),
        }
    }
}

/// Direction-set strategy for the high-level `HonestDiD` workflow API.
#[derive(Debug, Clone, PartialEq)]
pub enum HonestWorkflowDirectionMode {
    /// Canonical basis directions over post periods.
    Basis,
    /// User-supplied custom directions.
    Custom(Vec<Vec<f64>>),
    /// Auto-generated finite optimization-surface directions.
    OptimizationSurface(HonestOptimizationSurfaceConfig),
    /// Auto-generated adaptive optimization-surface directions.
    OptimizationSurfaceAdaptive {
        surface: HonestOptimizationSurfaceConfig,
        adaptive: HonestOptimizationSurfaceAdaptiveConfig,
    },
}

impl HonestWorkflowDirectionMode {
    /// Default direction mode for study-facing production workflow use.
    #[must_use]
    pub const fn default_for_production() -> Self {
        Self::OptimizationSurfaceAdaptive {
            surface: HonestOptimizationSurfaceConfig::default_for_production(),
            adaptive: HonestOptimizationSurfaceAdaptiveConfig::default_for_production(),
        }
    }
}

/// Bundled configuration for the high-level `HonestDiD` workflow API.
#[derive(Debug, Clone, PartialEq)]
pub struct HonestWorkflowConfig {
    /// `DeltaRM` conditional confidence-set construction settings.
    pub relative_magnitude: RelativeMagnitudeConfidenceSetConfig,
    /// Simultaneous path calibration settings.
    pub joint: HonestJointPathConfig,
    /// Direction-set strategy for the directional region.
    pub direction_mode: HonestWorkflowDirectionMode,
}

impl HonestWorkflowConfig {
    /// Build study-facing defaults for the supplied inference level.
    #[must_use]
    pub fn from_inference(inference: InferenceConfig) -> Self {
        Self {
            relative_magnitude: RelativeMagnitudeConfidenceSetConfig::from_inference(inference),
            joint: HonestJointPathConfig::default_for_production(),
            direction_mode: HonestWorkflowDirectionMode::default_for_production(),
        }
    }
}

/// One functional assessment inside the high-level `HonestDiD` workflow
/// result.
#[derive(Debug, Clone, PartialEq)]
pub struct HonestFunctionalAssessmentPoint {
    /// Functional descriptor.
    pub functional: HonestPostFunctional,
    /// Scalar assessment for this functional.
    pub assessment: HonestAssessment,
}

/// Consolidated high-level `HonestDiD` workflow result.
#[derive(Debug, Clone, PartialEq)]
pub struct HonestStudyWorkflowResult {
    /// Scalar assessments for requested functionals.
    pub functional_assessments: Vec<HonestFunctionalAssessmentPoint>,
    /// Simultaneous joint post-treatment path region.
    pub joint_path_region: HonestJointPathRegion,
    /// Simultaneous directional region.
    pub directional_region: HonestDirectionalRegion,
    /// Optional simultaneous multi-functional least-favorable interval result.
    pub multi_flci: Option<HonestMultiFlciResult>,
}

/// Typed post-treatment linear functionals for event-study robustness
/// assessment.
///
/// These functionals are converted to the `post_weights` weight representation used
/// by the lower-level `HonestDiD` routines for a post-treatment parameter
/// vector `τ_post`.
#[derive(Debug, Clone, PartialEq)]
pub enum HonestPostFunctional {
    /// Select a single post-treatment period.
    Period(i32),
    /// Uniform average over an inclusive post-treatment window.
    AverageWindow { start_period: i32, end_period: i32 },
    /// Arbitrary weighted linear functional over named post periods.
    Weighted(Vec<(i32, f64)>),
}

impl HonestEventStudyInput {
    /// Validate internal dimensions and numeric finiteness.
    ///
    /// # Errors
    /// Returns an error if the coefficient vector, covariance matrix, or period
    /// labels are inconsistent.
    pub fn validate(&self) -> Result<(), String> {
        let total_periods = self.pre_periods.len() + self.post_periods.len();
        if self.betahat.len() != total_periods {
            return Err(format!(
                "event-study coefficient length {} does not match pre+post periods {}",
                self.betahat.len(),
                total_periods
            ));
        }
        if self.covariance.len() != total_periods
            || self.covariance.iter().any(|row| row.len() != total_periods)
        {
            return Err(
                "event-study covariance matrix dimensions do not match coefficients".to_string(),
            );
        }
        if self.betahat.iter().any(|value| !value.is_finite()) {
            return Err("event-study coefficients must be finite".to_string());
        }
        if self
            .covariance
            .iter()
            .flat_map(|row| row.iter())
            .any(|value| !value.is_finite())
        {
            return Err("event-study covariance values must be finite".to_string());
        }
        Ok(())
    }

    #[must_use]
    pub const fn num_pre_periods(&self) -> usize {
        self.pre_periods.len()
    }

    #[must_use]
    pub const fn num_post_periods(&self) -> usize {
        self.post_periods.len()
    }
}
