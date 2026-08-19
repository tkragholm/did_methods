//! `did_methods` crate architecture.
//!
//! - `types`: Public data models and configs.
//! - `methods`: Estimator families (`standard`, `drdid`, `att_gt`).
//! - `inference`: Shared CI/bootstrap/statistical helpers.
//! - `estimators`: Lower-level optimization and nuisance-model components.
//! - `dataset`: Unified data abstraction for `DiD` inference.

pub(crate) mod dataset;
pub(crate) mod error;
pub(crate) mod estimators;
pub mod inference;
pub(crate) mod methods;
#[cfg(test)]
mod tests;
pub(crate) mod types;
pub(crate) mod util;

pub use dataset::{
    Diagnostic, InferenceDataset, PanelRecord, Severity, ValidatedInferenceDataset,
    ValidationReport,
};
pub use error::{DatasetError, InternalDidError};
pub use estimators::common::basis::{BSplineBasis, PolynomialBasis};
pub use inference::{
    ClusterIndex, ClusterSpec, InferenceError, clustered_ci_from_index,
    clustered_standard_error_from_index, clustered_variance_from_index,
};
#[cfg(feature = "honest")]
pub use methods::att_gt::WindowAssessment;
pub use methods::att_gt::{
    AggteConfig, AggteError, AggteEstimate, AggteResult, AggteType, EventStudyResult,
    EventTimeResult, UnitPanel, aggregate_att_gt, aggregate_att_gt_by_calendar_time,
    aggregate_att_gt_by_cohort, aggregate_att_gt_event_time,
    aggregate_att_gt_event_time_with_influence, aggregate_att_gt_overall,
    att_gt_clustered_standard_errors, att_gt_mboot_bands, att_gt_simultaneous_bands,
    att_gt_simultaneous_bands_with_influence, estimate_att_gt, estimate_att_gt_dr,
    estimate_att_gt_dr_efficient_with_influence, estimate_att_gt_dr_panel,
    estimate_att_gt_dr_panel_with_influence, estimate_att_gt_dr_with_influence,
    estimate_att_gt_ipw, estimate_att_gt_ipw_with_influence, estimate_att_gt_or,
    estimate_att_gt_or_with_influence, row_panel, unit_panel,
};
pub use methods::continuous::sieve::estimate_acrt_sieve;
pub use methods::did_cc::{
    DidCcObservationView, estimate_did_cc_robust, estimate_did_cc_robust_from_views,
    estimate_did_cc_stationary, estimate_did_cc_stationary_from_views, test_did_cc_stationarity,
    test_did_cc_stationarity_from_views,
};
pub use methods::drdid::improved::{
    estimate_drdid_improved_panel, estimate_drdid_improved_repeated_cross_section,
};
pub use methods::drdid::repeated_efficient::estimate_drdid_repeated_efficient;
pub use methods::efficient::{
    aggregate_efficient_att_gt_by_calendar_time, aggregate_efficient_att_gt_by_cohort,
    aggregate_efficient_att_gt_event_time, aggregate_efficient_att_gt_event_time_with_influence,
    aggregate_efficient_att_gt_overall, estimate_att_gt_efficient, estimate_att_gt_efficient_dr,
};
pub use methods::triple_did::aggregator::estimate_dr_ddd;
// Historical compatibility export: this currently points to the improved /
// locally efficient panel estimator.
pub use inference::export::{
    InferenceCovarianceRow, InferenceInfluenceRow, build_clustered_inference_covariance_rows,
    build_inference_covariance_rows, build_inference_influence_rows,
};
#[cfg(feature = "honest")]
pub use inference::sensitivity::{
    HonestAssessment, HonestBiasDirection, HonestConditionalConfidenceSet,
    HonestDirectionalFunctionalPoint, HonestDirectionalRegion, HonestDirectionalRegionDiagnostics,
    HonestDirectionalReport, HonestDirectionalReportRow, HonestEventStudyInput,
    HonestEventStudyPoint, HonestFunctionalAssessmentPoint, HonestIdentifiedSet,
    HonestJointPathConfig, HonestJointPathMethod, HonestJointPathPoint, HonestJointPathRegion,
    HonestJointPathReport, HonestJointPathReportRow, HonestMonotonicityDirection,
    HonestMultiFlciReport, HonestMultiFlciReportRow, HonestOptimizationSurfaceAdaptiveConfig,
    HonestOptimizationSurfaceAdaptiveRunConfig, HonestOptimizationSurfaceConfig,
    HonestOriginalConfidenceSet, HonestPostFunctional, HonestRelativeMagnitudeBound, HonestResult,
    HonestSensitivity, HonestStudyWorkflowResult, HonestWorkflowConfig,
    HonestWorkflowDirectionMode, RelativeMagnitudeConfidenceSetConfig,
    RelativeMagnitudeFlciProblem, RelativeMagnitudeFlciResult, RelativeMagnitudeHybrid,
    RelativeMagnitudeMultiFlciPoint, RelativeMagnitudeMultiFlciProblem,
    RelativeMagnitudeMultiFlciResult, SensitivitySummary, SensitivitySummaryMethod,
    SensitivitySummaryRow, assess_honest_event_study_directional_region,
    assess_honest_event_study_directional_region_with_config, assess_honest_event_study_functional,
    assess_honest_event_study_functional_with_config, assess_honest_event_study_joint_path_region,
    assess_honest_event_study_joint_path_region_with_config,
    assess_honest_event_study_optimization_surface_region,
    assess_honest_event_study_optimization_surface_region_adaptive,
    assess_honest_event_study_optimization_surface_region_adaptive_with_config,
    assess_honest_event_study_optimization_surface_region_with_config,
    assess_honest_event_study_period, assess_honest_event_study_period_with_config,
    assess_honest_event_study_post_functional, assess_honest_event_study_post_functional_flci,
    assess_honest_event_study_post_functional_flci_with_config,
    assess_honest_event_study_post_functional_multi_flci,
    assess_honest_event_study_post_functional_multi_flci_with_config,
    assess_honest_event_study_post_functional_with_config, assess_honest_event_study_workflow,
    assess_honest_event_study_workflow_with_config, build_honest_directional_report_rows,
    build_honest_joint_path_report_rows, build_honest_multi_flci_report_rows,
    build_relative_magnitude_flci_problem, build_relative_magnitude_multi_flci_problem,
    compute_honest_ci_scalar, compute_honest_event_study_scalar,
    compute_honest_event_study_scalar_for_input, compute_original_confidence_set,
    compute_relative_magnitude_confidence_set,
    compute_relative_magnitude_confidence_set_with_config, compute_relative_magnitude_flci,
    compute_relative_magnitude_flci_with_config, compute_relative_magnitude_identified_set,
    compute_relative_magnitude_multi_flci, compute_relative_magnitude_multi_flci_with_config,
    estimate_smoothness_upper_bound_from_pretrends, render_honest_directional_report_csv,
    render_honest_directional_report_json, render_honest_joint_path_report_csv,
    render_honest_joint_path_report_json, render_honest_multi_flci_report_csv,
    render_honest_multi_flci_report_json, summarize_post_period_sensitivity,
    summarize_relative_magnitude_sensitivity, summarize_smoothness_sensitivity,
};
pub use methods::drdid::panel::estimate_drdid_panel;
pub use methods::drdid::repeated::estimate_drdid_repeated_cross_section;
pub use methods::standard::{
    aggregate_event_time, estimate_att_from_summary, estimate_att_two_by_two, summarize_two_by_two,
};
pub use types::*;
