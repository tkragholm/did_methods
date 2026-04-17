use std::fmt::Write;

use serde::Serialize;

use super::{
    HonestDirectionalRegion, HonestJointPathMethod, HonestJointPathRegion, HonestMultiFlciResult,
};

const HONEST_REPORT_SCHEMA_V1: &str = "honestdid_joint_report_v1";

const fn method_label(method: HonestJointPathMethod) -> &'static str {
    match method {
        HonestJointPathMethod::Bonferroni => "bonferroni",
        HonestJointPathMethod::GaussianSimulated => "gaussian_simulated",
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct HonestJointPathReportRow {
    pub schema_version: &'static str,
    pub method: &'static str,
    pub confidence_level: f64,
    pub pointwise_confidence_level: f64,
    pub calibrated_max_t_critical_value: f64,
    pub post_period: i32,
    pub estimate: f64,
    pub se: f64,
    pub null_value: f64,
    pub sensitivity_value: f64,
    pub original_ci_low: f64,
    pub original_ci_high: f64,
    pub identified_lb: f64,
    pub identified_ub: f64,
    pub robust_ci_low: f64,
    pub robust_ci_high: f64,
    pub robustly_significant: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct HonestDirectionalReportRow {
    pub schema_version: &'static str,
    pub method: &'static str,
    pub confidence_level: f64,
    pub pointwise_confidence_level: f64,
    pub calibrated_max_t_critical_value: f64,
    pub direction_count: usize,
    pub random_direction_count: usize,
    pub iterations_run: usize,
    pub converged: bool,
    pub functional_id: usize,
    pub post_weights: Vec<f64>,
    pub estimate: f64,
    pub se: f64,
    pub null_value: f64,
    pub sensitivity_value: f64,
    pub original_ci_low: f64,
    pub original_ci_high: f64,
    pub identified_lb: f64,
    pub identified_ub: f64,
    pub robust_ci_low: f64,
    pub robust_ci_high: f64,
    pub robustly_significant: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct HonestMultiFlciReportRow {
    pub schema_version: &'static str,
    pub method: &'static str,
    pub confidence_level: f64,
    pub pointwise_confidence_level: f64,
    pub calibrated_max_t_critical_value: f64,
    pub functional_id: usize,
    pub post_weights: Vec<f64>,
    pub interval_low: f64,
    pub interval_high: f64,
    pub original_ci_low: f64,
    pub original_ci_high: f64,
    pub identified_lb: f64,
    pub identified_ub: f64,
    pub null_value: f64,
    pub robustly_significant: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct HonestJointPathReport {
    pub schema_version: &'static str,
    pub rows: Vec<HonestJointPathReportRow>,
}

#[derive(Debug, Clone, Serialize)]
pub struct HonestDirectionalReport {
    pub schema_version: &'static str,
    pub rows: Vec<HonestDirectionalReportRow>,
}

#[derive(Debug, Clone, Serialize)]
pub struct HonestMultiFlciReport {
    pub schema_version: &'static str,
    pub rows: Vec<HonestMultiFlciReportRow>,
}

#[must_use]
pub fn build_honest_joint_path_report_rows(
    region: &HonestJointPathRegion,
) -> Vec<HonestJointPathReportRow> {
    region
        .points
        .iter()
        .map(|point| HonestJointPathReportRow {
            schema_version: HONEST_REPORT_SCHEMA_V1,
            method: method_label(region.method),
            confidence_level: region.confidence_level,
            pointwise_confidence_level: region.pointwise_confidence_level,
            calibrated_max_t_critical_value: region.calibrated_max_t_critical_value,
            post_period: point.post_period,
            estimate: point.assessment.estimate,
            se: point.assessment.se,
            null_value: point.assessment.null_value,
            sensitivity_value: point.assessment.sensitivity_value,
            original_ci_low: point.assessment.original_ci.0,
            original_ci_high: point.assessment.original_ci.1,
            identified_lb: point.assessment.identified_set.0,
            identified_ub: point.assessment.identified_set.1,
            robust_ci_low: point.assessment.robust_ci.0,
            robust_ci_high: point.assessment.robust_ci.1,
            robustly_significant: point.assessment.robustly_significant,
        })
        .collect()
}

#[must_use]
pub fn build_honest_directional_report_rows(
    region: &HonestDirectionalRegion,
) -> Vec<HonestDirectionalReportRow> {
    region
        .points
        .iter()
        .enumerate()
        .map(|(functional_id, point)| HonestDirectionalReportRow {
            schema_version: HONEST_REPORT_SCHEMA_V1,
            method: method_label(region.method),
            confidence_level: region.confidence_level,
            pointwise_confidence_level: region.pointwise_confidence_level,
            calibrated_max_t_critical_value: region.calibrated_max_t_critical_value,
            direction_count: region.diagnostics.direction_count,
            random_direction_count: region.diagnostics.random_direction_count,
            iterations_run: region.diagnostics.iterations_run,
            converged: region.diagnostics.converged,
            functional_id,
            post_weights: point.post_weights.clone(),
            estimate: point.assessment.estimate,
            se: point.assessment.se,
            null_value: point.assessment.null_value,
            sensitivity_value: point.assessment.sensitivity_value,
            original_ci_low: point.assessment.original_ci.0,
            original_ci_high: point.assessment.original_ci.1,
            identified_lb: point.assessment.identified_set.0,
            identified_ub: point.assessment.identified_set.1,
            robust_ci_low: point.assessment.robust_ci.0,
            robust_ci_high: point.assessment.robust_ci.1,
            robustly_significant: point.assessment.robustly_significant,
        })
        .collect()
}

#[must_use]
pub fn build_honest_multi_flci_report_rows(
    result: &HonestMultiFlciResult,
) -> Vec<HonestMultiFlciReportRow> {
    result
        .points
        .iter()
        .enumerate()
        .map(|(functional_id, point)| HonestMultiFlciReportRow {
            schema_version: HONEST_REPORT_SCHEMA_V1,
            method: method_label(result.method),
            confidence_level: result.confidence_level,
            pointwise_confidence_level: result.pointwise_confidence_level,
            calibrated_max_t_critical_value: result.calibrated_max_t_critical_value,
            functional_id,
            post_weights: point.post_weights.clone(),
            interval_low: point.flci.0,
            interval_high: point.flci.1,
            original_ci_low: point.original_ci.0,
            original_ci_high: point.original_ci.1,
            identified_lb: point.identified_set.0,
            identified_ub: point.identified_set.1,
            null_value: point.null_value,
            robustly_significant: point.robustly_significant,
        })
        .collect()
}

#[must_use]
pub fn render_honest_joint_path_report_csv(region: &HonestJointPathRegion) -> String {
    let rows = build_honest_joint_path_report_rows(region);
    let mut out = String::from(
        "schema_version,method,confidence_level,pointwise_confidence_level,calibrated_max_t_critical_value,post_period,estimate,se,null_value,sensitivity_value,original_ci_low,original_ci_high,identified_lb,identified_ub,robust_ci_low,robust_ci_high,robustly_significant\n",
    );
    for row in rows {
        let _ = writeln!(
            out,
            "{},{},{:.6},{:.6},{:.6},{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{}",
            row.schema_version,
            row.method,
            row.confidence_level,
            row.pointwise_confidence_level,
            row.calibrated_max_t_critical_value,
            row.post_period,
            row.estimate,
            row.se,
            row.null_value,
            row.sensitivity_value,
            row.original_ci_low,
            row.original_ci_high,
            row.identified_lb,
            row.identified_ub,
            row.robust_ci_low,
            row.robust_ci_high,
            row.robustly_significant
        );
    }
    out
}

#[must_use]
pub fn render_honest_directional_report_csv(region: &HonestDirectionalRegion) -> String {
    let rows = build_honest_directional_report_rows(region);
    let mut out = String::from(
        "schema_version,method,confidence_level,pointwise_confidence_level,calibrated_max_t_critical_value,direction_count,random_direction_count,iterations_run,converged,functional_id,estimate,se,null_value,sensitivity_value,original_ci_low,original_ci_high,identified_lb,identified_ub,robust_ci_low,robust_ci_high,robustly_significant\n",
    );
    for row in rows {
        let _ = writeln!(
            out,
            "{},{},{:.6},{:.6},{:.6},{},{},{},{},{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{}",
            row.schema_version,
            row.method,
            row.confidence_level,
            row.pointwise_confidence_level,
            row.calibrated_max_t_critical_value,
            row.direction_count,
            row.random_direction_count,
            row.iterations_run,
            row.converged,
            row.functional_id,
            row.estimate,
            row.se,
            row.null_value,
            row.sensitivity_value,
            row.original_ci_low,
            row.original_ci_high,
            row.identified_lb,
            row.identified_ub,
            row.robust_ci_low,
            row.robust_ci_high,
            row.robustly_significant
        );
    }
    out
}

#[must_use]
pub fn render_honest_multi_flci_report_csv(result: &HonestMultiFlciResult) -> String {
    let rows = build_honest_multi_flci_report_rows(result);
    let mut out = String::from(
        "schema_version,method,confidence_level,pointwise_confidence_level,calibrated_max_t_critical_value,functional_id,interval_low,interval_high,original_ci_low,original_ci_high,identified_lb,identified_ub,null_value,robustly_significant\n",
    );
    for row in rows {
        let _ = writeln!(
            out,
            "{},{},{:.6},{:.6},{:.6},{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{}",
            row.schema_version,
            row.method,
            row.confidence_level,
            row.pointwise_confidence_level,
            row.calibrated_max_t_critical_value,
            row.functional_id,
            row.interval_low,
            row.interval_high,
            row.original_ci_low,
            row.original_ci_high,
            row.identified_lb,
            row.identified_ub,
            row.null_value,
            row.robustly_significant
        );
    }
    out
}

/// Render a JSON joint-path report payload.
///
/// # Errors
/// Returns an error if JSON serialization fails.
pub fn render_honest_joint_path_report_json(
    region: &HonestJointPathRegion,
) -> Result<String, String> {
    serde_json::to_string_pretty(&HonestJointPathReport {
        schema_version: HONEST_REPORT_SCHEMA_V1,
        rows: build_honest_joint_path_report_rows(region),
    })
    .map_err(|err| format!("failed to serialize joint path report json: {err}"))
}

/// Render a JSON directional report payload.
///
/// # Errors
/// Returns an error if JSON serialization fails.
pub fn render_honest_directional_report_json(
    region: &HonestDirectionalRegion,
) -> Result<String, String> {
    serde_json::to_string_pretty(&HonestDirectionalReport {
        schema_version: HONEST_REPORT_SCHEMA_V1,
        rows: build_honest_directional_report_rows(region),
    })
    .map_err(|err| format!("failed to serialize directional report json: {err}"))
}

/// Render a JSON multi-functional least-favorable interval report payload.
///
/// # Errors
/// Returns an error if JSON serialization fails.
pub fn render_honest_multi_flci_report_json(
    result: &HonestMultiFlciResult,
) -> Result<String, String> {
    serde_json::to_string_pretty(&HonestMultiFlciReport {
        schema_version: HONEST_REPORT_SCHEMA_V1,
        rows: build_honest_multi_flci_report_rows(result),
    })
    .map_err(|err| format!("failed to serialize honest multi-flci report json: {err}"))
}
