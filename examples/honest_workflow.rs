use did_methods::{HonestEventStudyInput, HonestWorkflowDirectionMode, InferenceConfig};
use did_methods::{
    HonestPostFunctional, HonestSensitivity, assess_honest_event_study_workflow,
    render_honest_directional_report_csv, render_honest_joint_path_report_csv,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let input = HonestEventStudyInput {
        betahat: vec![-0.05, -0.01, -0.02, -0.08, -0.12],
        covariance: vec![
            vec![0.010, 0.001, 0.000, 0.000, 0.000],
            vec![0.001, 0.009, 0.001, 0.000, 0.000],
            vec![0.000, 0.001, 0.010, 0.002, 0.001],
            vec![0.000, 0.000, 0.002, 0.012, 0.003],
            vec![0.000, 0.000, 0.001, 0.003, 0.013],
        ],
        pre_periods: vec![-2, -1],
        post_periods: vec![0, 1, 2],
    };

    let functionals = vec![
        HonestPostFunctional::Period(0),
        HonestPostFunctional::AverageWindow {
            start_period: 1,
            end_period: 2,
        },
    ];

    let workflow = assess_honest_event_study_workflow(
        &input,
        &functionals,
        HonestSensitivity::RelativeMagnitude(2.0),
        InferenceConfig::new(0.95),
        0.0,
    )?;

    println!("Functional assessments:");
    for point in &workflow.functional_assessments {
        println!(
            "{:?}: estimate={:.3} robust_ci=[{:.3}, {:.3}] robustly_significant={}",
            point.functional,
            point.assessment.estimate,
            point.assessment.robust_ci.0,
            point.assessment.robust_ci.1,
            point.assessment.robustly_significant
        );
    }

    println!(
        "\nJoint path points={} directional points={}",
        workflow.joint_path_region.points.len(),
        workflow.directional_region.points.len()
    );
    println!(
        "Directional mode default={:?}",
        HonestWorkflowDirectionMode::default_for_production()
    );

    let joint_csv = render_honest_joint_path_report_csv(&workflow.joint_path_region);
    let directional_csv = render_honest_directional_report_csv(&workflow.directional_region);
    println!(
        "\nCSV schemas: joint={} directional={}",
        joint_csv.lines().next().unwrap_or(""),
        directional_csv.lines().next().unwrap_or("")
    );

    Ok(())
}
