use did_methods::{
    AttGtAggregationConfig, AttGtConfig, AttGtObservation, ComparisonGroup,
    aggregate_att_gt_event_time, estimate_att_gt,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let observations = vec![
        AttGtObservation::with_unit_id(1, Some(2002), 2001, 10.0),
        AttGtObservation::with_unit_id(1, Some(2002), 2002, 14.0),
        AttGtObservation::with_unit_id(2, Some(2002), 2001, 11.0),
        AttGtObservation::with_unit_id(2, Some(2002), 2002, 15.0),
        AttGtObservation::with_unit_id(3, None, 2001, 9.0),
        AttGtObservation::with_unit_id(3, None, 2002, 10.0),
        AttGtObservation::with_unit_id(4, None, 2001, 8.0),
        AttGtObservation::with_unit_id(4, None, 2002, 9.0),
    ];

    let config = AttGtConfig {
        comparison_group: ComparisonGroup::NeverTreated,
        ..AttGtConfig::default()
    };

    let estimates = estimate_att_gt(&observations, config)?;
    let event_time = aggregate_att_gt_event_time(&estimates, AttGtAggregationConfig::default())?;

    println!("ATT(g,t) estimates:");
    for estimate in &estimates {
        println!(
            "group={} time={} event_time={} att={:.3} ci=[{:.3}, {:.3}]",
            estimate.group,
            estimate.time,
            estimate.event_time,
            estimate.att,
            estimate.ci_low,
            estimate.ci_high
        );
    }

    println!("\nEvent-time aggregation:");
    for estimate in &event_time {
        println!(
            "event_time={} att={:.3} ci=[{:.3}, {:.3}]",
            estimate.event_time,
            estimate.summary.estimate,
            estimate.summary.ci_low,
            estimate.summary.ci_high
        );
    }

    Ok(())
}
