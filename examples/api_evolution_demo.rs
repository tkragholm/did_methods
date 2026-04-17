use did_methods::{
    ClusterSpec, EventStudyConfig, HonestSensitivity, InferenceDataset, PanelRecord,
};

struct MyRecord {
    id: String,
    period: i32,
    treated: bool,
    ft: Option<i32>,
    outcome: f64,
}

impl PanelRecord for MyRecord {
    type UnitId = String;
    fn unit_id(&self) -> &Self::UnitId {
        &self.id
    }
    fn time_period(&self) -> i32 {
        self.period
    }
    fn treatment_status(&self) -> bool {
        self.treated
    }
    fn first_treated_time(&self) -> Option<i32> {
        self.ft
    }
    fn outcome(&self) -> f64 {
        self.outcome
    }
}

fn main() {
    let records = vec![
        MyRecord {
            id: "1".into(),
            period: 1,
            treated: false,
            ft: Some(2),
            outcome: 10.0,
        },
        MyRecord {
            id: "1".into(),
            period: 2,
            treated: true,
            ft: Some(2),
            outcome: 12.0,
        },
        MyRecord {
            id: "2".into(),
            period: 1,
            treated: false,
            ft: None,
            outcome: 5.0,
        },
        MyRecord {
            id: "2".into(),
            period: 2,
            treated: false,
            ft: None,
            outcome: 5.5,
        },
    ];

    let dataset = InferenceDataset::from_panel(records)
        .with_clusters(ClusterSpec::one_way(vec!["C1", "C1", "C2", "C2"]))
        .validate()
        .expect("validation failed");

    let config = EventStudyConfig::default();
    let result = dataset
        .estimate_att_gt_dr(&config)
        .expect("estimation failed");

    println!("Estimates: {}", result.estimates.len());

    let event_time_res = result.aggregate_event_time().expect("aggregation failed");
    println!("Event-time points: {}", event_time_res.estimates.len());

    for est in &event_time_res.estimates {
        println!(
            "Event time {}: estimate={}, se={}",
            est.event_time, est.summary.estimate, est.summary.se
        );
    }

    let honest_config = did_methods::HonestWorkflowConfig::from_inference(config.inference());
    let sensitivity = event_time_res
        .assess_window(
            0,
            1,
            HonestSensitivity::RelativeMagnitude(1.0),
            honest_config,
        )
        .expect("sensitivity failed");

    println!(
        "Robust CI: [{}, {}]",
        sensitivity.assessment.robust_ci.0, sensitivity.assessment.robust_ci.1
    );
}
