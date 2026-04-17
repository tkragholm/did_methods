use did_methods::{DatasetError, InferenceDataset, PanelRecord};

struct MockRow {
    id: String,
    period: i32,
    treated: bool,
    first_treated: Option<i32>,
    outcome: f64,
}

impl PanelRecord for MockRow {
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
        self.first_treated
    }
    fn outcome(&self) -> f64 {
        self.outcome
    }
}

#[test]
fn test_dataset_basic_validation() {
    let panel = vec![
        MockRow {
            id: "1".into(),
            period: 2020,
            treated: false,
            first_treated: None,
            outcome: 10.0,
        },
        MockRow {
            id: "1".into(),
            period: 2021,
            treated: false,
            first_treated: None,
            outcome: 12.0,
        },
        MockRow {
            id: "2".into(),
            period: 2020,
            treated: true,
            first_treated: Some(2020),
            outcome: 5.0,
        },
        MockRow {
            id: "2".into(),
            period: 2021,
            treated: true,
            first_treated: Some(2020),
            outcome: 8.0,
        },
    ];

    let dataset = InferenceDataset::from_panel(panel).validate();
    assert!(dataset.is_ok());
}

#[test]
fn test_dataset_unbalanced_panel_error() {
    let panel = vec![
        MockRow {
            id: "1".into(),
            period: 2020,
            treated: false,
            first_treated: None,
            outcome: 10.0,
        },
        MockRow {
            id: "2".into(),
            period: 2020,
            treated: true,
            first_treated: Some(2020),
            outcome: 5.0,
        },
        MockRow {
            id: "2".into(),
            period: 2021,
            treated: true,
            first_treated: Some(2020),
            outcome: 8.0,
        },
    ];

    let dataset = InferenceDataset::from_panel(panel).validate();
    assert!(matches!(dataset, Err(DatasetError::UnbalancedPanel { .. })));
}

#[test]
fn test_dataset_inconsistent_treatment_error() {
    let panel = vec![
        MockRow {
            id: "1".into(),
            period: 2020,
            treated: false,
            first_treated: Some(2020),
            outcome: 10.0,
        },
        MockRow {
            id: "1".into(),
            period: 2021,
            treated: false,
            first_treated: Some(2021),
            outcome: 12.0,
        },
    ];

    let dataset = InferenceDataset::from_panel(panel).validate();
    assert!(matches!(
        dataset,
        Err(DatasetError::InvalidTreatmentTiming { .. })
    ));
}
