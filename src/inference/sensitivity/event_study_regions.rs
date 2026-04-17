#[path = "event_study_directional_region.rs"]
mod directional_region;
#[path = "event_study_joint_path.rs"]
mod joint_path;
#[path = "event_study_optimization_surface.rs"]
mod optimization_surface;

pub(in crate::inference::sensitivity::event_study_assessment) use directional_region::assess_honest_event_study_directional_region_with_optional_prepared_relative_magnitude;
pub use directional_region::{
    assess_honest_event_study_directional_region,
    assess_honest_event_study_directional_region_with_config,
};
pub(in crate::inference::sensitivity::event_study_assessment) use joint_path::assess_honest_event_study_joint_path_region_with_optional_prepared_relative_magnitude;
pub use joint_path::{
    assess_honest_event_study_joint_path_region,
    assess_honest_event_study_joint_path_region_with_config,
};
pub use optimization_surface::{
    assess_honest_event_study_optimization_surface_region,
    assess_honest_event_study_optimization_surface_region_adaptive,
    assess_honest_event_study_optimization_surface_region_adaptive_with_config,
    assess_honest_event_study_optimization_surface_region_with_config,
};
