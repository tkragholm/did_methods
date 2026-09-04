#[cfg(not(feature = "dense-lp"))]
mod conditional_moment_lp_workspace;
#[cfg(feature = "dense-lp")]
mod dense_simplex;
#[cfg(feature = "dense-lp")]
mod dense_workspaces;
mod dual_geometry;
mod least_favorable_critical_value;

#[cfg(not(feature = "dense-lp"))]
pub(in crate::inference::sensitivity) use conditional_moment_lp_workspace::ConditionalMomentLpWorkspace;
#[cfg(feature = "dense-lp")]
pub(in crate::inference::sensitivity) use dense_workspaces::ConditionalMomentLpWorkspace;
pub(in crate::inference::sensitivity) use dual_geometry::{
    DualMaxLpWorkspace, build_v_b_row_major_into, dual_acceptance_region, dual_conditional_test,
    recover_dual_vertex_from_binding, row_nonbinding_coeff_row_major_into,
};
pub(in crate::inference::sensitivity) use least_favorable_critical_value::{
    compute_least_favorable_cv, compute_least_favorable_cv_from_draws,
    compute_least_favorable_cv_uncached, simulation_draws,
};
