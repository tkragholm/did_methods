//! The columns of a design matrix that one fit can identify.
//!
//! A doubly robust fit regresses the outcome on the covariates within the
//! comparison arm and models treatment on them across both arms. Either model
//! is singular when a column is constant, or a combination of the others,
//! inside the rows that model reads: a rare indicator whose only holders are
//! treated is all zero in the comparison arm, and eleven region dummies whose
//! reference level has no member in a cell sum to the intercept.
//!
//! A caller that prunes its covariates against the whole sample cannot see
//! this. Study I's staggered estimator fits one design per ATT(g,t) cell on
//! the units observed at both of the cell's periods, so a column that varies
//! in the slice is all zero in every cell that none of its holders reaches,
//! and on the 4 September 2026 run that was 315 of the pooled mothers' 325
//! cells: the survivors were exactly the cells one comparator holder of one
//! rare indicator was observed in. The pairwise route prunes per pair for
//! the same reason and never lost one.
//!
//! So the pruning is done here, per fit, on the rows the fit reads. A column
//! is kept when it is linearly independent of the intercept and the columns
//! kept before it within EVERY group the fit models separately, so that the
//! outcome regression on any one group is full rank and no combination of
//! columns separates the arms. Greedy in column order: which of a dependent
//! set survives is then a property of the caller's column order and not of
//! the rows.
//!
//! Dropping such a column changes nothing the fit could have estimated. A
//! column constant within a group carries no information about that group's
//! outcome; a column that is a combination of kept ones gives the same fitted
//! values without it. R's `did` returns `NA` for the cell instead.
//!
//! [`independent_design_columns`] applies the same rule for a caller, either
//! per arm, as the panel estimator does, or on the pooled rows, which is the
//! weaker test a caller pruning a whole sample before estimation applies. A
//! column that passes the pooled test can still fail within one arm, and the
//! estimator then drops it as well.

use crate::types::DrDidError;

/// The rows on which a design column has to be identified to be kept.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DesignPruning<'a> {
    /// Within each arm separately, `treated` giving each row's arm. This is
    /// the rule the panel estimator applies to every fit.
    PerArm { treated: &'a [bool] },
    /// On all rows together.
    Pooled,
}

/// A design reduced to the columns every group can identify.
pub(super) struct PrunedDesign {
    pub design_flat: Vec<f64>,
    pub feature_count: usize,
    /// How many of the caller's columns were dropped. Zero on a full-rank
    /// design, where `design_flat` is the input unchanged.
    pub dropped: usize,
    /// One flag per input column, true where the column was kept.
    pub kept: Vec<bool>,
}

/// The relative residual below which a column counts as dependent: the share
/// of its own weighted sum of squares left after projecting out the columns
/// kept before it. `1e-8` on the Gram is a relative column norm of `1e-4`,
/// which keeps a condition number the Cholesky downstream handles and drops
/// what floating point cannot tell from exact dependence.
pub(super) const DEPENDENCE_TOLERANCE: f64 = 1e-8;

/// The columns of a design that are linearly independent of the intercept and
/// of the columns kept before them, under `rule`.
///
/// `design_flat` is row-major with `feature_count` columns per row and the
/// intercept in column 0, which is always kept. `weights` are the rows'
/// weights in the Gram matrix `X'WX`. A column is dropped when its own entry
/// in that matrix is not positive, or when the share of it left after
/// projecting out the kept columns is at most `tolerance`; the estimators use
/// `1e-8`. Columns are tested in order, so of a dependent set the first is
/// kept. Under [`DesignPruning::PerArm`] a column must pass in each arm that
/// has rows.
///
/// Returns one flag per column, true where it is kept.
///
/// # Errors
///
/// [`DrDidError::InconsistentCovariateCount`] when `design_flat`, `weights` or
/// the arm flags do not match the row count, and [`DrDidError::InvalidConfig`]
/// for a `feature_count` of zero or a negative or non-finite `tolerance`.
pub fn independent_design_columns(
    design_flat: &[f64],
    feature_count: usize,
    weights: &[f64],
    rule: DesignPruning<'_>,
    tolerance: f64,
) -> Result<Vec<bool>, DrDidError> {
    if feature_count == 0 {
        return Err(DrDidError::InvalidConfig(
            "feature_count must be > 0".to_string(),
        ));
    }
    if !tolerance.is_finite() || tolerance < 0.0 {
        return Err(DrDidError::InvalidConfig(
            "tolerance must be finite and non-negative".to_string(),
        ));
    }
    let row_count = weights.len();
    if design_flat.len() != row_count * feature_count {
        return Err(DrDidError::InconsistentCovariateCount {
            expected: row_count * feature_count,
            actual: design_flat.len(),
        });
    }
    let group: Vec<usize> = match rule {
        DesignPruning::PerArm { treated } => {
            if treated.len() != row_count {
                return Err(DrDidError::InconsistentCovariateCount {
                    expected: row_count,
                    actual: treated.len(),
                });
            }
            treated.iter().map(|&t| usize::from(t)).collect()
        }
        DesignPruning::Pooled => vec![0; row_count],
    };
    Ok(prune_design(design_flat, feature_count, &group, weights, tolerance).kept)
}

/// Prune `design_flat` (row-major, `feature_count` per row, column 0 the
/// intercept) to the columns independent within every group.
///
/// `group` labels each row; `weights` are the rows' sampling weights. The
/// intercept is always kept.
pub(super) fn prune_design(
    design_flat: &[f64],
    feature_count: usize,
    group: &[usize],
    weights: &[f64],
    tolerance: f64,
) -> PrunedDesign {
    let row_count = group.len();
    if feature_count <= 1 || row_count == 0 {
        return PrunedDesign {
            design_flat: design_flat.to_vec(),
            feature_count,
            dropped: 0,
            kept: vec![true; feature_count],
        };
    }
    let group_count = group.iter().max().map_or(0, |g| g + 1);

    // One weighted Gram per group, upper triangle in full storage.
    let size = feature_count * feature_count;
    let mut grams = vec![0.0; group_count * size];
    for row in 0..row_count {
        let gram = &mut grams[group[row] * size..(group[row] + 1) * size];
        let x = &design_flat[row * feature_count..(row + 1) * feature_count];
        let w = weights[row];
        for i in 0..feature_count {
            let wi = w * x[i];
            for j in i..feature_count {
                gram[i * feature_count + j] += wi * x[j];
            }
        }
    }
    let entry = |gram: &[f64], i: usize, j: usize| {
        let (a, b) = if i <= j { (i, j) } else { (j, i) };
        gram[a * feature_count + b]
    };

    // Greedy Cholesky per group over the kept set. A candidate joins the kept
    // set only if it is independent in every group.
    let mut kept: Vec<usize> = Vec::with_capacity(feature_count);
    let mut factors: Vec<Vec<Vec<f64>>> = vec![Vec::new(); group_count];
    for candidate in 0..feature_count {
        let mut projections: Vec<Vec<f64>> = Vec::with_capacity(group_count);
        let mut independent = true;
        for g in 0..group_count {
            let gram = &grams[g * size..(g + 1) * size];
            let own = entry(gram, candidate, candidate);
            if own <= 0.0 {
                // The intercept is never zero in a group with rows; a group
                // without rows says nothing about the column.
                if gram[0] > 0.0 {
                    independent = false;
                }
                projections.push(Vec::new());
                continue;
            }
            let factor = &factors[g];
            let mut projection = vec![0.0; kept.len()];
            for (row, &k) in kept.iter().enumerate() {
                let mut total = entry(gram, candidate, k);
                for col in 0..row {
                    total -= factor[row][col] * projection[col];
                }
                projection[row] = total / factor[row][row];
            }
            let residual = own - projection.iter().map(|v| v * v).sum::<f64>();
            if candidate > 0 && residual <= tolerance * own {
                independent = false;
            }
            projection.push(residual.max(0.0).sqrt());
            projections.push(projection);
        }
        if !independent {
            continue;
        }
        for (g, projection) in projections.into_iter().enumerate() {
            if !projection.is_empty() {
                factors[g].push(projection);
            }
        }
        kept.push(candidate);
    }

    let dropped = feature_count - kept.len();
    let mut kept_flags = vec![false; feature_count];
    for &k in &kept {
        kept_flags[k] = true;
    }
    if dropped == 0 {
        return PrunedDesign {
            design_flat: design_flat.to_vec(),
            feature_count,
            dropped: 0,
            kept: kept_flags,
        };
    }
    let mut pruned = Vec::with_capacity(row_count * kept.len());
    for row in 0..row_count {
        let x = &design_flat[row * feature_count..(row + 1) * feature_count];
        pruned.extend(kept.iter().map(|&k| x[k]));
    }
    PrunedDesign {
        design_flat: pruned,
        feature_count: kept.len(),
        dropped,
        kept: kept_flags,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn design(rows: &[&[f64]]) -> (Vec<f64>, usize) {
        let k = rows[0].len();
        (rows.iter().flat_map(|r| r.iter().copied()).collect(), k)
    }

    #[test]
    fn a_full_rank_design_is_returned_unchanged() {
        let (flat, k) = design(&[
            &[1.0, 0.5, 1.0],
            &[1.0, 1.5, 0.0],
            &[1.0, 0.7, 0.0],
            &[1.0, -0.5, 1.0],
            &[1.0, 2.0, 0.0],
            &[1.0, 0.1, 1.0],
        ]);
        let group = [0, 0, 0, 1, 1, 1];
        let out = prune_design(&flat, k, &group, &[1.0; 6], DEPENDENCE_TOLERANCE);
        assert_eq!(out.dropped, 0);
        assert_eq!(out.feature_count, 3);
        assert_eq!(out.design_flat, flat);
    }

    #[test]
    fn a_column_constant_within_one_group_is_dropped() {
        // Column 2 is held by treated rows (group 1) only: zero in the
        // comparison arm, where the outcome regression would be singular.
        let (flat, k) = design(&[
            &[1.0, 0.5, 0.0],
            &[1.0, 1.5, 0.0],
            &[1.0, -0.5, 1.0],
            &[1.0, 2.0, 0.0],
        ]);
        let group = [0, 0, 1, 1];
        let out = prune_design(&flat, k, &group, &[1.0; 4], DEPENDENCE_TOLERANCE);
        assert_eq!(out.dropped, 1);
        assert_eq!(out.feature_count, 2);
        assert_eq!(
            out.design_flat,
            vec![1.0, 0.5, 1.0, 1.5, 1.0, -0.5, 1.0, 2.0]
        );
    }

    #[test]
    fn dummies_that_sum_to_the_intercept_lose_their_last_member() {
        // Three dummies covering every row: the dummy-variable trap. The
        // first two are kept in order, the third is their complement.
        let (flat, k) = design(&[
            &[1.0, 1.0, 0.0, 0.0],
            &[1.0, 0.0, 1.0, 0.0],
            &[1.0, 0.0, 0.0, 1.0],
            &[1.0, 1.0, 0.0, 0.0],
            &[1.0, 0.0, 1.0, 0.0],
            &[1.0, 0.0, 0.0, 1.0],
        ]);
        let group = [0, 0, 0, 1, 1, 1];
        let out = prune_design(
            &flat,
            k,
            &group,
            &[1.0, 2.0, 1.0, 1.0, 1.0, 3.0],
            DEPENDENCE_TOLERANCE,
        );
        assert_eq!(out.dropped, 1);
        assert_eq!(out.feature_count, 3);
        assert_eq!(&out.design_flat[..3], &[1.0, 1.0, 0.0]);
    }

    #[test]
    fn the_pooled_rule_keeps_a_column_that_one_arm_cannot_identify() {
        // Column 2 is zero among the comparators and varies among the treated.
        let (flat, k) = design(&[
            &[1.0, 0.5, 0.0],
            &[1.0, 1.5, 0.0],
            &[1.0, 0.7, 0.0],
            &[1.0, -0.5, 1.0],
            &[1.0, 2.0, 0.0],
            &[1.0, 0.1, 1.0],
        ]);
        let treated = [false, false, false, true, true, true];
        let weights = [1.0; 6];
        let per_arm = independent_design_columns(
            &flat,
            k,
            &weights,
            DesignPruning::PerArm { treated: &treated },
            DEPENDENCE_TOLERANCE,
        )
        .unwrap();
        let pooled = independent_design_columns(
            &flat,
            k,
            &weights,
            DesignPruning::Pooled,
            DEPENDENCE_TOLERANCE,
        )
        .unwrap();
        assert_eq!(per_arm, vec![true, true, false]);
        assert_eq!(pooled, vec![true, true, true]);
    }

    #[test]
    fn the_per_arm_rule_is_the_one_the_estimators_apply() {
        let (flat, k) = design(&[
            &[1.0, 1.0, 0.0, 0.0, 0.3],
            &[1.0, 0.0, 1.0, 0.0, 0.0],
            &[1.0, 0.0, 0.0, 1.0, 0.0],
            &[1.0, 1.0, 0.0, 0.0, 1.0],
            &[1.0, 0.0, 1.0, 0.0, 0.2],
            &[1.0, 0.0, 0.0, 1.0, 0.0],
        ]);
        let treated = [false, false, false, true, true, true];
        let weights = [1.0, 2.0, 1.0, 1.0, 1.0, 3.0];
        let mask = independent_design_columns(
            &flat,
            k,
            &weights,
            DesignPruning::PerArm { treated: &treated },
            DEPENDENCE_TOLERANCE,
        )
        .unwrap();
        let groups: Vec<usize> = treated.iter().map(|&t| usize::from(t)).collect();
        let pruned = prune_design(&flat, k, &groups, &weights, DEPENDENCE_TOLERANCE);
        assert_eq!(mask, pruned.kept);
        assert_eq!(mask, vec![true, true, true, false, false]);
    }

    #[test]
    fn the_tolerance_decides_a_near_dependence() {
        let (flat, k) = design(&[
            &[1.0, 0.0, 0.0],
            &[1.0, 1.0, 1.001],
            &[1.0, 2.0, 2.0],
            &[1.0, 3.0, 3.0],
        ]);
        let weights = [1.0; 4];
        let loose =
            independent_design_columns(&flat, k, &weights, DesignPruning::Pooled, 1e-3).unwrap();
        let tight =
            independent_design_columns(&flat, k, &weights, DesignPruning::Pooled, 1e-12).unwrap();
        assert_eq!(loose, vec![true, true, false]);
        assert_eq!(tight, vec![true, true, true]);
    }

    #[test]
    fn inputs_that_do_not_fit_together_are_refused() {
        let flat = [1.0, 0.5, 1.0, 1.5];
        assert!(matches!(
            independent_design_columns(&flat, 2, &[1.0; 3], DesignPruning::Pooled, 1e-8),
            Err(DrDidError::InconsistentCovariateCount { .. })
        ));
        assert!(matches!(
            independent_design_columns(
                &flat,
                2,
                &[1.0; 2],
                DesignPruning::PerArm { treated: &[true] },
                1e-8
            ),
            Err(DrDidError::InconsistentCovariateCount { .. })
        ));
        assert!(matches!(
            independent_design_columns(&flat, 2, &[1.0; 2], DesignPruning::Pooled, f64::NAN),
            Err(DrDidError::InvalidConfig(_))
        ));
        assert!(matches!(
            independent_design_columns(&flat, 0, &[1.0; 2], DesignPruning::Pooled, 1e-8),
            Err(DrDidError::InvalidConfig(_))
        ));
    }

    #[test]
    fn an_all_zero_column_is_dropped_and_the_intercept_never_is() {
        let (flat, k) = design(&[&[1.0, 0.0], &[1.0, 0.0], &[1.0, 0.0]]);
        let out = prune_design(&flat, k, &[0, 0, 1], &[1.0; 3], DEPENDENCE_TOLERANCE);
        assert_eq!(out.dropped, 1);
        assert_eq!(out.feature_count, 1);
        assert_eq!(out.design_flat, vec![1.0; 3]);
    }
}
