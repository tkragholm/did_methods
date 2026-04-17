#[derive(Clone, Copy)]
pub struct AcceptedGridRange {
    pub(crate) lower_idx: usize,
    pub(crate) upper_idx: usize,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AdaptiveGridDiagnostics {
    pub anchor_idx: usize,
    pub accepted_anchor_idx: Option<usize>,
    pub unique_evaluations: usize,
    pub cache_hits: usize,
    pub anchor_accepted: bool,
    pub used_nearest_accepted_anchor: bool,
    pub used_full_grid_fallback: bool,
}

pub fn nearest_grid_index(grid: &[f64], value: f64) -> usize {
    grid.iter()
        .enumerate()
        .min_by(|(_, left), (_, right)| (**left - value).abs().total_cmp(&(**right - value).abs()))
        .map_or(0, |(idx, _)| idx)
}

pub fn accepted_grid_range_from_flags(flags: &[bool]) -> Option<AcceptedGridRange> {
    let lower_idx = flags.iter().position(|accepted| *accepted)?;
    let upper_idx = flags.iter().rposition(|accepted| *accepted)?;
    Some(AcceptedGridRange {
        lower_idx,
        upper_idx,
    })
}

pub fn compute_accepted_grid_range_adaptive<F>(
    grid: &[f64],
    anchor_idx: usize,
    evaluate: F,
) -> Result<Option<AcceptedGridRange>, String>
where
    F: FnMut(f64) -> Result<bool, String>,
{
    compute_accepted_grid_range_adaptive_with_diagnostics(grid, anchor_idx, evaluate)
        .map(|(range, _)| range)
}

#[expect(
    clippy::too_many_lines,
    reason = "adaptive grid logic keeps acceptance-search and fallback invariants together"
)]
pub fn compute_accepted_grid_range_adaptive_with_diagnostics<F>(
    grid: &[f64],
    anchor_idx: usize,
    mut evaluate: F,
) -> Result<(Option<AcceptedGridRange>, AdaptiveGridDiagnostics), String>
where
    F: FnMut(f64) -> Result<bool, String>,
{
    let mut cache = vec![None; grid.len()];
    let mut diagnostics = AdaptiveGridDiagnostics {
        anchor_idx: anchor_idx.min(grid.len().saturating_sub(1)),
        ..AdaptiveGridDiagnostics::default()
    };
    let mut evaluate_idx = |idx: usize| -> Result<bool, String> {
        if let Some(value) = cache[idx] {
            diagnostics.cache_hits = diagnostics.cache_hits.saturating_add(1);
            return Ok(value);
        }
        let accepted = evaluate(grid[idx])?;
        cache[idx] = Some(accepted);
        diagnostics.unique_evaluations = diagnostics.unique_evaluations.saturating_add(1);
        Ok(accepted)
    };
    let anchor_idx = diagnostics.anchor_idx;
    let accepted_anchor_idx = if evaluate_idx(anchor_idx)? {
        diagnostics.anchor_accepted = true;
        anchor_idx
    } else if let Some(idx) =
        find_nearest_accepted_index(anchor_idx, grid.len(), &mut evaluate_idx)?
    {
        diagnostics.used_nearest_accepted_anchor = true;
        idx
    } else {
        diagnostics.used_full_grid_fallback = true;
        let range = compute_accepted_grid_range_full_grid_cached(grid, &mut cache, evaluate)?;
        return Ok((range, diagnostics));
    };
    diagnostics.accepted_anchor_idx = Some(accepted_anchor_idx);

    let lower_idx = if accepted_anchor_idx == 0 || evaluate_idx(0)? {
        0
    } else {
        let mut lower_lo = 0usize;
        let mut lower_hi = accepted_anchor_idx;
        while lower_lo < lower_hi {
            let mid = usize::midpoint(lower_lo, lower_hi);
            if evaluate_idx(mid)? {
                lower_hi = mid;
            } else {
                lower_lo = mid + 1;
            }
        }
        lower_lo
    };

    let last_idx = grid.len() - 1;
    let upper_idx = if accepted_anchor_idx == last_idx || evaluate_idx(last_idx)? {
        last_idx
    } else {
        let mut upper_lo = accepted_anchor_idx;
        let mut upper_hi = last_idx;
        while upper_lo < upper_hi {
            let mid = (upper_lo + upper_hi).div_ceil(2);
            if evaluate_idx(mid)? {
                upper_lo = mid;
            } else {
                upper_hi = mid - 1;
            }
        }
        upper_lo
    };

    if !diagnostics.anchor_accepted {
        let opposite_side_has_acceptance = match accepted_anchor_idx.cmp(&anchor_idx) {
            std::cmp::Ordering::Greater => {
                let mut found = false;
                for idx in (0..accepted_anchor_idx).rev() {
                    if evaluate_idx(idx)? {
                        found = true;
                        break;
                    }
                }
                found
            }
            std::cmp::Ordering::Less => {
                let mut found = false;
                for idx in accepted_anchor_idx + 1..grid.len() {
                    if evaluate_idx(idx)? {
                        found = true;
                        break;
                    }
                }
                found
            }
            std::cmp::Ordering::Equal => false,
        };
        if opposite_side_has_acceptance {
            diagnostics.used_full_grid_fallback = true;
            let range = compute_accepted_grid_range_full_grid_cached(grid, &mut cache, evaluate)?;
            return Ok((range, diagnostics));
        }
    }

    if lower_idx == 0 && upper_idx == last_idx {
        return Ok((
            Some(AcceptedGridRange {
                lower_idx,
                upper_idx,
            }),
            diagnostics,
        ));
    }

    let lower_boundary_ok = lower_idx == 0 || !evaluate_idx(lower_idx - 1)?;
    let upper_boundary_ok = upper_idx + 1 == grid.len() || !evaluate_idx(upper_idx + 1)?;
    let left_mid_ok = lower_idx == accepted_anchor_idx
        || evaluate_idx(usize::midpoint(lower_idx, accepted_anchor_idx))?;
    let right_mid_ok = upper_idx == accepted_anchor_idx
        || evaluate_idx((accepted_anchor_idx + upper_idx).div_ceil(2))?;
    if !(lower_boundary_ok && upper_boundary_ok && left_mid_ok && right_mid_ok) {
        diagnostics.used_full_grid_fallback = true;
        let range = compute_accepted_grid_range_full_grid_cached(grid, &mut cache, evaluate)?;
        return Ok((range, diagnostics));
    }

    Ok((
        Some(AcceptedGridRange {
            lower_idx,
            upper_idx,
        }),
        diagnostics,
    ))
}

fn find_nearest_accepted_index<F>(
    anchor_idx: usize,
    grid_len: usize,
    evaluate_idx: &mut F,
) -> Result<Option<usize>, String>
where
    F: FnMut(usize) -> Result<bool, String>,
{
    for radius in 1..grid_len {
        if let Some(left_idx) = anchor_idx.checked_sub(radius)
            && evaluate_idx(left_idx)?
        {
            return Ok(Some(left_idx));
        }
        let right_idx = anchor_idx + radius;
        if right_idx < grid_len && evaluate_idx(right_idx)? {
            return Ok(Some(right_idx));
        }
    }
    Ok(None)
}

#[cfg(test)]
pub fn compute_accepted_grid_range_full_grid<F>(
    grid: &[f64],
    evaluate: F,
) -> Result<Option<AcceptedGridRange>, String>
where
    F: FnMut(f64) -> Result<bool, String>,
{
    let mut cache = vec![None; grid.len()];
    compute_accepted_grid_range_full_grid_cached(grid, &mut cache, evaluate)
}

fn compute_accepted_grid_range_full_grid_cached<F>(
    grid: &[f64],
    cache: &mut [Option<bool>],
    mut evaluate: F,
) -> Result<Option<AcceptedGridRange>, String>
where
    F: FnMut(f64) -> Result<bool, String>,
{
    let mut flags = vec![false; grid.len()];
    for (idx, theta) in grid.iter().enumerate() {
        let accepted = if let Some(value) = cache.get(idx).copied().flatten() {
            value
        } else {
            let value = evaluate(*theta)?;
            if let Some(slot) = cache.get_mut(idx) {
                *slot = Some(value);
            }
            value
        };
        flags[idx] = accepted;
    }
    Ok(accepted_grid_range_from_flags(&flags))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adaptive_search_matches_contiguous_interval() {
        let grid = (0..64).map(f64::from).collect::<Vec<_>>();
        let accepted = compute_accepted_grid_range_adaptive(&grid, 22, |theta| {
            let idx = grid
                .iter()
                .position(|value| value.to_bits() == theta.to_bits())
                .expect("theta on grid");
            Ok((19..=27).contains(&idx))
        })
        .expect("adaptive search succeeds")
        .expect("interval exists");
        assert_eq!(accepted.lower_idx, 19);
        assert_eq!(accepted.upper_idx, 27);
    }

    #[test]
    fn adaptive_fallback_reuses_cached_evaluations() {
        let grid = (0..32).map(f64::from).collect::<Vec<_>>();
        let mut eval_count = 0usize;
        let accepted = compute_accepted_grid_range_adaptive(&grid, 16, |theta| {
            eval_count += 1;
            let idx = grid
                .iter()
                .position(|value| value.to_bits() == theta.to_bits())
                .expect("theta on grid");
            Ok(matches!(idx, 3..=5 | 20..=24))
        })
        .expect("adaptive search succeeds")
        .expect("interval exists");
        assert_eq!(accepted.lower_idx, 3);
        assert_eq!(accepted.upper_idx, 24);
        assert_eq!(eval_count, grid.len());
    }

    #[test]
    fn adaptive_search_handles_rejected_anchor_without_full_scan() {
        let grid = (0..128).map(f64::from).collect::<Vec<_>>();
        let mut eval_count = 0usize;
        let accepted = compute_accepted_grid_range_adaptive(&grid, 64, |theta| {
            eval_count += 1;
            let idx = grid
                .iter()
                .position(|value| value.to_bits() == theta.to_bits())
                .expect("theta on grid");
            Ok((70..=82).contains(&idx))
        })
        .expect("adaptive search succeeds")
        .expect("interval exists");
        assert_eq!(accepted.lower_idx, 70);
        assert_eq!(accepted.upper_idx, 82);
        assert!(eval_count < grid.len());
    }

    #[test]
    fn adaptive_search_reports_diagnostics() {
        let grid = (0..32).map(f64::from).collect::<Vec<_>>();
        let (accepted, diagnostics) =
            compute_accepted_grid_range_adaptive_with_diagnostics(&grid, 16, |theta| {
                let idx = grid
                    .iter()
                    .position(|value| value.to_bits() == theta.to_bits())
                    .expect("theta on grid");
                Ok((20..=24).contains(&idx))
            })
            .expect("adaptive search succeeds");
        let accepted = accepted.expect("interval exists");
        assert_eq!(accepted.lower_idx, 20);
        assert_eq!(accepted.upper_idx, 24);
        assert!(!diagnostics.anchor_accepted);
        assert!(diagnostics.used_nearest_accepted_anchor);
        assert!(!diagnostics.used_full_grid_fallback);
        assert!(diagnostics.unique_evaluations < grid.len());
    }

    #[test]
    fn adaptive_search_short_circuits_fully_accepted_grid() {
        let grid = (0..128).map(f64::from).collect::<Vec<_>>();
        let mut eval_count = 0usize;
        let accepted = compute_accepted_grid_range_adaptive(&grid, 64, |_theta| {
            eval_count += 1;
            Ok(true)
        })
        .expect("adaptive search succeeds")
        .expect("interval exists");
        assert_eq!(accepted.lower_idx, 0);
        assert_eq!(accepted.upper_idx, grid.len() - 1);
        assert!(eval_count <= 3);
    }
}
