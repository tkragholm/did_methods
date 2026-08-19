#[must_use]
/// Convert a count to `f64` through an explicit `u32` bound.
///
/// # Panics
///
/// Panics if `value` exceeds `u32::MAX`.
pub fn usize_to_f64(value: usize) -> f64 {
    u32::try_from(value).map_or_else(|error| panic!("count exceeds u32::MAX: {error}"), f64::from)
}

#[must_use]
/// Convert a positional index to `i64` through an explicit `u32` bound.
///
/// The same bound as [`usize_to_f64`], for the same reason: a count that will
/// not fit in a `u32` is a bug upstream rather than a number to truncate.
///
/// # Panics
///
/// Panics if `value` exceeds `u32::MAX`.
pub fn usize_to_i64(value: usize) -> i64 {
    u32::try_from(value).map_or_else(|error| panic!("count exceeds u32::MAX: {error}"), i64::from)
}
