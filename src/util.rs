#[must_use]
/// Convert a count to `f64` through an explicit `u32` bound.
///
/// # Panics
///
/// Panics if `value` exceeds `u32::MAX`.
pub fn usize_to_f64(value: usize) -> f64 {
    u32::try_from(value).map_or_else(|error| panic!("count exceeds u32::MAX: {error}"), f64::from)
}
