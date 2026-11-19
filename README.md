# did_methods

Difference-in-differences estimators in Rust.

## Estimators

- Callaway and Sant'Anna group-time effects, ATT(g,t), by outcome regression,
  inverse probability weighting or doubly robust estimation, for panel data
  and repeated cross-sections. Effects aggregate by event time, cohort,
  calendar time or overall.
- Doubly robust two-period DiD (DRDID), including the locally efficient
  estimators for panels and repeated cross-sections.
- Efficient ATT(g,t) that combines all baseline periods (Chen, Sant'Anna and
  Xie).
- DiD with compositional changes in repeated cross-sections (Sant'Anna and
  Xu).
- Doubly robust triple differences.
- Continuous treatments, by a sieve estimator of the average causal response.
- HonestDiD sensitivity analysis (Rambachan and Roth), behind the `honest`
  feature.

Inference uses influence functions: cluster-robust standard errors and
multiplier-bootstrap simultaneous confidence bands.

## Checked against R

The parity tests compare estimates and standard errors with the R packages
`did` 2.5.1, `DRDID` 1.3.0, `HonestDiD`, `triplediff` and `contdid`. The
reference values come from the R scripts in `tests/`. Run them with

```sh
cargo test --features parity-tests
```

## Use

The crate is on crates.io. Its shared numerical code comes from the
`epistat-*` crates, also on crates.io.

```toml
[dependencies]
did_methods = "0.2"
```

`examples/att_gt_basic.rs` estimates ATT(g,t) on a small panel and
aggregates it by event time:

```sh
cargo run --example att_gt_basic
```

## Licence

MIT or Apache-2.0, at your option.
