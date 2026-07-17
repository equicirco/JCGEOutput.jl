# Usage

`JCGEOutput` renders equations and exports results in backend-agnostic form.

## Render equations

```julia
using JCGEOutput
text = render_equations(result; format=:markdown)
```

Equation payloads can use the `JCGECore` expression tree directly. `JCGEOutput`
renders equality equations (`EEq`), inequality equations (`ELe`, `EGe`), and
natural logarithms (`ELog`) to plain text, Markdown/MathJax, or LaTeX.

To disclose which equations are solver-enforced conditions and which are
post-solution accounting checks, request the optional role labels:

```julia
text = render_equations(result; format=:markdown, show_condition_roles=true)
```

## Results container

```julia
res = collect_results(result)
long = tidy(res)
```

Accounting-check residuals are included in `Results.accounting_checks` and in
tidy exports with `kind = :accounting_check`. They are also exported as
zero-dual constraints in DualSignals datasets, with the absolute residual as
their slack.

## Export

Use `to_json`, `to_csv`, `to_arrow`, or `to_parquet` to persist results.

## Satellite quantities and balance checks

Satellite reporting links a solved model-volume variable to a quantity outside
the monetary CGE core, such as mass, energy, emissions, or a physical product
flow.  It does not add equations or constraints to the model.

```julia
anchors = [
    SatelliteAnchor(:recycled_metal, "tonnes", 500.0, :Z_REC, 400.0),
]
baseline_reference = satellite_reference(baseline_results, anchors)
projection = satellite_projection(scenario_results, anchors;
    reference = baseline_reference)
```

The projected quantity is the calibrated physical quantity multiplied by the
solved driver relative to its solved baseline level.  Thus the base solution
reproduces the observed quantity exactly and every scenario uses the same
denominator. `satellite_calibration_report` retains any difference between a
rounded monetary calibration driver and the solved baseline driver. Units are
declared with each anchor. When the necessary anchors are available, a signed
balance can be checked after solution:

```julia
balances = [
    SatelliteBalance(:metal_balance, "tonnes", [
        :primary_metal => 1.0,
        :recycled_metal => 1.0,
        :metal_use => -1.0,
    ]),
]
checks = satellite_balances(projection, balances)
```

All terms in a balance must have the declared unit.  Missing drivers or
anchors raise an error by default; reporting mode can instead retain them as
explicit missing values with `strict=false`.
