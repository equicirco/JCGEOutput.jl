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

## Equation reports

For a manuscript-ready inventory, use the model-derived equation report:

```julia
latex = render_equation_report(result; format=:latex, view=:family)
```

`view=:family` groups only registered equations with an identical AST, block,
tag, closure role, and objective sense. It reports the number of registered
instances without replacing model equations with hand-written templates.
`view=:expanded` lists every registered instance instead. Objective functions
are separated from equality and inequality equations. LaTeX reports use
`align*`, so the receiving document must load `amsmath`. Solver annotations
(start values and bounds) are excluded by default because they configure the
numerical solve rather than define the mathematical model. Set
`include_solver_annotations=true` to audit the full registry.

For indexed reports whose concrete registered identifiers encode several model
dimensions, the consumer can declare those dimensions explicitly. This avoids
parsing identifier names and validates that every selected equation is covered:

```julia
mapping = EquationReportMapping(
    source_block=:eol_source,
    source_tag=:choice,
    index_names=(:region, :product, :route),
    coordinates=Dict(
        (:EOL_DE_ELMA_REC,) => (:DE, :ELMA, :REC),
        # one entry for every selected registered equation
    ),
    domain_values=Dict(
        :FAC_DE_LAB => :LAB,
        :FAC_DE_CAP => :CAP,
    ),
    index_projections=Dict(
        :activity => (:region, :product),
    ),
    reference_indices=Dict(
        (:variable, :pz, (:IND_DE_NEW_ELMA,)) => ((:region, :product),),
    ),
)
latex = render_equation_report(result; format=:latex, view=:indexed,
    report_mappings=[mapping])
```

Each key is an exact registered `payload.indices` tuple. The mapping is a
reporting declaration only: it does not alter the model equations.
`domain_values` applies only to the concrete domains of generated sums and
products, so a compact multi-region formula can show, for example, `{LAB, CAP}`
instead of a representative region's factor identifiers. Strict validation also
rejects declared domain values that are not used by the selected equations.
`index_projections` maps a source `EIndex` used by an expression to one or more
of the declared report indices; for example, an `:activity` index can be
rendered as `(region, product)`. It is likewise explicitly declared and checked
for use.
`reference_indices` applies to a concrete variable or parameter reference inside
an equation, including explicit additive terms; its value provides one report
index tuple for each original index position.

When a model registers an identity as an explicit enumeration, the consumer can
render it as a compact sum only by declaring that enumeration explicitly. For
example, a regional market identity can use:

```julia
additive_sums = [
    AdditiveSumMapping(path=(:lhs,), index=:region,
        domain=[:DE, :FR], term_name=:EU_SALE, index_position=2),
]
```

The selected expression must be a direct addition of the declared reference;
the domain and the non-summation indices are checked for every registered
instance. This changes only the report, not the model equation.

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
