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
