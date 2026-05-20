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

## Results container

```julia
res = collect_results(result)
long = tidy(res)
```

## Export

Use `to_json`, `to_csv`, `to_arrow`, or `to_parquet` to persist results.
