# Usage

`JCGEOutput` renders equations and exports results in backend-agnostic form.

## Render equations

```julia
using JCGEOutput
text = render_equations(result; format=:markdown)
```

## Results container

```julia
res = collect_results(result)
long = tidy(res)
```

## Export

Use `to_json`, `to_csv`, `to_arrow`, or `to_parquet` to persist results.

