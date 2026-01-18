<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/src/assets/jcge_output_logo_dark.png">
  <source media="(prefers-color-scheme: light)" srcset="docs/src/assets/jcge_output_logo_light.png">
  <img alt="JCGE Output logo" src="docs/src/assets/jcge_output_logo_light.png" height="150">
</picture>

# JCGEOutput

## What is a CGE?
A Computable General Equilibrium (CGE) model is a quantitative economic model that represents an economy as interconnected markets for goods and services, factors of production, institutions, and the rest of the world. It is calibrated with data (typically a Social Accounting Matrix) and solved numerically as a system of nonlinear equations until equilibrium conditions (zero-profit, market-clearing, and income-balance) hold within tolerance.

## What is JCGE?
JCGE is a block-based CGE modeling and execution framework in Julia ([jcge.org](https://jcge.org)). It defines a shared RunSpec structure and reusable blocks so models can be assembled, validated, solved, and compared consistently across packages.

## What is this package?
Backend-agnostic output and reporting utilities for JCGE ([jcge.org](https://jcge.org)).

## How to cite

If you use the JCGE framework, please cite:

Boero, R. *JCGE - Julia Computable General Equilibrium Framework* [software], 2026.
DOI: 10.5281/zenodo.18282436
URL: https://JCGE.org

```bibtex
@software{boero_jcge_2026,
  title  = {JCGE - Julia Computable General Equilibrium Framework},
  author = {Boero, Riccardo},
  year   = {2026},
  doi    = {10.5281/zenodo.18282436},
  url    = {https://JCGE.org}
}
```

If you use this package, please cite:

Boero, R. *JCGEOutput.jl - Equation rendering and results handling for JCGE.* [software], 2026.
DOI: 10.5281/zenodo.18290750
URL: https://Output.JCGE.org
SourceCode: https://github.com/equicirco/JCGEOutput.jl

```bibtex
@software{boero_jcgeoutput_jl_2026,
  title  = {JCGEOutput.jl - Equation rendering and results handling for JCGE.},
  author = {Boero, Riccardo},
  year   = {2026},
  doi    = {10.5281/zenodo.18290750},
  url    = {https://Output.JCGE.org}
}
```

If you use a specific tagged release, please cite the version DOI assigned on Zenodo for that release (preferred for exact reproducibility).

## 1) Equation and block rendering (model introspection)

Render the generated system (or each block) to:
- Markdown with MathJax/LaTeX fragments
- LaTeX document snippets
- plain text (debug)

Recommended API surface:

```
render_equations(model; format=:markdown, level=:block|:equation, show_defs=true)
render_block(model, block_id; format=:markdown)
render_symbols(model; format=:markdown, show_values=true) # set false to omit numeric values
render_blocks(spec; format=:markdown)
render_sections(sections; format=:markdown)
```

Include symbol tables: variables, parameters, indices/sets, domain restrictions.

Key design point: render from a stable internal equation AST (not solver-specific
objects), so it works regardless of backend.

## 2) Results container + persistence

Canonical results object (backend-agnostic):
- primals (levels), duals (multipliers), reduced costs, complement status where relevant
- metadata: scenario id, calibration hash, solver info, timestamps, units/currency,
  numeraire, closure flags

Exports:
- `collect_results`, `tidy`, `to_json`, `to_csv`
- “tidy” long-table form: `(symbol, index_tuple, value, kind=:level|:dual|:reduced_cost|:complement)`

Integration: `DualSignals.jl` adapters should live here to reuse its
saving/analysis utilities.

Example:

```julia
using JCGEOutput
using JCGERuntime

result = run!(spec; optimizer=Ipopt.Optimizer)
results = collect_results(result; metadata=Dict(:scenario_id => "base"))

rows = tidy(results)
to_json(results, "results.json")
to_csv(results, "results.csv")
to_arrow(results, "results.arrow")
to_parquet(results, "results.parquet")

dataset = to_dualsignals(results; dataset_id="jcge", description="baseline")
write_dualsignals_json(results, "dualsignals.json"; dataset_id="jcge")
write_dualsignals_csv(results, "dualsignals_csv"; dataset_id="jcge", prefix="jcge")
```

Index handling:
- `to_csv` encodes index tuples as `a|b|c` strings for round-tripping.
- `to_arrow`/`to_parquet` use the same encoding.
- `results_from_csv` decodes that field back to tuples.
- `results_from_arrow`/`results_from_parquet` decode the same.
- `to_json` keeps tuple structure; `results_from_json` restores it.

Adapters:
- `to_dualsignals` accepts `Results`, `KernelContext`, or a `run!` result.
- `write_dualsignals_json`/`write_dualsignals_csv` accept the same.

Mapping options (production-level defaults included):
- `sections` or `block_sections` to classify blocks into canonical sections.
- `component_type_by_block` / `component_type_by_section` to override component types.
- `constraint_kind_by_tag` / `constraint_kind_by_block` / `constraint_kind_by_section` to override constraint kinds.
- `variable_component_type` to tag the variable component group.
- `DEFAULT_CONSTRAINT_KIND_TAG_MAP` is exported for reuse or extension.

Example override:

```julia
using JCGEOutput

custom = copy(DEFAULT_CONSTRAINT_KIND_TAG_MAP)
custom[:eqXv] = constraint_kind_enum(:balance)

dataset = to_dualsignals(results; constraint_kind_by_tag=custom)
```

## 3) SAM/IO-style reporting (dump solution back to a SAM)

```julia
using JCGEOutput
using JCGECalibrate

sam_table = JCGECalibrate.load_sam_table("sam.csv")
start = JCGECalibrate.compute_starting_values(sam_table)

out = sam_from_solution(result;
    spec=spec,
    sam_table=sam_table,
    start=start,
    valuation=:baseline,        # :model or :baseline
    include_quantities=true,
    include_taxes=true)

write_sam_csv(out.values, "sam_values.csv")
write_sam_csv(out.quantities, "sam_quantities.csv")
```

Valuation modes:
- `:model` uses solver prices (numeraire-based).
- `:baseline` uses calibration prices from `StartingValues` to re-express results
  in baseline SAM units.

Notes:
- Only model-backed entries are populated; missing flows remain zero/NaN.
- Taxes and savings/investment are included only if the corresponding variables
  exist in the solution.
- This reporting is opt-in; no SAM output is produced unless you call it.

Requirements and behavior:
- `sam_table` provides the row/column labels and baseline SAM structure.
- `valuation=:baseline` requires `start::JCGECalibrate.StartingValues`; values are
  re-expressed in base SAM units using calibrated prices.
- `valuation=:model` uses solver prices in numeraire units (relative values).
- Output includes a SAM of values and, when `include_quantities=true`, a SAM of
  quantities (where defined by the model).

Limitations:
- The SAM is partial: only flows backed by model variables are populated.
- Missing variables mean missing SAM entries; these remain zero/NaN.
- Depending on closure, the resulting SAM may be imbalanced; this is reported
  implicitly by row/column sums and is not auto-corrected.

Status: rendering API available (equations/blocks and symbol tables from KernelContext or run result).
Blocks should attach `payload.expr::EquationExpr` (from `JCGECore`); `ERaw(info)` is an acceptable fallback while blocks are upgraded.

## AST example (equation rendering)

```julia
using JCGERuntime
using JCGEOutput

ctx = KernelContext()
register_equation!(ctx; tag=:demo, block=:example,
    payload=(indices=(), expr=EAdd([EVar(:x), EConst(1)]), constraint=nothing))

println(render_equations(ctx; format=:plain, level=:equation))
```
