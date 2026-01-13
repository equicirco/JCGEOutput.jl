"""
Output utilities: equation rendering, results collection, persistence, and reporting.
"""
module JCGEOutput

import JCGECore
using JCGECalibrate
using Arrow
using CSV
using JSON3
using JCGECore: EquationExpr, EIndex, EVar, EParam, EConst, EAdd, EMul, EPow, EDiv, ENeg, ESum, EProd, EEq, ERaw
using JCGERuntime
using Dates
using JuMP
using Parquet
using Tables
using DualSignals

export render_equations, render_block
export render_symbols, render_blocks, render_sections
export Results, collect_results, tidy, to_json, to_csv
export results_from_json, results_from_csv, results_from_arrow, results_from_parquet
export to_arrow, to_parquet
export to_dualsignals, write_dualsignals_json, write_dualsignals_csv
export sam_from_solution, write_sam_csv
export DEFAULT_CONSTRAINT_KIND_TAG_MAP, constraint_kind_enum, component_type_enum
export EquationExpr, EIndex, EVar, EParam, EConst, EAdd, EMul, EPow, EDiv, ENeg, ESum, EProd, EEq, ERaw
export render_expr

"""
Container for solver results and metadata.

Fields:
- `primals`: variable levels by symbol.
- `reduced_costs`: reduced costs by symbol.
- `duals`: vector of (block, tag, indices, value).
- `complements`: complementarity diagnostics for MCP equations.
- `metadata`: solver and scenario metadata.
"""
struct Results
    primals::Dict{Symbol,Float64}
    reduced_costs::Dict{Symbol,Float64}
    duals::Vector{NamedTuple}
    complements::Vector{NamedTuple}
    metadata::Dict{Symbol,Any}
end

"""
    collect_results(obj; metadata=Dict()) -> Results

Collect primals, reduced costs, duals, and complementarity diagnostics
from a `KernelContext`, `RunSpec` result, or equivalent object.
"""
function collect_results(obj; metadata=Dict{Symbol,Any}())
    ctx = _context(obj)
    primals = Dict{Symbol,Float64}()
    reduced_costs = Dict{Symbol,Float64}()
    duals = NamedTuple[]
    complements = NamedTuple[]

    model = ctx.model
    if model isa JuMP.Model
        for (name, var) in ctx.variables
            var isa JuMP.VariableRef || continue
            val = try
                JuMP.value(var)
            catch
                nothing
            end
            if val !== nothing && isfinite(val)
                primals[name] = val
            end
            rc = try
                JuMP.reduced_cost(var)
            catch
                nothing
            end
            if rc !== nothing && isfinite(rc)
                reduced_costs[name] = rc
            end
        end

        for eq in JCGERuntime.list_equations(ctx)
            payload = eq.payload
            payload isa NamedTuple || continue
            constraint = get(payload, :constraint, nothing)
            if constraint isa JuMP.ConstraintRef
                dual = try
                    JuMP.dual(constraint)
                catch
                    nothing
                end
                if dual !== nothing && isfinite(dual)
                    push!(duals, (block=eq.block, tag=eq.tag, indices=get(payload, :indices, ()), value=dual))
                end
            end
            if haskey(payload, :mcp_var)
                mcp_var = payload.mcp_var
                var_name = _resolve_var_name(mcp_var, payload)
                var_value = nothing
                if var_name !== nothing && haskey(ctx.variables, var_name)
                    var = ctx.variables[var_name]
                    if var isa JuMP.VariableRef
                        var_value = try
                            JuMP.value(var)
                        catch
                            nothing
                        end
                    end
                end
                residual = nothing
                if constraint isa JuMP.ConstraintRef
                    residual = try
                        JuMP.value(constraint)
                    catch
                        nothing
                    end
                end
                push!(complements, (
                    block=eq.block,
                    tag=eq.tag,
                    indices=get(payload, :indices, ()),
                    var=var_name,
                    value=var_value,
                    residual=residual,
                ))
            end
        end
    end

    meta = Dict{Symbol,Any}()
    meta[:timestamp] = Dates.now()
    if model isa JuMP.Model
        meta[:solver_name] = try JuMP.solver_name(model) catch nothing end
        meta[:termination_status] = try JuMP.termination_status(model) catch nothing end
        meta[:primal_status] = try JuMP.primal_status(model) catch nothing end
        meta[:dual_status] = try JuMP.dual_status(model) catch nothing end
        meta[:objective_value] = try JuMP.objective_value(model) catch nothing end
    end
    for (k, v) in metadata
        meta[k] = v
    end

    return Results(primals, reduced_costs, duals, complements, meta)
end

"""
    tidy(results; kinds=..., encode_indices=false) -> Vector{NamedTuple}

Return a long-table representation of results suitable for CSV/Arrow/Parquet.
"""
function tidy(results::Results; kinds=(:level, :dual, :reduced_cost, :complement), encode_indices::Bool=false)
    rows = NamedTuple[]
    if :level in kinds
        for (name, val) in results.primals
            push!(rows, (symbol=string(name), index_tuple=_index_field((), encode_indices), value=val, kind=:level))
        end
    end
    if :reduced_cost in kinds
        for (name, val) in results.reduced_costs
            push!(rows, (symbol=string(name), index_tuple=_index_field((), encode_indices), value=val, kind=:reduced_cost))
        end
    end
    if :dual in kinds
        for entry in results.duals
            sym = Symbol(string(entry.block), ".", string(entry.tag))
            push!(rows, (symbol=string(sym), index_tuple=_index_field(Tuple(entry.indices), encode_indices), value=entry.value, kind=:dual))
        end
    end
    if :complement in kinds
        for entry in results.complements
            sym = entry.var === nothing ? Symbol(string(entry.block), ".", string(entry.tag)) : entry.var
            value = entry.value === nothing ? NaN : entry.value
            push!(rows, (symbol=string(sym), index_tuple=_index_field(Tuple(entry.indices), encode_indices), value=value, kind=:complement))
        end
    end
    return rows
end

"""
    to_json(results, path)

Write results and a tidy table to a JSON file.
"""
function to_json(results::Results, path::AbstractString)
    payload = (
        metadata=_stringify_keys(results.metadata),
        primals=_stringify_keys(results.primals),
        reduced_costs=_stringify_keys(results.reduced_costs),
        duals=results.duals,
        complements=results.complements,
        tidy=tidy(results),
    )
    open(path, "w") do io
        JSON3.write(io, payload)
    end
    return path
end

"""
    to_csv(results, path; kinds=...)

Write a tidy table to CSV.
"""
function to_csv(results::Results, path::AbstractString; kinds=(:level, :dual, :reduced_cost, :complement))
    rows = tidy(results; kinds=kinds, encode_indices=true)
    if isempty(rows)
        open(path, "w") do io
            write(io, "symbol,index_tuple,value,kind\n")
        end
    else
        CSV.write(path, rows)
    end
    return path
end

"""
    to_arrow(results, path; kinds=...)

Write a tidy table to Arrow.
"""
function to_arrow(results::Results, path::AbstractString; kinds=(:level, :dual, :reduced_cost, :complement))
    rows = tidy(results; kinds=kinds, encode_indices=true)
    Arrow.write(path, rows)
    return path
end

"""
    to_parquet(results, path; kinds=...)

Write a tidy table to Parquet.
"""
function to_parquet(results::Results, path::AbstractString; kinds=(:level, :dual, :reduced_cost, :complement))
    rows = tidy(results; kinds=kinds, encode_indices=true)
    parquet_rows = [
        (
            symbol=string(row.symbol),
            index_tuple=string(row.index_tuple),
            value=row.value,
            kind=string(row.kind),
        ) for row in rows
    ]
    Parquet.write_parquet(path, parquet_rows)
    return path
end

"""
    results_from_json(path) -> Results

Load a Results object from JSON.
"""
function results_from_json(path::AbstractString)
    data = JSON3.read(read(path, String))
    metadata = Dict{Symbol,Any}()
    if haskey(data, "metadata")
        for (k, v) in pairs(data["metadata"])
            metadata[Symbol(k)] = v
        end
    end
    primals = _symbol_dict(get(data, "primals", Dict{String,Any}()))
    reduced = _symbol_dict(get(data, "reduced_costs", Dict{String,Any}()))
    duals = _namedtuple_list(get(data, "duals", Any[]))
    complements = _namedtuple_list(get(data, "complements", Any[]))
    return Results(primals, reduced, duals, complements, metadata)
end

"""
    results_from_csv(path) -> Results

Load a Results object from a tidy CSV file.
"""
function results_from_csv(path::AbstractString)
    return results_from_table(CSV.File(path))
end

"""
    results_from_arrow(path) -> Results

Load a Results object from Arrow.
"""
function results_from_arrow(path::AbstractString)
    return results_from_table(Arrow.Table(path))
end

"""
    results_from_parquet(path) -> Results

Load a Results object from Parquet.
"""
function results_from_parquet(path::AbstractString)
    return results_from_table(Parquet.read_parquet(path))
end

"""
    to_dualsignals(results; kwargs...) -> DualSignalsDataset

Map results into DualSignals components and constraints for analysis.
"""
function to_dualsignals(results::Results; dataset_id::String="jcge",
    description::Union{String,Nothing}=nothing,
    scenario::Union{String,Nothing}=nothing,
    include_variables::Bool=true,
    var_component_fn::Function=name -> "variables",
    variable_component_type::Symbol=:other,
    sections::Union{Nothing,Vector{JCGECore.SectionSpec}}=nothing,
    block_sections::Dict{Symbol,Symbol}=Dict{Symbol,Symbol}(),
    component_type_by_block::Dict{Symbol,Symbol}=Dict{Symbol,Symbol}(),
    component_type_by_section::Dict{Symbol,Symbol}=Dict{Symbol,Symbol}(),
    constraint_kind_by_tag::Dict{Symbol,Symbol}=Dict{Symbol,Symbol}(),
    constraint_kind_by_block::Dict{Symbol,Symbol}=Dict{Symbol,Symbol}(),
    constraint_kind_by_section::Dict{Symbol,Symbol}=Dict{Symbol,Symbol}())
    components = Dict{String,DualSignals.Component}()
    constraints = DualSignals.Constraint[]
    solutions = DualSignals.ConstraintSolution[]
    variables = include_variables ? DualSignals.VariableValue[] : nothing

    section_map = _resolve_sections(sections, block_sections)

    for entry in results.duals
        block = entry.block
        section = get(section_map, block, nothing)
        component_id = string(block)
        component_type = _component_type(block; section=section, overrides=component_type_by_block, section_overrides=component_type_by_section)
        _ensure_component!(components, component_id, component_type)
        constraint_id = _constraint_id(block, entry.tag, entry.indices)
        kind = _constraint_kind(block, entry.tag; section=section, tag_overrides=constraint_kind_by_tag,
            block_overrides=constraint_kind_by_block, section_overrides=constraint_kind_by_section)
        push!(constraints, DualSignals.Constraint(
            constraint_id=constraint_id,
            kind=kind,
            sense=_constraint_sense_enum(:eq),
            component_ids=[component_id],
        ))
        push!(solutions, DualSignals.ConstraintSolution(
            constraint_id=constraint_id,
            dual=entry.value,
            slack=nothing,
            is_binding=nothing,
            scenario=scenario,
        ))
    end

    for entry in results.complements
        block = get(entry, :block, :mcp)
        tag = get(entry, :tag, :mcp)
        section = get(section_map, block, nothing)
        component_id = string(block)
        component_type = _component_type(block; section=section, overrides=component_type_by_block, section_overrides=component_type_by_section)
        _ensure_component!(components, component_id, component_type)
        constraint_id = _constraint_id(block, tag, get(entry, :indices, ()))
        kind = _constraint_kind(block, tag; section=section, tag_overrides=constraint_kind_by_tag,
            block_overrides=constraint_kind_by_block, section_overrides=constraint_kind_by_section)
        push!(constraints, DualSignals.Constraint(
            constraint_id=constraint_id,
            kind=kind,
            sense=_constraint_sense_enum(:eq),
            component_ids=[component_id],
        ))
        slack = entry.residual === nothing ? nothing : abs(entry.residual)
        dual = entry.value === nothing ? 0.0 : entry.value
        push!(solutions, DualSignals.ConstraintSolution(
            constraint_id=constraint_id,
            dual=dual,
            slack=slack,
            is_binding=slack === nothing ? nothing : slack <= 1e-8,
            scenario=scenario,
        ))
    end

    if include_variables
        for (name, value) in results.primals
            component_id = string(var_component_fn(name))
        component_type = _component_type_from_symbol(variable_component_type)
            _ensure_component!(components, component_id, component_type)
            push!(variables, DualSignals.VariableValue(
                component_id=component_id,
                name=string(name),
                value=value,
                scenario=scenario,
            ))
        end
    end

    metadata = DualSignals.DatasetMetadata(
        description=description,
        created_at=get(results.metadata, :timestamp, nothing),
        objective_sense=_objective_sense(get(results.metadata, :objective_sense, nothing)),
        objective_value=_float_or_nothing(get(results.metadata, :objective_value, nothing)),
        units_convention=_string_or_nothing(get(results.metadata, :units_convention, nothing)),
        notes=_notes_from_metadata(results.metadata, description),
    )
    return DualSignals.DualSignalsDataset(
        dataset_id=dataset_id,
        metadata=metadata,
        components=collect(values(components)),
        constraints=constraints,
        constraint_solutions=solutions,
        variables=variables,
    )
end

const BASELINE_PRICE_FIELDS = Dict(
    :pf => :pf0,
    :py => :py0,
    :pz => :pz0,
    :pq => :pq0,
    :pe => :pe0,
    :pm => :pm0,
    :pd => :pd0,
    :pWe => :pWe,
    :pWm => :pWm,
)

"""
    sam_from_solution(obj; kwargs...) -> LabeledMatrix

Build a SAM-style table from a solved model.
Only entries backed by model flows are populated.
"""
function sam_from_solution(obj;
    spec::JCGECore.RunSpec,
    sam_table::JCGECalibrate.SAMTable,
    start::Union{Nothing,JCGECalibrate.StartingValues}=nothing,
    valuation::Symbol=:model,
    include_quantities::Bool=false,
    include_taxes::Bool=true,
    include_savings::Bool=true,
    include_trade::Bool=true,
)
    results = _results(obj)
    valuation in (:model, :baseline) || error("valuation must be :model or :baseline")
    if valuation == :baseline && start === nothing
        error("baseline valuation requires `start` (JCGECalibrate.StartingValues)")
    end

    base_sam = sam_table.sam
    values = zeros(length(base_sam.row_labels), length(base_sam.col_labels))
    quantities = include_quantities ? fill(NaN, size(values)) : nothing
    value_sam = JCGECalibrate.LabeledMatrix(values, base_sam.row_labels, base_sam.col_labels)
    qty_sam = quantities === nothing ? nothing :
        JCGECalibrate.LabeledMatrix(quantities, base_sam.row_labels, base_sam.col_labels)

    function add_entry!(row::Symbol, col::Symbol, value::Float64; quantity=nothing)
        if !haskey(value_sam.row_index, row) || !haskey(value_sam.col_index, col)
            return nothing
        end
        value_sam.data[value_sam.row_index[row], value_sam.col_index[col]] += value
        if qty_sam !== nothing && quantity !== nothing
            qty_sam.data[value_sam.row_index[row], value_sam.col_index[col]] += quantity
        end
        return nothing
    end

    commodities = spec.model.sets.commodities
    activities = spec.model.sets.activities
    factors = spec.model.sets.factors

    hoh = sam_table.households_label
    gov = sam_table.government_label
    inv = sam_table.investment_label
    ext = sam_table.restOfTheWorld_label
    idt = sam_table.indirectTax_label
    trf = sam_table.tariff_label

    for j in activities, i in commodities
        q = _var_value(results, :X, i, j)
        q === nothing && continue
        p = _price_value(results, start, valuation, (:pq, :px, :p), i)
        p === nothing && continue
        add_entry!(i, j, p * q; quantity=q)
    end

    for j in activities, h in factors
        q = _var_value(results, :F, h, j)
        q === nothing && continue
        p = _price_value(results, start, valuation, (:pf,), h, j)
        p === nothing && continue
        add_entry!(h, j, p * q; quantity=q)
    end

    for i in commodities
        q = _var_value(results, :Xp, i)
        if q !== nothing
            p = _price_value(results, start, valuation, (:pq, :px, :p), i)
            p !== nothing && add_entry!(i, hoh, p * q; quantity=q)
        end
        q = _var_value(results, :Xg, i)
        if q !== nothing
            p = _price_value(results, start, valuation, (:pq, :px, :p), i)
            p !== nothing && add_entry!(i, gov, p * q; quantity=q)
        end
        q = _var_value(results, :Xv, i)
        if q !== nothing
            p = _price_value(results, start, valuation, (:pq, :px, :p), i)
            p !== nothing && add_entry!(i, inv, p * q; quantity=q)
        end
    end

    if include_trade
        for i in commodities
            q = _var_value(results, :E, i)
            if q !== nothing
                p = _price_value(results, start, valuation, (:pe,), i)
                p = p === nothing ? _trade_price(results, start, valuation, :pWe, i) : p
                p !== nothing && add_entry!(i, ext, p * q; quantity=q)
            end
            q = _var_value(results, :M, i)
            if q !== nothing
                p = _price_value(results, start, valuation, (:pm,), i)
                p = p === nothing ? _trade_price(results, start, valuation, :pWm, i) : p
                p !== nothing && add_entry!(ext, i, p * q; quantity=q)
            end
        end
    end

    if include_taxes
        for i in commodities
            val = _var_value(results, :Tz, i)
            val !== nothing && add_entry!(idt, i, val)
            val = _var_value(results, :Tm, i)
            val !== nothing && add_entry!(trf, i, val)
        end
        td = _var_value(results, :Td)
        td !== nothing && add_entry!(gov, hoh, td)
    end

    if include_savings
        sp = _var_value(results, :Sp)
        sp !== nothing && add_entry!(inv, hoh, sp)
        sg = _var_value(results, :Sg)
        sg !== nothing && add_entry!(inv, gov, sg)
        sf = _var_value(results, :Sf)
        sf !== nothing && add_entry!(inv, ext, sf)
    end

    for h in factors
        q = _var_value(results, :FF, h)
        q === nothing && continue
        p = _price_value(results, start, valuation, (:pf,), h)
        p === nothing && continue
        add_entry!(hoh, h, p * q; quantity=q)
    end

    return qty_sam === nothing ? (values=value_sam,) : (values=value_sam, quantities=qty_sam)
end

"""
    write_sam_csv(sam, path; label_col="label")

Write a labeled SAM matrix to CSV.
"""
function write_sam_csv(sam::JCGECalibrate.LabeledMatrix, path::AbstractString; label_col::String="label")
    rows = NamedTuple[]
    for (i, row) in pairs(sam.row_labels)
        entry = Dict{Symbol,Any}()
        entry[Symbol(label_col)] = String(row)
        for (j, col) in pairs(sam.col_labels)
            entry[Symbol(col)] = sam.data[i, j]
        end
        push!(rows, NamedTuple(entry))
    end
    CSV.write(path, rows)
    return path
end

function _results(obj)
    return obj isa Results ? obj : collect_results(obj)
end

function _global_var_name(base::Symbol, idxs::Symbol...)
    isempty(idxs) && return base
    return Symbol(string(base), "_", join(string.(idxs), "_"))
end

function _var_value(results::Results, base::Symbol, idxs::Symbol...)
    name = _global_var_name(base, idxs...)
    return get(results.primals, name, nothing)
end

function _price_value(results::Results, start, valuation::Symbol, bases::Tuple{Vararg{Symbol}}, idxs::Symbol...)
    for base in bases
        if valuation == :model
            val = _var_value(results, base, idxs...)
            if val === nothing && length(idxs) > 1
                val = _var_value(results, base, idxs[1])
            end
            val === nothing && continue
            return val
        else
            val = _baseline_price(start, base, idxs...)
            if val === nothing && length(idxs) > 1
                val = _baseline_price(start, base, idxs[1])
            end
            val === nothing && continue
            return val
        end
    end
    return nothing
end

function _baseline_price(start::JCGECalibrate.StartingValues, base::Symbol, idxs::Symbol...)
    field = get(BASELINE_PRICE_FIELDS, base, nothing)
    field === nothing && return nothing
    data = getfield(start, field)
    if data isa JCGECalibrate.LabeledVector
        idxs = isempty(idxs) ? () : (idxs[1],)
        isempty(idxs) && return nothing
        return data[idxs[1]]
    end
    if data isa JCGECalibrate.LabeledMatrix
        length(idxs) == 2 || return nothing
        return data[idxs[1], idxs[2]]
    end
    return data
end

function _trade_price(results::Results, start, valuation::Symbol, base::Symbol, idx::Symbol)
    price = _price_value(results, start, valuation, (base,), idx)
    price === nothing && return nothing
    epsilon = valuation == :model ? _var_value(results, :epsilon) :
        (start === nothing ? nothing : start.epsilon0)
    epsilon === nothing && return price
    return price * epsilon
end

function to_dualsignals(obj::JCGERuntime.KernelContext; kwargs...)
    results = collect_results(obj)
    return to_dualsignals(results; kwargs...)
end

function to_dualsignals(obj::NamedTuple; kwargs...)
    ctx = _context(obj)
    results = collect_results(ctx)
    return to_dualsignals(results; kwargs...)
end

"""
    write_dualsignals_json(results, path; kwargs...)

Write a DualSignals dataset to JSON.
"""
function write_dualsignals_json(results::Results, path::AbstractString; kwargs...)
    dataset = to_dualsignals(results; kwargs...)
    DualSignals.write_json(path, dataset)
    return path
end

function write_dualsignals_json(obj::JCGERuntime.KernelContext, path::AbstractString; kwargs...)
    return write_dualsignals_json(collect_results(obj), path; kwargs...)
end

function write_dualsignals_json(obj::NamedTuple, path::AbstractString; kwargs...)
    return write_dualsignals_json(collect_results(_context(obj)), path; kwargs...)
end

"""
    write_dualsignals_csv(results, dir; prefix="dualsignals", kwargs...)

Write DualSignals tables to CSV files in a directory.
"""
function write_dualsignals_csv(results::Results, dir::AbstractString; prefix::AbstractString="dualsignals", kwargs...)
    dataset = to_dualsignals(results; kwargs...)
    DualSignals.write_csv(dataset, dir; prefix=prefix)
    return dir
end

function write_dualsignals_csv(obj::JCGERuntime.KernelContext, dir::AbstractString; prefix::AbstractString="dualsignals", kwargs...)
    return write_dualsignals_csv(collect_results(obj), dir; prefix=prefix, kwargs...)
end

function write_dualsignals_csv(obj::NamedTuple, dir::AbstractString; prefix::AbstractString="dualsignals", kwargs...)
    return write_dualsignals_csv(collect_results(_context(obj)), dir; prefix=prefix, kwargs...)
end

"""
    render_equations(obj; format=:markdown, level=:block, show_defs=true)

Render equations registered in a `KernelContext` or run result.

Inputs
- `obj`: a `JCGERuntime.KernelContext`, a run result (`NamedTuple` with `context`),
  or a `JCGECore.RunSpec` (via a `KernelContext`).
- `format`: `:markdown`, `:latex`, or `:plain`.
- `level`: `:block` to group equations by block, or `:equation` for a flat list.
- `show_defs`: include equation labels and block tags.

Returns a formatted string. The output is derived from the equation AST, not
solver-specific objects, so it is backend-agnostic.
"""
function render_equations(obj; format::Symbol=:markdown, level::Symbol=:block, show_defs::Bool=true)
    ctx = _context(obj)
    eqs = JCGERuntime.list_equations(ctx)
    return _render_equations(eqs; format=format, level=level, show_defs=show_defs)
end

"""
    render_block(obj, block_id; format=:markdown, show_defs=true)

Render equations for one block.

`block_id` can be a `Symbol` or a string-like identifier; it is converted to a
`Symbol` and matched against `EquationInfo.block` entries.
"""
function render_block(obj, block_id; format::Symbol=:markdown, show_defs::Bool=true)
    ctx = _context(obj)
    eqs = JCGERuntime.list_equations(ctx)
    block_sym = Symbol(block_id)
    eqs_block = filter(eq -> eq.block == block_sym, eqs)
    return _render_equations(eqs_block; format=format, level=:equation, show_defs=show_defs)
end

"""
    render_symbols(obj; format=:markdown, show_values=true)

Render a symbol table for registered variables.

When `show_values=true`, the table includes current values from the snapshot
state in the context/run result.
"""
function render_symbols(obj; format::Symbol=:markdown, show_values::Bool=true)
    ctx = _context(obj)
    names = sort(collect(keys(ctx.variables)); by=string)
    state = JCGERuntime.snapshot_state(ctx)
    values = JCGERuntime.snapshot(ctx)

    rows = Vector{NamedTuple}(undef, 0)
    for name in names
        val = show_values ? get(values, name, nothing) : nothing
        push!(rows, (
            symbol=name,
            value=val,
            lower=get(state.lower, name, nothing),
            upper=get(state.upper, name, nothing),
            fixed=get(state.fixed, name, nothing),
        ))
    end
    return _render_symbol_table(rows; format=format, show_values=show_values)
end

"""
    render_blocks(obj; format=:markdown)

Render a block list.

Accepts either a `JCGECore.RunSpec` or a `Vector{JCGECore.SectionSpec}` and
renders block names grouped by section when section metadata is available.
"""
function render_blocks(obj; format::Symbol=:markdown)
    if obj isa JCGECore.RunSpec
        return _render_block_list(obj.model.blocks; format=format)
    elseif obj isa Vector{JCGECore.SectionSpec}
        return render_sections(obj; format=format)
    else
        error("Unsupported input for render_blocks: expected RunSpec or Vector{SectionSpec}")
    end
end

"""
    render_sections(sections; format=:markdown)

Render the section/block skeleton from a vector of `SectionSpec`.
"""
function render_sections(sections::Vector{JCGECore.SectionSpec}; format::Symbol=:markdown)
    lines = String[]
    header = "Sections"
    if format == :markdown
        push!(lines, "# $(header)")
    elseif format == :latex
        push!(lines, "% $(header)")
    else
        push!(lines, header)
    end
    for sec in sections
        sec_name = string(sec.name)
        if format == :markdown
            push!(lines, "## $(sec_name)")
        elseif format == :latex
            push!(lines, "% $(sec_name)")
        else
            push!(lines, sec_name)
        end
        for block in sec.blocks
            push!(lines, _format_bullet(format, _block_label(block)))
        end
    end
    return join(lines, "\n")
end

function _context(obj)
    if obj isa JCGERuntime.KernelContext
        return obj
    elseif obj isa NamedTuple && haskey(obj, :context)
        return obj.context
    else
        error("Unsupported input for rendering: expected KernelContext or result NamedTuple")
    end
end

function _resolve_var_name(expr, payload)
    if expr isa EVar
        indices = get(payload, :indices, ())
        index_names = get(payload, :index_names, nothing)
        env = _index_env(index_names, indices)
        resolved = _resolve_indices(expr.idxs, indices, env)
        return _global_var(expr.name, resolved...)
    elseif expr isa Symbol
        return expr
    else
        return nothing
    end
end

function _ensure_component!(components, component_id::String, component_type::DualSignals.ComponentType)
    if !haskey(components, component_id)
        components[component_id] = DualSignals.Component(
            component_id=component_id,
            component_type=component_type,
            name=component_id,
        )
    end
    return nothing
end

function _constraint_id(block, tag, indices)
    suffix = isempty(indices) ? "" : ":" * join(string.(indices), ",")
    return string(block, ".", tag, suffix)
end

function _resolve_sections(sections, block_sections::Dict{Symbol,Symbol})
    section_map = Dict{Symbol,Symbol}()
    if sections !== nothing
        for sec in sections
            for block in sec.blocks
                if hasproperty(block, :name)
                    section_map[getproperty(block, :name)] = sec.name
                else
                    section_map[Symbol(nameof(typeof(block)))] = sec.name
                end
            end
        end
    end
    for (k, v) in block_sections
        section_map[k] = v
    end
    return section_map
end

function _component_type(block::Symbol; section=nothing,
    overrides::Dict{Symbol,Symbol}=Dict{Symbol,Symbol}(),
    section_overrides::Dict{Symbol,Symbol}=Dict{Symbol,Symbol}())
    if haskey(overrides, block)
        return _component_type_from_symbol(overrides[block])
    end
    if section !== nothing && haskey(section_overrides, section)
        return _component_type_from_symbol(section_overrides[section])
    end
    if section !== nothing
        default = _default_component_type_by_section()
        if haskey(default, section)
            return default[section]
        end
    end
    return _component_type_from_block(block)
end

"""
    constraint_kind_enum(sym)

Convert a `Symbol` to `DualSignals.ConstraintKind`.

This is used by tag/component mappers to attach consistent constraint metadata
to rendered equations and results.
"""
constraint_kind_enum(sym::Symbol) = _enum_by_name(DualSignals.ConstraintKind, sym)

"""
    component_type_enum(sym)

Convert a `Symbol` to `DualSignals.ComponentType`.
"""
component_type_enum(sym::Symbol) = _enum_by_name(DualSignals.ComponentType, sym)

function _enum_by_name(::Type{T}, name::Symbol) where {T}
    for val in Base.Enums.instances(T)
        if string(val) == string(name)
            return val
        end
    end
    error("Unknown enum value $(name) for $(T)")
end

_constraint_kind_enum(sym::Symbol) = constraint_kind_enum(sym)
_component_type_enum(sym::Symbol) = component_type_enum(sym)
_constraint_sense_enum(sym::Symbol) = _enum_by_name(DualSignals.ConstraintSense, sym)
_objective_sense_enum(sym::Symbol) = _enum_by_name(DualSignals.ObjectiveSense, sym)

function _component_type_from_symbol(sym)
    text = lowercase(string(sym))
    if text in ("sector",)
        return _component_type_enum(:sector)
    elseif text in ("product", "commodity")
        return _component_type_enum(:product)
    elseif text in ("agent", "household", "government", "firm")
        return _component_type_enum(:agent)
    elseif text in ("source", "factor")
        return _component_type_enum(:source)
    elseif text in ("sink",)
        return _component_type_enum(:sink)
    elseif text in ("link", "trade", "external")
        return _component_type_enum(:link)
    elseif text in ("node", "market")
        return _component_type_enum(:node)
    else
        return _component_type_enum(:other)
    end
end

function _default_component_type_by_section()
    return Dict(
        :production => _component_type_enum(:sector),
        :factors => _component_type_enum(:source),
        :households => _component_type_enum(:agent),
        :government => _component_type_enum(:agent),
        :savings => _component_type_enum(:node),
        :prices => _component_type_enum(:node),
        :external => _component_type_enum(:link),
        :trade => _component_type_enum(:link),
        :markets => _component_type_enum(:node),
        :objective => _component_type_enum(:other),
        :init => _component_type_enum(:other),
        :closure => _component_type_enum(:other),
    )
end

function _component_type_from_block(block::Symbol)
    name = lowercase(string(block))
    if occursin("prod", name) || occursin("production", name)
        return _component_type_enum(:sector)
    elseif occursin("household", name) || occursin("hh", name)
        return _component_type_enum(:agent)
    elseif occursin("gov", name) || occursin("government", name)
        return _component_type_enum(:agent)
    elseif occursin("factor", name) || occursin("endowment", name)
        return _component_type_enum(:source)
    elseif occursin("trade", name) || occursin("armington", name) || occursin("cet", name) || occursin("external", name)
        return _component_type_enum(:link)
    elseif occursin("market", name) || occursin("price", name) || occursin("numeraire", name)
        return _component_type_enum(:node)
    else
        return _component_type_enum(:other)
    end
end

function _constraint_kind(block::Symbol, tag::Symbol; section=nothing,
    tag_overrides::Dict{Symbol,Symbol}=Dict{Symbol,Symbol}(),
    block_overrides::Dict{Symbol,Symbol}=Dict{Symbol,Symbol}(),
    section_overrides::Dict{Symbol,Symbol}=Dict{Symbol,Symbol}())
    if haskey(tag_overrides, tag)
        return _constraint_kind_from_symbol(tag_overrides[tag])
    end
    if haskey(block_overrides, block)
        return _constraint_kind_from_symbol(block_overrides[block])
    end
    if section !== nothing && haskey(section_overrides, section)
        return _constraint_kind_from_symbol(section_overrides[section])
    end
    if section !== nothing
        default = _default_constraint_kind_by_section()
        if haskey(default, section)
            return default[section]
        end
    end
    return _constraint_kind_from_tag(block, tag)
end

function _constraint_kind_from_symbol(sym)
    text = lowercase(string(sym))
    if text in ("balance", "market")
        return _constraint_kind_enum(:balance)
    elseif text in ("technology", "production")
        return _constraint_kind_enum(:technology)
    elseif text in ("resource", "endowment")
        return _constraint_kind_enum(:resource)
    elseif text in ("policy", "tax", "tariff")
        return _constraint_kind_enum(:policy_cap)
    elseif text in ("capacity",)
        return _constraint_kind_enum(:capacity)
    else
        return _constraint_kind_enum(:other)
    end
end

function _default_constraint_kind_by_section()
    return Dict(
        :production => _constraint_kind_enum(:technology),
        :factors => _constraint_kind_enum(:resource),
        :households => _constraint_kind_enum(:balance),
        :government => _constraint_kind_enum(:policy_cap),
        :savings => _constraint_kind_enum(:balance),
        :prices => _constraint_kind_enum(:balance),
        :external => _constraint_kind_enum(:balance),
        :trade => _constraint_kind_enum(:technology),
        :markets => _constraint_kind_enum(:balance),
        :objective => _constraint_kind_enum(:other),
        :init => _constraint_kind_enum(:other),
        :closure => _constraint_kind_enum(:policy_cap),
    )
end

function _constraint_kind_from_tag(block::Symbol, tag::Symbol)
    tag_map = DEFAULT_CONSTRAINT_KIND_TAG_MAP
    if haskey(tag_map, tag)
        return tag_map[tag]
    end
    name = lowercase(string(tag))
    if occursin("tax", name) || occursin("tariff", name) || occursin("tm", name) || occursin("tz", name) || occursin("td", name)
        return _constraint_kind_enum(:policy_cap)
    elseif occursin("market", name) || occursin("clear", name) || occursin("balance", name) || occursin("eqpqd", name) || occursin("eqpx", name) || occursin("eqpf", name) || occursin("eqepsilon", name)
        return _constraint_kind_enum(:balance)
    elseif occursin("supply", name) || occursin("endowment", name) || occursin("ff", name)
        return _constraint_kind_enum(:resource)
    elseif occursin("prod", name) || occursin("tech", name) || occursin("armington", name) || occursin("cet", name) || occursin("eqpqs", name) || occursin("eqpzd", name) || occursin("eqpzs", name)
        return _constraint_kind_enum(:technology)
    elseif occursin("capacity", name) || occursin("quota", name)
        return _constraint_kind_enum(:capacity)
    else
        return _constraint_kind_enum(:other)
    end
end


"""
    DEFAULT_CONSTRAINT_KIND_TAG_MAP

Default map of equation tags to `DualSignals.ConstraintKind`.

Used by result export and reporting to categorize constraints when explicit
metadata is not provided in the model.
"""
const DEFAULT_CONSTRAINT_KIND_TAG_MAP = Dict{Symbol,DualSignals.ConstraintKind}(
    # Technology / production / transformation
    :eqpy => _constraint_kind_enum(:technology),
    :eqF => _constraint_kind_enum(:technology),
    :eqX => _constraint_kind_enum(:technology),
    :eqY => _constraint_kind_enum(:technology),
    :eqpzs => _constraint_kind_enum(:technology),
    :eqZ => _constraint_kind_enum(:technology),
    :eqpqs => _constraint_kind_enum(:technology),
    :eqM => _constraint_kind_enum(:technology),
    :eqD => _constraint_kind_enum(:technology),
    :eqpzd => _constraint_kind_enum(:technology),
    :eqE => _constraint_kind_enum(:technology),
    :eqDs => _constraint_kind_enum(:technology),
    :eqfe => _constraint_kind_enum(:technology),
    :eqfm => _constraint_kind_enum(:technology),
    :eqRT => _constraint_kind_enum(:technology),
    :eqII => _constraint_kind_enum(:technology),
    :eqIII => _constraint_kind_enum(:technology),
    :eqCC => _constraint_kind_enum(:technology),
    :eqUU => _constraint_kind_enum(:technology),

    # Balance / market clearing / price links
    :eqQ => _constraint_kind_enum(:balance),
    :eqP => _constraint_kind_enum(:balance),
    :eqpf => _constraint_kind_enum(:balance),
    :eqpf1 => _constraint_kind_enum(:balance),
    :eqpf2 => _constraint_kind_enum(:balance),
    :eqpf3 => _constraint_kind_enum(:balance),
    :eqBOP => _constraint_kind_enum(:balance),
    :eqpe => _constraint_kind_enum(:balance),
    :eqpm => _constraint_kind_enum(:balance),
    :eqpw => _constraint_kind_enum(:balance),
    :eqw => _constraint_kind_enum(:balance),
    :eqPRICE => _constraint_kind_enum(:balance),
    :eqpk => _constraint_kind_enum(:balance),

    # Policy / taxes / savings
    :eqTd => _constraint_kind_enum(:policy_cap),
    :eqTz => _constraint_kind_enum(:policy_cap),
    :eqTm => _constraint_kind_enum(:policy_cap),
    :eqSp => _constraint_kind_enum(:policy_cap),
    :eqSg => _constraint_kind_enum(:policy_cap),
    :eqXg => _constraint_kind_enum(:policy_cap),
    :eqXv => _constraint_kind_enum(:policy_cap),

    # Resource constraints / endowments
    :eqFF => _constraint_kind_enum(:resource),

    # Capacity / quotas / complementarity
    :eqchi1 => _constraint_kind_enum(:capacity),
    :eqchi2 => _constraint_kind_enum(:capacity),
)

function _objective_sense(value)
    if value isa DualSignals.ObjectiveSense
        return value
    elseif value isa Symbol
        return _objective_sense(String(value))
    elseif value isa AbstractString
        if lowercase(value) in ("max", "maximize")
            return _objective_sense_enum(:maximize)
        elseif lowercase(value) in ("min", "minimize")
            return _objective_sense_enum(:minimize)
        end
    end
    return nothing
end

function _float_or_nothing(value)
    value === nothing && return nothing
    return Float64(value)
end

function _string_or_nothing(value)
    value === nothing && return nothing
    return string(value)
end

function _notes_from_metadata(metadata::Dict{Symbol,Any}, description)
    notes = String[]
    base = get(metadata, :notes, nothing)
    if base !== nothing
        push!(notes, string(base))
    end
    if haskey(metadata, :currency)
        push!(notes, "currency=$(metadata[:currency])")
    end
    if haskey(metadata, :numeraire)
        push!(notes, "numeraire=$(metadata[:numeraire])")
    end
    if haskey(metadata, :closure_flags)
        push!(notes, "closure_flags=$(metadata[:closure_flags])")
    elseif haskey(metadata, :closure)
        push!(notes, "closure=$(metadata[:closure])")
    end
    if isempty(notes)
        return _string_or_nothing(description)
    end
    return join(notes, "; ")
end

function _symbol_dict(obj)
    out = Dict{Symbol,Float64}()
    for (k, v) in pairs(obj)
        out[Symbol(k)] = Float64(v)
    end
    return out
end

function _namedtuple_list(obj)
    out = NamedTuple[]
    for entry in obj
        push!(out, NamedTuple(entry))
    end
    return out
end

function _split_symbol(sym::Symbol)
    text = String(sym)
    parts = split(text, ".")
    if length(parts) >= 2
        return Symbol(parts[1]), Symbol(parts[2])
    end
    return :unknown, sym
end

function results_from_table(tbl)
    primals = Dict{Symbol,Float64}()
    reduced = Dict{Symbol,Float64}()
    duals = NamedTuple[]
    complements = NamedTuple[]

    cols = Tables.columntable(tbl)
    symbols = get(cols, :symbol, String[])
    values = get(cols, :value, Float64[])
    kinds = get(cols, :kind, Symbol[])
    indices_col = get(cols, :index_tuple, fill("", length(symbols)))

    for i in eachindex(symbols)
        kind = Symbol(kinds[i])
        sym = Symbol(symbols[i])
        value = Float64(values[i])
        indices = _decode_indices(indices_col[i])
        if kind == :level
            primals[sym] = value
        elseif kind == :reduced_cost
            reduced[sym] = value
        elseif kind == :dual
            block, tag = _split_symbol(sym)
            push!(duals, (block=block, tag=tag, indices=indices, value=value))
        elseif kind == :complement
            push!(complements, (block=:mcp, tag=sym, indices=indices, var=sym, value=value, residual=nothing))
        end
    end
    return Results(primals, reduced, duals, complements, Dict{Symbol,Any}())
end

function _index_field(indices::Tuple, encode::Bool)
    if !encode
        return indices
    end
    return _encode_indices(indices)
end

function _encode_indices(indices::Tuple)
    if isempty(indices)
        return ""
    end
    return join(string.(indices), "|")
end

function _decode_indices(value)
    if value === nothing || value === missing
        return ()
    end
    text = string(value)
    isempty(text) && return ()
    parts = split(text, "|")
    return Tuple(Symbol.(parts))
end

function _global_var(base::Symbol, idxs::Symbol...)
    if isempty(idxs)
        return base
    end
    return Symbol(string(base), "_", join(string.(idxs), "_"))
end

function _resolve_indices(idxs, default_idxs, env::Dict{Symbol,Symbol})
    if idxs === nothing
        if default_idxs isa Tuple
            return Symbol[default_idxs...]
        elseif default_idxs isa AbstractVector
            return Symbol[default_idxs...]
        else
            return Symbol[]
        end
    elseif isempty(idxs)
        return Symbol[]
    end
    out = Symbol[]
    for idx in idxs
        if idx isa EIndex
            haskey(env, idx.name) || error("Unbound index: $(idx.name)")
            push!(out, env[idx.name])
        elseif idx isa Symbol
            push!(out, idx)
        else
            error("Unsupported index type: $(typeof(idx))")
        end
    end
    return out
end

function _index_env(index_names, indices)
    env = Dict{Symbol,Symbol}()
    index_names === nothing && return env
    for (name, value) in zip(index_names, indices)
        env[name] = value
    end
    return env
end

function _stringify_keys(dict)
    out = Dict{String,Any}()
    for (k, v) in dict
        out[string(k)] = v
    end
    return out
end

function _render_equations(eqs; format::Symbol, level::Symbol, show_defs::Bool)
    if level != :block && level != :equation
        error("Unsupported level: $(level). Use :block or :equation")
    end
    if format != :markdown && format != :latex && format != :plain
        error("Unsupported format: $(format). Use :markdown, :latex, or :plain")
    end
    lines = String[]
    if format == :markdown
        push!(lines, "# Equations")
    elseif format == :latex
        push!(lines, "% Equations")
    else
        push!(lines, "EQUATIONS")
    end
    if isempty(eqs)
        push!(lines, _format_text(format, "No equations registered."))
        return join(lines, "\n")
    end
    if level == :block
        by_block = Dict{Symbol,Vector{NamedTuple}}()
        for eq in eqs
            get!(by_block, eq.block, NamedTuple[])
            push!(by_block[eq.block], eq)
        end
        for (block, block_eqs) in sort(collect(by_block); by=first)
            append!(lines, _render_block_section(block, block_eqs; format=format, show_defs=show_defs))
        end
    else
        for eq in eqs
            push!(lines, _render_equation_line(eq; format=format, show_defs=show_defs))
        end
    end
    return join(lines, "\n")
end

"""
    _render_block_list(blocks; format)

Render a compact list of blocks for a section or model summary.
"""
function _render_block_list(blocks; format::Symbol)
    lines = String[]
    header = "Blocks"
    if format == :markdown
        push!(lines, "# $(header)")
    elseif format == :latex
        push!(lines, "% $(header)")
    else
        push!(lines, header)
    end
    for block in blocks
        push!(lines, _format_bullet(format, _block_label(block)))
    end
    return join(lines, "\n")
end

function _block_label(block)
    if hasproperty(block, :name)
        return string(getproperty(block, :name))
    end
    return string(nameof(typeof(block)))
end

"""
    _render_symbol_table(rows; format, show_values)

Dispatch to the table renderer for the chosen `format`.
"""
function _render_symbol_table(rows; format::Symbol, show_values::Bool)
    if format == :markdown
        return _render_symbol_table_markdown(rows; show_values=show_values)
    elseif format == :latex
        return _render_symbol_table_latex(rows; show_values=show_values)
    else
        return _render_symbol_table_plain(rows; show_values=show_values)
    end
end

"""
    _render_symbol_table_markdown(rows; show_values)

Render a symbol table in Markdown.
"""
function _render_symbol_table_markdown(rows; show_values::Bool)
    cols = show_values ? ["symbol", "value", "lower", "upper", "fixed"] : ["symbol", "lower", "upper", "fixed"]
    lines = String[]
    push!(lines, "# Symbols")
    push!(lines, "| " * join(cols, " | ") * " |")
    push!(lines, "|" * join(fill("---", length(cols)), "|") * "|")
    for row in rows
        vals = show_values ?
            [_fmt_cell(row.symbol), _fmt_cell(row.value), _fmt_cell(row.lower), _fmt_cell(row.upper), _fmt_cell(row.fixed)] :
            [_fmt_cell(row.symbol), _fmt_cell(row.lower), _fmt_cell(row.upper), _fmt_cell(row.fixed)]
        push!(lines, "| " * join(vals, " | ") * " |")
    end
    return join(lines, "\n")
end

"""
    _render_symbol_table_latex(rows; show_values)

Render a symbol table in LaTeX.
"""
function _render_symbol_table_latex(rows; show_values::Bool)
    cols = show_values ? ["symbol", "value", "lower", "upper", "fixed"] : ["symbol", "lower", "upper", "fixed"]
    lines = String[]
    push!(lines, "% Symbols")
    push!(lines, "% " * join(cols, " | "))
    for row in rows
        vals = show_values ?
            [_fmt_cell(row.symbol), _fmt_cell(row.value), _fmt_cell(row.lower), _fmt_cell(row.upper), _fmt_cell(row.fixed)] :
            [_fmt_cell(row.symbol), _fmt_cell(row.lower), _fmt_cell(row.upper), _fmt_cell(row.fixed)]
        push!(lines, "% " * join(vals, " | "))
    end
    return join(lines, "\n")
end

"""
    _render_symbol_table_plain(rows; show_values)

Render a symbol table in plain text.
"""
function _render_symbol_table_plain(rows; show_values::Bool)
    cols = show_values ? ["symbol", "value", "lower", "upper", "fixed"] : ["symbol", "lower", "upper", "fixed"]
    lines = String[]
    push!(lines, "SYMBOLS")
    push!(lines, join(cols, "\t"))
    for row in rows
        vals = show_values ?
            [_fmt_cell(row.symbol), _fmt_cell(row.value), _fmt_cell(row.lower), _fmt_cell(row.upper), _fmt_cell(row.fixed)] :
            [_fmt_cell(row.symbol), _fmt_cell(row.lower), _fmt_cell(row.upper), _fmt_cell(row.fixed)]
        push!(lines, join(vals, "\t"))
    end
    return join(lines, "\n")
end

"""
    _render_block_section(block, eqs; format, show_defs)

Render a block heading followed by its equations.
"""
function _render_block_section(block, eqs; format::Symbol, show_defs::Bool)
    lines = String[]
    header = "Block: $(block)"
    if format == :markdown
        push!(lines, "## $(header)")
    elseif format == :latex
        push!(lines, "% $(header)")
    else
        push!(lines, header)
    end
    for eq in eqs
        push!(lines, _render_equation_line(eq; format=format, show_defs=show_defs))
    end
    return lines
end

"""
    _render_equation_line(eq; format, show_defs)

Render a single equation line with label and optional domain annotations.
"""
function _render_equation_line(eq; format::Symbol, show_defs::Bool)
    info, is_math = _equation_info(eq; format=format)
    domains = Pair{String,Vector{String}}[]
    if format == :markdown
        domains = _extract_domains(eq)
    end
    label = ""
    if show_defs
        label = _equation_label(eq)
    end
    if format == :markdown
        if is_math
            if isempty(label)
                return string("\$\$\n", info, "\n\$\$\n", _render_domains(domains))
            end
            return string("`", label, "`\n\n\$\$\n", info, "\n\$\$\n", _render_domains(domains))
        end
        return isempty(label) ? "$(info)\n" : "`$(label)` $(info)\n"
    elseif format == :latex
        if isempty(label)
            return "% $(info)"
        else
            return "% $(label) $(info)"
        end
    else
        if isempty(label)
            return "* $(info)"
        else
            return "* $(label) $(info)"
        end
    end
end

function _extract_domains(eq)
    payload = eq.payload
    if payload isa NamedTuple
        expr = get(payload, :expr, nothing)
        if expr isa EquationExpr
            return _collect_domains(expr)
        end
    end
    return Pair{String,Vector{String}}[]
end

function _collect_domains(expr::EquationExpr)
    domains = Pair{String,Vector{String}}[]
    _collect_domains!(domains, expr)
    return domains
end

function _collect_domains!(domains::Vector{Pair{String,Vector{String}}}, expr::EquationExpr)
    if expr isa ESum || expr isa EProd
        idx = string(expr.index)
        domain = map(x -> string(x), expr.domain)
        push!(domains, idx => domain)
        _collect_domains!(domains, expr.expr)
    elseif expr isa EAdd
        for term in expr.terms
            _collect_domains!(domains, term)
        end
    elseif expr isa EMul
        for factor in expr.factors
            _collect_domains!(domains, factor)
        end
    elseif expr isa EPow
        _collect_domains!(domains, expr.base)
        _collect_domains!(domains, expr.exponent)
    elseif expr isa EDiv
        _collect_domains!(domains, expr.numerator)
        _collect_domains!(domains, expr.denominator)
    elseif expr isa ENeg
        _collect_domains!(domains, expr.expr)
    elseif expr isa EEq
        _collect_domains!(domains, expr.lhs)
        _collect_domains!(domains, expr.rhs)
    end
    return domains
end

"""
    _render_domains(domains)

Render domain annotations (index sets) for an equation.
"""
function _render_domains(domains::Vector{Pair{String,Vector{String}}})
    if isempty(domains)
        return ""
    end
    lines = String[]
    for (idx, domain) in domains
        push!(lines, "Domain $(idx) in { $(join(domain, ", ")) }\n")
    end
    return "\n" * join(lines, "")
end

function _equation_info(eq; format::Symbol)
    payload = eq.payload
    if payload isa NamedTuple
        expr = get(payload, :expr, nothing)
        if expr !== nothing
            if expr isa EquationExpr
                is_math = !(expr isa ERaw)
                return render_expr(expr; format=format), is_math
            end
            return string(expr), false
        end
        info = get(payload, :info, nothing)
        if info === nothing
            constraint = get(payload, :constraint, nothing)
            return (constraint === nothing ? "(no info)" : string(constraint)), false
        end
        return string(info), false
    elseif payload isa AbstractString
        return payload, false
    else
        return string(payload), false
    end
end

function _equation_label(eq)
    payload = eq.payload
    idxs = ()
    if payload isa NamedTuple && haskey(payload, :indices)
        idxs = payload.indices
    end
    idx_text = _format_indices(idxs)
    if isempty(idx_text)
        return string(eq.block, ".", eq.tag)
    end
    return string(eq.block, ".", eq.tag, "[", idx_text, "]")
end

function _format_indices(idxs)
    if idxs === nothing
        return ""
    elseif idxs isa Tuple || idxs isa AbstractVector
        return join(map(string, idxs), ",")
    else
        return string(idxs)
    end
end

function _format_text(format::Symbol, text::String)
    if format == :markdown
        return text
    elseif format == :latex
        return "% $(text)"
    else
        return text
    end
end

"""
    render_expr(expr; format=:plain)

Render an equation AST node to a string.

Supported formats: `:plain`, `:markdown`, `:latex`. This is the lowest-level
renderer used by `render_equations` and the equation file generators.
"""
function render_expr(expr::EquationExpr; format::Symbol=:plain)
    return _render_expr(expr; format=format)
end

"""
    _render_expr(expr; format)

Recursive expression renderer for the equation AST.
"""
function _render_expr(expr::EquationExpr; format::Symbol)
    if format == :markdown
        format = :latex
    end
    if expr isa EVar
        return _render_symbol(expr.name, expr.idxs; format=format)
    elseif expr isa EParam
        return _render_symbol(expr.name, expr.idxs; format=format)
    elseif expr isa EConst
        if expr.value isa AbstractFloat && isfinite(expr.value) && isinteger(expr.value)
            return string(Int(expr.value))
        end
        return string(expr.value)
    elseif expr isa ERaw
        return expr.text
    elseif expr isa EIndex
        return string(expr.name)
    elseif expr isa EAdd
        parts = String[]
        for (i, term) in enumerate(expr.terms)
            sign = "+"
            render_term = term
            if term isa ENeg
                sign = "-"
                render_term = term.expr
            elseif term isa EConst && term.value < 0
                sign = "-"
                render_term = EConst(-term.value)
            end
            rendered = _render_expr(render_term; format=format)
            if i == 1
                push!(parts, sign == "-" ? string("-", rendered) : rendered)
            else
                push!(parts, sign == "-" ? " - " : " + ")
                push!(parts, rendered)
            end
        end
        return join(parts, "")
    elseif expr isa EMul
        parts = map(t -> _wrap_if_needed(t, _render_expr(t; format=format); format=format), expr.factors)
        op = format == :latex ? " \\cdot " : " * "
        return join(parts, op)
    elseif expr isa EPow
        base = _wrap_if_needed(expr.base, _render_expr(expr.base; format=format); format=format)
        exp = _render_expr(expr.exponent; format=format)
        if format == :latex
            if expr.base isa EVar || expr.base isa EParam
                base = string("{", base, "}")
            end
            if expr.exponent isa EDiv
                num = expr.exponent.numerator
                den = expr.exponent.denominator
                if num isa EConst && num.value == 1
                    den_render = _render_exponent_expr(den)
                    return string(base, "^{1/(", den_render, ")}")
                end
            end
            exp = _render_exponent_expr(expr.exponent)
            return string(base, "^{", exp, "}")
        end
        return string(base, "^", exp)
    elseif expr isa EDiv
        num = _render_expr(expr.numerator; format=format)
        den = _render_expr(expr.denominator; format=format)
        if format == :latex
            if expr.numerator isa EAdd || expr.numerator isa ENeg
                num = string("(", num, ")")
            end
            if expr.denominator isa EAdd || expr.denominator isa ENeg
                den = string("(", den, ")")
            end
            return string(num, " / ", den)
        end
        return string(num, " / ", den)
    elseif expr isa ENeg
        inner = _wrap_if_needed(expr.expr, _render_expr(expr.expr; format=format); format=format)
        return string("-", inner)
    elseif expr isa ESum
        inner = _render_expr(expr.expr; format=format)
        if format == :latex
            idx = _latex_escape(string(expr.index))
            return string("\\sum_{", idx, " \\in \\mathcal{D}_{", idx, "}} ", inner)
        end
        domain = join(map(idx -> _latex_escape(string(idx)), expr.domain), ", ")
        return string("sum_", expr.index, "∈{", domain, "}(", inner, ")")
    elseif expr isa EProd
        inner = _render_expr(expr.expr; format=format)
        if format == :latex
            idx = _latex_escape(string(expr.index))
            return string("\\prod_{", idx, " \\in \\mathcal{D}_{", idx, "}} ", inner)
        end
        domain = join(map(idx -> _latex_escape(string(idx)), expr.domain), ", ")
        return string("prod_", expr.index, "∈{", domain, "}(", inner, ")")
    elseif expr isa EEq
        lhs = _render_expr(expr.lhs; format=format)
        rhs = _render_expr(expr.rhs; format=format)
        return string(lhs, " = ", rhs)
    else
        return string(expr)
    end
end

function _simplify_exponent_latex(text::AbstractString)
    simplified = replace(text, r"\\mathrm\\{([^}]*)\\}" => s"\1")
    simplified = replace(simplified, "\\_" => "_")
    return simplified
end

"""
    _render_exponent_expr(expr)

Render exponents using a plain representation to keep power formatting stable.
"""
function _render_exponent_expr(expr::EquationExpr)
    text = _render_expr(expr; format=:plain)
    text = replace(text, "[" => "_", "]" => "")
    text = replace(text, " " => "")
    return _latex_escape_exponent(text)
end

function _latex_escape_exponent(text::AbstractString)
    escaped = replace(text, "\\" => "\\textbackslash{}")
    escaped = replace(escaped, "#" => "\\#", "%" => "\\%", "&" => "\\&", "\$" => "\\\$", "^" => "\\^{}", "~" => "\\~{}")
    return escaped
end

function _wrap_if_needed(expr::EquationExpr, rendered::AbstractString; format::Symbol)
    if expr isa EAdd
        return string("(", rendered, ")")
    elseif expr isa EDiv
        return string("(", rendered, ")")
    end
    return rendered
end

"""
    _render_symbol(name, idxs; format)

Render a symbol name with optional indices.
"""
function _render_symbol(name::Symbol, idxs::Union{Nothing,Vector{Any}}; format::Symbol)
    if idxs === nothing || isempty(idxs)
        text = string(name)
        return format == :latex ? _latex_escape(text) : text
    end
    if format == :latex
        idx_text = join(map(idx -> _render_index(idx; format=format), idxs), ",")
        base = _latex_escape(string(name))
        if occursin("_", base)
            base = string("{", base, "}")
        end
        return string(base, "_{", idx_text, "}")
    end
    idx_text = join(map(idx -> _render_index(idx; format=format), idxs), ",")
    return string(name, "[", idx_text, "]")
end

"""
    _render_index(idx; format)

Render a single index value (symbol/number/string) in the chosen format.
"""
function _render_index(idx; format::Symbol)
    if idx isa EIndex
        text = string(idx.name)
        return format == :latex ? _latex_escape(text) : text
    end
    text = string(idx)
    return format == :latex ? string("\\text{", _latex_escape(text), "}") : text
end

function _latex_escape(text::AbstractString)
    escaped = replace(text, "\\" => "\\textbackslash{}")
    escaped = replace(escaped, "{" => "\\{", "}" => "\\}")
    escaped = replace(escaped, "_" => "\\_")
    escaped = replace(escaped, "#" => "\\#", "%" => "\\%", "&" => "\\&", "\$" => "\\\$", "^" => "\\^{}", "~" => "\\~{}")
    return escaped
end

function _format_bullet(format::Symbol, text::String)
    if format == :markdown
        return "- $(text)"
    elseif format == :latex
        return "% $(text)"
    else
        return "* $(text)"
    end
end

function _fmt_cell(value)
    if value === nothing
        return "-"
    elseif value isa Symbol
        return string(value)
    else
        return string(value)
    end
end

end # module
