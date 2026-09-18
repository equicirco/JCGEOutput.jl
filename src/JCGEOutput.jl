"""
Output utilities: equation rendering, results collection, persistence, and reporting.
"""
module JCGEOutput

import JCGECore
using JCGECalibrate
using Arrow
using CSV
using JSON3
using JCGECore: EquationExpr, EIndex, EVar, EParam, EConst, EAdd, EMul, EPow, EDiv, ENeg, ELog, ESum, EProd, EEq, ELe, EGe, ERaw
using JCGERuntime
using Dates
using JuMP
using Parquet
using Tables
using DualSignals

export render_equations, render_block
export EquationFamily, equation_families, EquationTemplate, EquationReportMapping, AdditiveSumMapping, equation_templates
export render_equation_report
export render_symbols, render_blocks, render_sections
export Results, collect_results, tidy, to_json, to_csv
export results_from_json, results_from_csv, results_from_arrow, results_from_parquet
export to_arrow, to_parquet
export to_dualsignals, write_dualsignals_json, write_dualsignals_csv
export sam_from_solution, write_sam_csv
export SatelliteAnchor, SatelliteReference, SatelliteBalance
export satellite_reference, satellite_calibration_report, satellite_projection, satellite_balances
export DEFAULT_CONSTRAINT_KIND_TAG_MAP, constraint_kind_enum, component_type_enum
export EquationExpr, EIndex, EVar, EParam, EConst, EAdd, EMul, EPow, EDiv, ENeg, ELog, ESum, EProd, EEq, ELe, EGe, ERaw
export render_expr

"""
Container for solver results and metadata.

Fields:
- `primals`: variable levels by symbol.
- `reduced_costs`: reduced costs by symbol.
- `duals`: vector of (block, tag, indices, value).
- `complements`: complementarity diagnostics for MCP equations.
- `accounting_checks`: residuals for closure conditions evaluated after solving.
- `metadata`: solver and scenario metadata.
"""
struct Results
    primals::Dict{Symbol,Float64}
    reduced_costs::Dict{Symbol,Float64}
    duals::Vector{NamedTuple}
    complements::Vector{NamedTuple}
    accounting_checks::Vector{NamedTuple}
    metadata::Dict{Symbol,Any}
end

Results(primals::Dict{Symbol,Float64}, reduced_costs::Dict{Symbol,Float64},
    duals::Vector{NamedTuple}, complements::Vector{NamedTuple},
    metadata::Dict{Symbol,Any}) =
    Results(primals, reduced_costs, duals, complements, NamedTuple[], metadata)

include("satellites.jl")

"""
    collect_results(obj; metadata=Dict()) -> Results

Collect primals, reduced costs, duals, complementarity diagnostics, and
post-solution accounting checks
from a `KernelContext`, `RunSpec` result, or equivalent object.
"""
function collect_results(obj; metadata=Dict{Symbol,Any}())
    ctx = _context(obj)
    primals = Dict{Symbol,Float64}()
    reduced_costs = Dict{Symbol,Float64}()
    duals = NamedTuple[]
    complements = NamedTuple[]
    accounting_checks = NamedTuple[]

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

    for entry in JCGERuntime.equation_residuals(ctx)
        entry.role == :accounting_check || continue
        push!(accounting_checks, (
            block=entry.block,
            tag=entry.tag,
            indices=entry.indices,
            residual=entry.residual,
        ))
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

    return Results(primals, reduced_costs, duals, complements, accounting_checks, meta)
end

"""
    tidy(results; kinds=..., encode_indices=false) -> Vector{NamedTuple}

Return a long-table representation of results suitable for CSV/Arrow/Parquet.
"""
function tidy(results::Results;
    kinds=(:level, :dual, :reduced_cost, :complement, :accounting_check),
    encode_indices::Bool=false)
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
    if :accounting_check in kinds
        for entry in results.accounting_checks
            sym = Symbol(string(entry.block), ".", string(entry.tag))
            push!(rows, (
                symbol=string(sym),
                index_tuple=_index_field(Tuple(entry.indices), encode_indices),
                value=entry.residual,
                kind=:accounting_check,
            ))
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
        accounting_checks=results.accounting_checks,
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
function to_csv(results::Results, path::AbstractString;
    kinds=(:level, :dual, :reduced_cost, :complement, :accounting_check))
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
function to_arrow(results::Results, path::AbstractString;
    kinds=(:level, :dual, :reduced_cost, :complement, :accounting_check))
    rows = tidy(results; kinds=kinds, encode_indices=true)
    Arrow.write(path, rows)
    return path
end

"""
    to_parquet(results, path; kinds=...)

Write a tidy table to Parquet.
"""
function to_parquet(results::Results, path::AbstractString;
    kinds=(:level, :dual, :reduced_cost, :complement, :accounting_check))
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
    accounting_checks = _namedtuple_list(get(data, "accounting_checks", Any[]))
    return Results(primals, reduced, duals, complements, accounting_checks, metadata)
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

    for entry in results.accounting_checks
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
        slack = abs(entry.residual)
        push!(solutions, DualSignals.ConstraintSolution(
            constraint_id=constraint_id,
            dual=0.0,
            slack=slack,
            is_binding=slack <= 1e-8,
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
    render_equations(obj; format=:markdown, level=:block, view=:expanded,
        show_defs=true, show_condition_roles=false)

Render equations registered in a `KernelContext` or run result.

Inputs
- `obj`: a `JCGERuntime.KernelContext`, a run result (`NamedTuple` with `context`),
  or a `JCGECore.RunSpec` (via a `KernelContext`).
- `format`: `:markdown`, `:latex`, or `:plain`.
- `level`: `:block` to group equations by block, or `:equation` for a flat list.
- `view`: `:expanded` for every registered instance or `:family` for exact
  equation families.
- `show_defs`: include equation labels and block tags.
- `show_condition_roles`: append each equation's closure role to its label.

Returns a formatted string. The output is derived from the equation AST, not
solver-specific objects, so it is backend-agnostic.
"""
function render_equations(obj; format::Symbol=:markdown, level::Symbol=:block,
    view::Symbol=:expanded, show_defs::Bool=true, show_condition_roles::Bool=false)
    ctx = _context(obj)
    eqs = JCGERuntime.list_equations(ctx)
    if view == :family
        return _render_equation_families(_equation_families(eqs); format=format,
            level=level, show_defs=show_defs,
            show_condition_roles=show_condition_roles)
    elseif view != :expanded
        error("Unsupported view: $(view). Use :expanded or :family")
    end
    return _render_equations(eqs; format=format, level=level, show_defs=show_defs,
        show_condition_roles=show_condition_roles)
end

"""
    render_block(obj, block_id; format=:markdown, view=:expanded,
        show_defs=true, show_condition_roles=false)

Render equations for one block.

`block_id` can be a `Symbol` or a string-like identifier; it is converted to a
`Symbol` and matched against `EquationInfo.block` entries.

`view=:family` groups only structurally identical registered equations; the
default `:expanded` view retains every instance.
"""
function render_block(obj, block_id; format::Symbol=:markdown, view::Symbol=:expanded,
    show_defs::Bool=true, show_condition_roles::Bool=false)
    ctx = _context(obj)
    eqs = JCGERuntime.list_equations(ctx)
    block_sym = Symbol(block_id)
    eqs_block = filter(eq -> eq.block == block_sym, eqs)
    if view == :family
        return _render_equation_families(_equation_families(eqs_block); format=format,
            level=:equation, show_defs=show_defs,
            show_condition_roles=show_condition_roles)
    elseif view != :expanded
        error("Unsupported view: $(view). Use :expanded or :family")
    end
    return _render_equations(eqs_block; format=format, level=:equation,
        show_defs=show_defs, show_condition_roles=show_condition_roles)
end

"""
    EquationFamily

One exact structural family of registered equations. A family groups registered
equation instances only when their block, tag, equation kind, objective sense,
and equation AST are identical. It therefore never substitutes a hand-written
template for a model-derived equation.
"""
struct EquationFamily
    block::Symbol
    tag::Symbol
    kind::Symbol
    expression::Union{Nothing,EquationExpr}
    objective_sense
    condition_role
    instances::Vector{NamedTuple}
    domains::Vector{Pair{String,Vector{String}}}
end

"""
    EquationTemplate

One compact, indexed rendering of registered equation instances. A template is
formed only when the registry metadata identifies how every varying reference
depends on the declared equation indices. Calibrated numeric literals are
rendered as either their recorded parameter symbols or generated calibration
coefficient symbols. References that cannot be verified remain concrete,
causing separate templates rather than an inferred formula.
"""
struct EquationTemplate
    block::Symbol
    tag::Symbol
    kind::Symbol
    expression::Union{Nothing,EquationExpr}
    objective_sense
    condition_role
    instances::Vector{NamedTuple}
    domains::Vector{Pair{String,Vector{String}}}
    index_names::Tuple{Vararg{Symbol}}
end

"""
    AdditiveSumMapping(; path, index, domain, term_kind=:variable,
        term_name, index_position)

Explicitly render an enumerated additive expression as an indexed sum in an
equation report. `path` identifies the target expression from the root equation
(for example `(:lhs,)` or `(:rhs,)`). The target must be a direct `EAdd` of
references named `term_name` with kind `term_kind`. Their `index_position`-th
source index must cover `domain` exactly. All other source indices must be
identical within an equation instance.

The report consumer supplies this declaration. It is validated for every
registered equation instance and never inferred from encoded identifiers.
"""
struct AdditiveSumMapping
    path::Tuple{Vararg{Any}}
    index::Symbol
    domain::Vector{Symbol}
    term_kind::Symbol
    term_name::Symbol
    index_position::Int
end

function AdditiveSumMapping(; path, index, domain, term_kind=:variable,
    term_name, index_position)
    normalized_path = Tuple(path)
    isempty(normalized_path) && throw(ArgumentError(
        "additive-sum mapping `path` cannot be empty"))
    normalized_index = Symbol(index)
    normalized_domain = Symbol.(collect(domain))
    isempty(normalized_domain) && throw(ArgumentError(
        "additive-sum mapping `domain` cannot be empty"))
    length(unique(normalized_domain)) == length(normalized_domain) || throw(ArgumentError(
        "additive-sum mapping `domain` values must be unique"))
    normalized_kind = Symbol(term_kind)
    normalized_kind in (:variable, :parameter) || throw(ArgumentError(
        "additive-sum mapping `term_kind` must be :variable or :parameter"))
    normalized_position = Int(index_position)
    normalized_position > 0 || throw(ArgumentError(
        "additive-sum mapping `index_position` must be positive"))
    return AdditiveSumMapping(normalized_path, normalized_index, normalized_domain,
        normalized_kind, Symbol(term_name), normalized_position)
end

"""
    EquationReportMapping(; source_block, source_tag,
        report_block=source_block, report_tag=source_tag,
        index_names, coordinates, domain_values=Dict(), index_projections=Dict(),
        reference_indices=Dict(), additive_sums=AdditiveSumMapping[])

Explicitly declare how registered equation instances are represented by report
indices. `coordinates` maps exact registered index tuples to report coordinates;
the model or report consumer supplies and validates this mapping. `domain_values`,
`index_projections`, and `reference_indices` map explicit source values to report
indices without relying on encoded identifiers. `additive_sums` replaces declared
enumerated additive expressions by validated indexed sums in the report.
"""
struct EquationReportMapping
    source_block::Symbol
    source_tag::Symbol
    report_block::Symbol
    report_tag::Symbol
    index_names::Tuple{Vararg{Symbol}}
    coordinates::Dict{Tuple,Tuple}
    domain_values::Dict{Symbol,Symbol}
    index_projections::Dict{Symbol,Tuple{Vararg{Symbol}}}
    reference_indices::Dict{Tuple{Symbol,Symbol,Tuple},Tuple}
    additive_sums::Vector{AdditiveSumMapping}
end

function EquationReportMapping(; source_block, source_tag, report_block=source_block,
    report_tag=source_tag, index_names, coordinates, domain_values=Dict(),
    index_projections=Dict(), reference_indices=Dict(),
    additive_sums=AdditiveSumMapping[])
    names = Tuple(Symbol.(collect(index_names)))
    isempty(names) && throw(ArgumentError("report mapping `index_names` cannot be empty"))
    length(unique(names)) == length(names) ||
        throw(ArgumentError("report mapping `index_names` must be unique"))
    mapped_coordinates = Dict{Tuple,Tuple}()
    for (source_indices, report_indices) in coordinates
        source_key = _report_index_tuple(source_indices)
        report_value = _report_index_tuple(report_indices)
        length(report_value) == length(names) || throw(ArgumentError(
            "report coordinate $(report_value) has $(length(report_value)) values, " *
            "but $(length(names)) report indices were declared"))
        haskey(mapped_coordinates, source_key) &&
            throw(ArgumentError("duplicate report coordinate for source indices $(source_key)"))
        mapped_coordinates[source_key] = report_value
    end
    isempty(mapped_coordinates) &&
        throw(ArgumentError("report mapping `coordinates` cannot be empty"))
    mapped_domains = Dict{Symbol,Symbol}()
    for (source_value, report_value) in domain_values
        source_key = Symbol(source_value)
        haskey(mapped_domains, source_key) &&
            throw(ArgumentError("duplicate report domain value $(source_key)"))
        mapped_domains[source_key] = Symbol(report_value)
    end
    mapped_index_projections = Dict{Symbol,Tuple{Vararg{Symbol}}}()
    for (source_index, report_positions) in index_projections
        source_key = Symbol(source_index)
        positions = Tuple(Symbol.(collect(report_positions)))
        isempty(positions) && throw(ArgumentError(
            "report index projection $(source_key) must declare at least one report index"))
        all(position -> position in names, positions) || throw(ArgumentError(
            "report index projection $(source_key) contains an index not declared by this mapping"))
        haskey(mapped_index_projections, source_key) && throw(ArgumentError(
            "duplicate report index projection for $(source_key)"))
        mapped_index_projections[source_key] = positions
    end
    mapped_references = Dict{Tuple{Symbol,Symbol,Tuple},Tuple}()
    for (source_reference, report_positions) in reference_indices
        length(source_reference) == 3 || throw(ArgumentError(
            "report reference keys must be `(kind, name, source_indices)` tuples"))
        raw_kind, raw_name, raw_indices = source_reference
        kind = Symbol(raw_kind)
        kind in (:variable, :parameter) || throw(ArgumentError(
            "report reference kind must be :variable or :parameter, not $(kind)"))
        key = (kind, Symbol(raw_name), _report_index_tuple(raw_indices))
        positions = Tuple(Tuple(Symbol.(collect(position))) for position in report_positions)
        length(positions) == length(key[3]) || throw(ArgumentError(
            "report reference $(key) has $(length(key[3])) source indices, but " *
            "$(length(positions)) report index positions were declared"))
        all(!isempty, positions) || throw(ArgumentError(
            "each report reference position must declare at least one index name"))
        haskey(mapped_references, key) && throw(ArgumentError(
            "duplicate report reference mapping for $(key)"))
        mapped_references[key] = positions
    end
    mapped_additive_sums = AdditiveSumMapping[]
    for mapping in additive_sums
        mapping isa AdditiveSumMapping || throw(ArgumentError(
            "`additive_sums` entries must be AdditiveSumMapping values"))
        push!(mapped_additive_sums, mapping)
    end
    paths = [mapping.path for mapping in mapped_additive_sums]
    length(unique(paths)) == length(paths) || throw(ArgumentError(
        "additive-sum mapping paths must be unique within an equation report mapping"))
    return EquationReportMapping(Symbol(source_block), Symbol(source_tag), Symbol(report_block),
        Symbol(report_tag), names, mapped_coordinates, mapped_domains,
        mapped_index_projections, mapped_references, mapped_additive_sums)
end

"""
    equation_families(obj)

Return exact structural families from the equation registry of `obj`. Each
family retains the registered instances that it summarizes. When equations
differ structurally across indices, they remain separate families rather than
being silently generalized.
"""
function equation_families(obj)
    ctx = _context(obj)
    return _equation_families(JCGERuntime.list_equations(ctx))
end

"""
    equation_templates(obj; include_solver_annotations=false,
        report_mappings=EquationReportMapping[], strict_report_mappings=true)

Return compact indexed templates inferred solely from registered equation ASTs,
their declared `index_names`, their index tuples, and their parameter payloads.
In this compact representation, calibrated numeric literals are replaced by
symbols: a recorded parameter name when available, otherwise a deterministic
calibration-coefficient symbol. The inference is conservative: it generalizes
an instance only where the same mapping is verified across every candidate
instance. It never relies on symbol naming conventions. `report_mappings` lets
the consumer explicitly assign report indices to concrete registered instances;
with `strict_report_mappings=true`, missing or unused assignments are errors.
"""
function equation_templates(obj; include_solver_annotations::Bool=false,
    report_mappings::AbstractVector{<:EquationReportMapping}=EquationReportMapping[],
    strict_report_mappings::Bool=true)
    ctx = _context(obj)
    eqs = JCGERuntime.list_equations(ctx)
    if !include_solver_annotations
        eqs = filter(!_is_solver_annotation, eqs)
    end
    eqs = _apply_report_mappings(eqs, report_mappings; strict=strict_report_mappings)
    return _equation_templates(eqs)
end

"""
    render_equation_report(obj; format=:latex, view=:family,
        show_defs=true, show_condition_roles=false,
        include_solver_annotations=false, report_mappings=EquationReportMapping[],
        strict_report_mappings=true)

Render a model-derived equation report. `view=:family` gives one display for
each exact registered equation family and its number of instances;
`view=:expanded` displays every registered equation instance; and
`view=:indexed` derives compact indexed templates conservatively from the
registered AST and metadata. The indexed view replaces calibrated numeric
literals with symbols, so it is suitable for a mathematical specification;
numerical values belong in calibration tables. Objectives are reported
separately from equations. LaTeX output contains real mathematical environments
(not comments) and requires `amsmath` for `align*`.

By default, solver annotations (`start`, `lower`, `upper`, and `fixed`) are
excluded because they configure numerical solution rather than define the
model mathematically. Set `include_solver_annotations=true` for a complete
registry audit.

`report_mappings` is accepted only by `view=:indexed`. It lets the report
consumer explicitly map concrete registered instances to report indices, with
strict coverage validation enabled by default.
"""
function render_equation_report(obj; format::Symbol=:latex, view::Symbol=:family,
    show_defs::Bool=true, show_condition_roles::Bool=false,
    include_solver_annotations::Bool=false,
    report_mappings::AbstractVector{<:EquationReportMapping}=EquationReportMapping[],
    strict_report_mappings::Bool=true)
    ctx = _context(obj)
    eqs = JCGERuntime.list_equations(ctx)
    if !include_solver_annotations
        eqs = filter(!_is_solver_annotation, eqs)
    end
    !isempty(report_mappings) && view != :indexed && error(
        "`report_mappings` can only be used with `view=:indexed`")
    if view == :family
        return _render_equation_report_families(_equation_families(eqs); format=format,
            show_defs=show_defs, show_condition_roles=show_condition_roles)
    elseif view == :indexed
        eqs = _apply_report_mappings(eqs, report_mappings; strict=strict_report_mappings)
        return _render_equation_report_families(_equation_templates(eqs); format=format,
            show_defs=show_defs, show_condition_roles=show_condition_roles)
    elseif view == :expanded
        return _render_equation_report_expanded(eqs; format=format,
            show_defs=show_defs, show_condition_roles=show_condition_roles)
    end
    error("Unsupported view: $(view). Use :expanded, :family, or :indexed")
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
    accounting_checks = NamedTuple[]

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
        elseif kind == :accounting_check
            block, tag = _split_symbol(sym)
            push!(accounting_checks, (block=block, tag=tag, indices=indices, residual=value))
        end
    end
    return Results(primals, reduced, duals, complements, accounting_checks, Dict{Symbol,Any}())
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

function _render_equations(eqs; format::Symbol, level::Symbol, show_defs::Bool,
    show_condition_roles::Bool)
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
            append!(lines, _render_block_section(block, block_eqs; format=format,
                show_defs=show_defs, show_condition_roles=show_condition_roles))
        end
    else
        for eq in eqs
            push!(lines, _render_equation_line(eq; format=format, show_defs=show_defs,
                show_condition_roles=show_condition_roles))
        end
    end
    return join(lines, "\n")
end

function _equation_expression(eq)
    payload = eq.payload
    payload isa NamedTuple || return nothing, :description, nothing
    expr = get(payload, :expr, nothing)
    if expr isa EquationExpr
        return expr, :equation, nothing
    end
    objective_expr = get(payload, :objective_expr, nothing)
    if objective_expr isa EquationExpr
        return objective_expr, :objective, get(payload, :objective_sense, :Max)
    end
    return nothing, :description, nothing
end

const _SOLVER_ANNOTATION_TAGS = Set((:start, :lower, :upper, :fixed))

_is_solver_annotation(eq) = eq.tag in _SOLVER_ANNOTATION_TAGS

function _equation_condition_role(eq)
    payload = eq.payload
    return payload isa NamedTuple ? get(payload, :condition_role, :enforce) : nothing
end

function _equation_family_key(eq)
    expression, kind, objective_sense = _equation_expression(eq)
    expression_key = expression === nothing ? _equation_info(eq; format=:plain)[1] : repr(expression)
    return (eq.block, eq.tag, kind, objective_sense, _equation_condition_role(eq), expression_key)
end

function _equation_families(eqs)
    grouped = Dict{Tuple,Vector{NamedTuple}}()
    keys_in_order = Tuple[]
    for eq in eqs
        key = _equation_family_key(eq)
        if !haskey(grouped, key)
            grouped[key] = NamedTuple[]
            push!(keys_in_order, key)
        end
        push!(grouped[key], eq)
    end
    families = EquationFamily[]
    for key in keys_in_order
        instances = grouped[key]
        representative = first(instances)
        expression, kind, objective_sense = _equation_expression(representative)
        domains = expression === nothing ? Pair{String,Vector{String}}[] : _collect_domains(expression)
        push!(families, EquationFamily(representative.block, representative.tag, kind,
            expression, objective_sense, _equation_condition_role(representative), instances,
            domains))
    end
    return families
end

"""An inferred projection from one concrete index to declared equation indices."""
struct _IndexProjection
    names::Tuple{Vararg{Symbol}}
end

_report_index_tuple(value::Tuple) = value
_report_index_tuple(value::AbstractVector) = Tuple(value)
_report_index_tuple(value) = (value,)

function _registered_report_indices(eq)
    payload = eq.payload
    payload isa NamedTuple || return ()
    return _report_index_tuple(get(payload, :indices, ()))
end

function _report_mapping_lookup(mappings)
    lookup = Dict{Tuple{Symbol,Symbol},EquationReportMapping}()
    for mapping in mappings
        selector = (mapping.source_block, mapping.source_tag)
        haskey(lookup, selector) && throw(ArgumentError(
            "multiple report mappings select $(mapping.source_block).$(mapping.source_tag)"))
        lookup[selector] = mapping
    end
    return lookup
end

function _expression_at_path(expr::EquationExpr, path::Tuple{Vararg{Any}})
    isempty(path) && return expr
    step = first(path)
    remaining = Tuple(path[2:end])
    if expr isa EEq || expr isa ELe || expr isa EGe
        step === :lhs && return _expression_at_path(expr.lhs, remaining)
        step === :rhs && return _expression_at_path(expr.rhs, remaining)
    elseif expr isa EAdd
        step === :term || throw(ArgumentError("expected `:term` in additive expression path"))
        isempty(remaining) && throw(ArgumentError("additive expression path is missing a term position"))
        position = first(remaining)
        position isa Integer || throw(ArgumentError(
            "additive expression path term position must be an integer"))
        1 <= position <= length(expr.terms) || throw(ArgumentError(
            "additive expression path term position $(position) is out of bounds"))
        return _expression_at_path(expr.terms[position], Tuple(remaining[2:end]))
    elseif expr isa EMul
        step === :factor || throw(ArgumentError("expected `:factor` in multiplicative expression path"))
        isempty(remaining) && throw(ArgumentError(
            "multiplicative expression path is missing a factor position"))
        position = first(remaining)
        position isa Integer || throw(ArgumentError(
            "multiplicative expression path factor position must be an integer"))
        1 <= position <= length(expr.factors) || throw(ArgumentError(
            "multiplicative expression path factor position $(position) is out of bounds"))
        return _expression_at_path(expr.factors[position], Tuple(remaining[2:end]))
    elseif expr isa EPow
        step === :base && return _expression_at_path(expr.base, remaining)
        step === :exponent && return _expression_at_path(expr.exponent, remaining)
    elseif expr isa EDiv
        step === :numerator && return _expression_at_path(expr.numerator, remaining)
        step === :denominator && return _expression_at_path(expr.denominator, remaining)
    elseif expr isa ENeg || expr isa ELog || expr isa ESum || expr isa EProd
        step === :expression && return _expression_at_path(expr.expr, remaining)
    end
    throw(ArgumentError("expression path $(path) does not select a valid expression node"))
end

function _replace_expression_at_path(expr::EquationExpr, path::Tuple{Vararg{Any}}, replacement)
    isempty(path) && return replacement(expr)
    step = first(path)
    remaining = Tuple(path[2:end])
    if expr isa EEq || expr isa ELe || expr isa EGe
        if step === :lhs
            lhs = _replace_expression_at_path(expr.lhs, remaining, replacement)
            return expr isa EEq ? EEq(lhs, expr.rhs) :
                expr isa ELe ? ELe(lhs, expr.rhs) : EGe(lhs, expr.rhs)
        elseif step === :rhs
            rhs = _replace_expression_at_path(expr.rhs, remaining, replacement)
            return expr isa EEq ? EEq(expr.lhs, rhs) :
                expr isa ELe ? ELe(expr.lhs, rhs) : EGe(expr.lhs, rhs)
        end
    elseif expr isa EAdd
        step === :term || throw(ArgumentError("expected `:term` in additive expression path"))
        isempty(remaining) && throw(ArgumentError("additive expression path is missing a term position"))
        position = first(remaining)
        position isa Integer || throw(ArgumentError(
            "additive expression path term position must be an integer"))
        1 <= position <= length(expr.terms) || throw(ArgumentError(
            "additive expression path term position $(position) is out of bounds"))
        terms = copy(expr.terms)
        terms[position] = _replace_expression_at_path(terms[position],
            Tuple(remaining[2:end]), replacement)
        return EAdd(terms)
    elseif expr isa EMul
        step === :factor || throw(ArgumentError("expected `:factor` in multiplicative expression path"))
        isempty(remaining) && throw(ArgumentError(
            "multiplicative expression path is missing a factor position"))
        position = first(remaining)
        position isa Integer || throw(ArgumentError(
            "multiplicative expression path factor position must be an integer"))
        1 <= position <= length(expr.factors) || throw(ArgumentError(
            "multiplicative expression path factor position $(position) is out of bounds"))
        factors = copy(expr.factors)
        factors[position] = _replace_expression_at_path(factors[position],
            Tuple(remaining[2:end]), replacement)
        return EMul(factors)
    elseif expr isa EPow
        step === :base && return EPow(_replace_expression_at_path(expr.base, remaining, replacement),
            expr.exponent)
        step === :exponent && return EPow(expr.base,
            _replace_expression_at_path(expr.exponent, remaining, replacement))
    elseif expr isa EDiv
        step === :numerator && return EDiv(
            _replace_expression_at_path(expr.numerator, remaining, replacement), expr.denominator)
        step === :denominator && return EDiv(expr.numerator,
            _replace_expression_at_path(expr.denominator, remaining, replacement))
    elseif expr isa ENeg
        step === :expression && return ENeg(_replace_expression_at_path(expr.expr, remaining, replacement))
    elseif expr isa ELog
        step === :expression && return ELog(_replace_expression_at_path(expr.expr, remaining, replacement))
    elseif expr isa ESum
        step === :expression && return ESum(expr.index, expr.domain,
            _replace_expression_at_path(expr.expr, remaining, replacement))
    elseif expr isa EProd
        step === :expression && return EProd(expr.index, expr.domain,
            _replace_expression_at_path(expr.expr, remaining, replacement))
    end
    throw(ArgumentError("expression path $(path) does not select a valid expression node"))
end

function _additive_sum_reference(expr::EquationExpr)
    expr isa EVar || expr isa EParam || return nothing
    isnothing(expr.idxs) && return nothing
    kind = expr isa EVar ? :variable : :parameter
    return (kind, expr.name, Tuple(expr.idxs))
end

function _validate_additive_sum_mapping(expr::EquationExpr, mapping::AdditiveSumMapping)
    target = _expression_at_path(expr, mapping.path)
    target isa EAdd || throw(ArgumentError(
        "additive-sum mapping at $(mapping.path) must select an EAdd expression"))
    length(target.terms) == length(mapping.domain) || throw(ArgumentError(
        "additive-sum mapping at $(mapping.path) expects $(length(mapping.domain)) terms, " *
        "but found $(length(target.terms))"))
    observed = Symbol[]
    for term in target.terms
        reference = _additive_sum_reference(term)
        isnothing(reference) && throw(ArgumentError(
            "additive-sum mapping at $(mapping.path) requires direct variable or parameter terms"))
        reference[1] == mapping.term_kind && reference[2] == mapping.term_name || throw(ArgumentError(
            "additive-sum mapping at $(mapping.path) expected $(mapping.term_kind) " *
            "$(mapping.term_name) terms"))
        length(reference[3]) >= mapping.index_position || throw(ArgumentError(
            "additive-sum mapping index position $(mapping.index_position) exceeds " *
            "the arity of $(mapping.term_name)"))
        value = reference[3][mapping.index_position]
        value isa Symbol || throw(ArgumentError(
            "additive-sum mapping at $(mapping.path) requires symbolic domain values"))
        push!(observed, value)
    end
    Set(observed) == Set(mapping.domain) && length(unique(observed)) == length(observed) ||
        throw(ArgumentError(
            "additive-sum mapping at $(mapping.path) does not cover its declared domain exactly"))
    return nothing
end

function _aggregate_additive_sum(expr::EquationExpr, mapping::AdditiveSumMapping)
    target = _expression_at_path(expr, mapping.path)
    target isa EAdd || throw(ArgumentError(
        "additive-sum mapping at $(mapping.path) must select an EAdd expression"))
    prototype = first(target.terms)
    reference = _additive_sum_reference(prototype)
    isnothing(reference) && throw(ArgumentError(
        "additive-sum mapping at $(mapping.path) requires direct variable or parameter terms"))
    for term in target.terms
        candidate = _additive_sum_reference(term)
        !isnothing(candidate) && candidate[1:2] == reference[1:2] || throw(ArgumentError(
            "additive-sum mapping at $(mapping.path) does not have homogeneous terms"))
        length(candidate[3]) == length(reference[3]) || throw(ArgumentError(
            "additive-sum mapping at $(mapping.path) does not have consistent term arity"))
        for position in eachindex(reference[3])
            position == mapping.index_position && continue
            candidate[3][position] == reference[3][position] || throw(ArgumentError(
                "additive-sum mapping at $(mapping.path) changes a non-summation index"))
        end
    end
    indices = Any[reference[3]...]
    indices[mapping.index_position] = EIndex(mapping.index)
    inner = mapping.term_kind === :variable ? EVar(mapping.term_name, indices) :
        EParam(mapping.term_name, indices)
    return _replace_expression_at_path(expr, mapping.path,
        _ -> ESum(mapping.index, mapping.domain, inner))
end

function _apply_additive_sum_mappings(payload::NamedTuple,
    mappings::Vector{AdditiveSumMapping})
    isempty(mappings) && return payload
    expression = get(payload, :expr, nothing)
    expression isa EquationExpr || throw(ArgumentError(
        "additive-sum mappings require a registered equation expression"))
    for mapping in mappings
        _validate_additive_sum_mapping(expression, mapping)
    end
    return merge(payload, (expr=foldl(_aggregate_additive_sum, mappings;
        init=expression),))
end

function _apply_report_mappings(eqs, mappings; strict::Bool)
    isempty(mappings) && return eqs
    lookup = _report_mapping_lookup(mappings)
    selected = Dict(selector => 0 for selector in keys(lookup))
    used = Dict(selector => Set{Tuple}() for selector in keys(lookup))
    used_domains = Dict(selector => Set{Symbol}() for selector in keys(lookup))
    used_index_projections = Dict(selector => Set{Symbol}() for selector in keys(lookup))
    used_references = Dict(selector => Set{Tuple{Symbol,Symbol,Tuple}}() for selector in keys(lookup))
    remapped = NamedTuple[]
    for eq in eqs
        selector = (eq.block, eq.tag)
        mapping = get(lookup, selector, nothing)
        if isnothing(mapping)
            push!(remapped, eq)
            continue
        end
        selected[selector] += 1
        source_indices = _registered_report_indices(eq)
        if !haskey(mapping.coordinates, source_indices)
            strict && throw(ArgumentError(
                "missing report coordinate for $(eq.block).$(eq.tag)$(source_indices)"))
            push!(remapped, eq)
            continue
        end
        push!(used[selector], source_indices)
        payload = eq.payload
        payload isa NamedTuple || throw(ArgumentError(
            "cannot map $(eq.block).$(eq.tag): its payload has no registered indices"))
        mapped_payload = _map_payload_report_values(payload, mapping.domain_values,
            mapping.index_projections, mapping.reference_indices, used_domains[selector],
            used_index_projections[selector], used_references[selector])
        mapped_payload = _apply_additive_sum_mappings(mapped_payload,
            mapping.additive_sums)
        mapped_payload = merge(mapped_payload, (index_names=mapping.index_names,
            indices=mapping.coordinates[source_indices]))
        push!(remapped, merge(eq, (block=mapping.report_block, tag=mapping.report_tag,
            payload=mapped_payload)))
    end
    for (selector, mapping) in lookup
        selected[selector] > 0 || throw(ArgumentError(
            "report mapping selects no registered equations: " *
            "$(mapping.source_block).$(mapping.source_tag)"))
        unused = setdiff(Set(keys(mapping.coordinates)), used[selector])
        (!strict || isempty(unused)) || throw(ArgumentError(
            "report mapping has coordinates with no registered equation: $(collect(unused))"))
        unused_domains = setdiff(Set(keys(mapping.domain_values)), used_domains[selector])
        (!strict || isempty(unused_domains)) || throw(ArgumentError(
            "report mapping has domain values not used by its selected equations: " *
            "$(collect(unused_domains))"))
        unused_index_projections = setdiff(Set(keys(mapping.index_projections)),
            used_index_projections[selector])
        (!strict || isempty(unused_index_projections)) || throw(ArgumentError(
            "report mapping has index projections not used by its selected equations: " *
            "$(collect(unused_index_projections))"))
        unused_references = setdiff(Set(keys(mapping.reference_indices)),
            used_references[selector])
        (!strict || isempty(unused_references)) || throw(ArgumentError(
            "report mapping has reference indices not used by its selected equations: " *
            "$(collect(unused_references))"))
    end
    return remapped
end

function _map_payload_report_values(payload::NamedTuple, domain_values::Dict{Symbol,Symbol},
    index_projections::Dict{Symbol,Tuple{Vararg{Symbol}}},
    reference_indices::Dict{Tuple{Symbol,Symbol,Tuple},Tuple}, used_domains::Set{Symbol},
    used_index_projections::Set{Symbol}, used_references::Set{Tuple{Symbol,Symbol,Tuple}})
    isempty(domain_values) && isempty(index_projections) && isempty(reference_indices) && return payload
    replacements = Pair{Symbol,Any}[]
    for name in (:expr, :objective_expr)
        expression = get(payload, name, nothing)
        expression isa EquationExpr || continue
        push!(replacements, name => _map_expression_report_values(expression, domain_values,
            index_projections, reference_indices, used_domains, used_index_projections,
            used_references))
    end
    isempty(replacements) && return payload
    return merge(payload, NamedTuple(replacements))
end

function _map_expression_report_values(expr::EquationExpr, domain_values::Dict{Symbol,Symbol},
    index_projections::Dict{Symbol,Tuple{Vararg{Symbol}}},
    reference_indices::Dict{Tuple{Symbol,Symbol,Tuple},Tuple}, used_domains::Set{Symbol},
    used_index_projections::Set{Symbol}, used_references::Set{Tuple{Symbol,Symbol,Tuple}})
    if expr isa EIndex
        positions = get(index_projections, expr.name, nothing)
        isnothing(positions) && return expr
        push!(used_index_projections, expr.name)
        return _IndexProjection(positions)
    elseif expr isa EVar || expr isa EParam
        isnothing(expr.idxs) && return expr
        kind = expr isa EVar ? :variable : :parameter
        key = (kind, expr.name, Tuple(expr.idxs))
        positions = get(reference_indices, key, nothing)
        if isnothing(positions)
            indices = Any[index isa EquationExpr ?
                _map_expression_report_values(index, domain_values, index_projections,
                    reference_indices, used_domains, used_index_projections, used_references) :
                index for index in expr.idxs]
            return expr isa EVar ? EVar(expr.name, indices) : EParam(expr.name, indices)
        end
        push!(used_references, key)
        indices = Any[_IndexProjection(position) for position in positions]
        return expr isa EVar ? EVar(expr.name, indices) : EParam(expr.name, indices)
    elseif expr isa EAdd
        return EAdd([_map_expression_report_values(term, domain_values, index_projections,
            reference_indices, used_domains, used_index_projections, used_references)
            for term in expr.terms])
    elseif expr isa EMul
        return EMul([_map_expression_report_values(factor, domain_values, index_projections,
            reference_indices, used_domains, used_index_projections, used_references)
            for factor in expr.factors])
    elseif expr isa EPow
        return EPow(_map_expression_report_values(expr.base, domain_values, index_projections,
                reference_indices, used_domains, used_index_projections, used_references),
            _map_expression_report_values(expr.exponent, domain_values, index_projections,
                reference_indices, used_domains, used_index_projections, used_references))
    elseif expr isa EDiv
        return EDiv(_map_expression_report_values(expr.numerator, domain_values, index_projections,
                reference_indices, used_domains, used_index_projections, used_references),
            _map_expression_report_values(expr.denominator, domain_values, index_projections,
                reference_indices, used_domains, used_index_projections, used_references))
    elseif expr isa ENeg
        return ENeg(_map_expression_report_values(expr.expr, domain_values, index_projections,
            reference_indices, used_domains, used_index_projections, used_references))
    elseif expr isa ELog
        return ELog(_map_expression_report_values(expr.expr, domain_values, index_projections,
            reference_indices, used_domains, used_index_projections, used_references))
    elseif expr isa ESum || expr isa EProd
        domain = Symbol[]
        for value in expr.domain
            mapped_value = get(domain_values, value, value)
            haskey(domain_values, value) && push!(used_domains, value)
            push!(domain, mapped_value)
        end
        inner = _map_expression_report_values(expr.expr, domain_values, index_projections,
            reference_indices, used_domains, used_index_projections, used_references)
        return expr isa ESum ? ESum(expr.index, domain, inner) : EProd(expr.index, domain, inner)
    elseif expr isa EEq
        return EEq(_map_expression_report_values(expr.lhs, domain_values, index_projections,
                reference_indices, used_domains, used_index_projections, used_references),
            _map_expression_report_values(expr.rhs, domain_values, index_projections,
                reference_indices, used_domains, used_index_projections, used_references))
    elseif expr isa ELe
        return ELe(_map_expression_report_values(expr.lhs, domain_values, index_projections,
                reference_indices, used_domains, used_index_projections, used_references),
            _map_expression_report_values(expr.rhs, domain_values, index_projections,
                reference_indices, used_domains, used_index_projections, used_references))
    elseif expr isa EGe
        return EGe(_map_expression_report_values(expr.lhs, domain_values, index_projections,
                reference_indices, used_domains, used_index_projections, used_references),
            _map_expression_report_values(expr.rhs, domain_values, index_projections,
                reference_indices, used_domains, used_index_projections, used_references))
    end
    return expr
end

function _template_index_data(eq)
    payload = eq.payload
    payload isa NamedTuple || return (), ()
    raw_names = get(payload, :index_names, nothing)
    raw_indices = get(payload, :indices, ())
    raw_names === nothing && return (), ()
    names = Tuple(Symbol.(collect(raw_names)))
    indices = Tuple(raw_indices)
    length(names) == length(indices) || return (), ()
    return names, indices
end

function _expression_skeleton(expr::EquationExpr)
    if expr isa EVar
        return (:var, expr.name, isnothing(expr.idxs) ? 0 : length(expr.idxs))
    elseif expr isa EParam
        return (:param, expr.name, isnothing(expr.idxs) ? 0 : length(expr.idxs))
    elseif expr isa EConst
        return :constant
    elseif expr isa ERaw
        return (:raw, expr.text)
    elseif expr isa EIndex
        return (:index, expr.name)
    elseif expr isa EAdd
        return (:add, map(_expression_skeleton, expr.terms))
    elseif expr isa EMul
        return (:mul, map(_expression_skeleton, expr.factors))
    elseif expr isa EPow
        return (:pow, _expression_skeleton(expr.base), _expression_skeleton(expr.exponent))
    elseif expr isa EDiv
        return (:div, _expression_skeleton(expr.numerator), _expression_skeleton(expr.denominator))
    elseif expr isa ENeg
        return (:neg, _expression_skeleton(expr.expr))
    elseif expr isa ELog
        return (:log, _expression_skeleton(expr.expr))
    elseif expr isa ESum
        return (:sum, expr.index, _expression_skeleton(expr.expr))
    elseif expr isa EProd
        return (:prod, expr.index, _expression_skeleton(expr.expr))
    elseif expr isa EEq
        return (:eq, _expression_skeleton(expr.lhs), _expression_skeleton(expr.rhs))
    elseif expr isa ELe
        return (:le, _expression_skeleton(expr.lhs), _expression_skeleton(expr.rhs))
    elseif expr isa EGe
        return (:ge, _expression_skeleton(expr.lhs), _expression_skeleton(expr.rhs))
    end
    return repr(expr)
end

function _template_seed_key(eq)
    expression, kind, objective_sense = _equation_expression(eq)
    expression === nothing && return _equation_family_key(eq)
    names, _ = _template_index_data(eq)
    return (eq.block, eq.tag, kind, objective_sense, _equation_condition_role(eq),
        names, _expression_skeleton(expression))
end

_template_path_key(path::Vector{Any}) = join(string.(path), "\u001f")

function _collect_reference_values!(out::Dict{String,Vector{Any}}, expr::EquationExpr,
    path::Vector{Any}=Any[])
    if expr isa EVar || expr isa EParam
        if !isnothing(expr.idxs)
            for (position, value) in enumerate(expr.idxs)
                push!(path, :index, position)
                push!(get!(out, _template_path_key(path), Any[]), value)
                pop!(path); pop!(path)
            end
        end
    elseif expr isa EAdd || expr isa EMul
        for (position, term) in enumerate(expr isa EAdd ? expr.terms : expr.factors)
            push!(path, :term, position)
            _collect_reference_values!(out, term, path)
            pop!(path); pop!(path)
        end
    elseif expr isa EPow
        push!(path, :base); _collect_reference_values!(out, expr.base, path); pop!(path)
        push!(path, :exponent); _collect_reference_values!(out, expr.exponent, path); pop!(path)
    elseif expr isa EDiv
        push!(path, :numerator); _collect_reference_values!(out, expr.numerator, path); pop!(path)
        push!(path, :denominator); _collect_reference_values!(out, expr.denominator, path); pop!(path)
    elseif expr isa ENeg || expr isa ELog || expr isa ESum || expr isa EProd
        push!(path, :expression); _collect_reference_values!(out, expr.expr, path); pop!(path)
    elseif expr isa EEq || expr isa ELe || expr isa EGe
        push!(path, :lhs); _collect_reference_values!(out, expr.lhs, path); pop!(path)
        push!(path, :rhs); _collect_reference_values!(out, expr.rhs, path); pop!(path)
    end
    return out
end

function _collect_constant_values!(out::Dict{String,Vector{Float64}}, expr::EquationExpr,
    path::Vector{Any}=Any[])
    if expr isa EConst
        push!(get!(out, _template_path_key(path), Float64[]), Float64(expr.value))
    elseif expr isa EAdd || expr isa EMul
        for (position, term) in enumerate(expr isa EAdd ? expr.terms : expr.factors)
            push!(path, :term, position)
            _collect_constant_values!(out, term, path)
            pop!(path); pop!(path)
        end
    elseif expr isa EPow
        push!(path, :base); _collect_constant_values!(out, expr.base, path); pop!(path)
        push!(path, :exponent); _collect_constant_values!(out, expr.exponent, path); pop!(path)
    elseif expr isa EDiv
        push!(path, :numerator); _collect_constant_values!(out, expr.numerator, path); pop!(path)
        push!(path, :denominator); _collect_constant_values!(out, expr.denominator, path); pop!(path)
    elseif expr isa ENeg || expr isa ELog || expr isa ESum || expr isa EProd
        push!(path, :expression); _collect_constant_values!(out, expr.expr, path); pop!(path)
    elseif expr isa EEq || expr isa ELe || expr isa EGe
        push!(path, :lhs); _collect_constant_values!(out, expr.lhs, path); pop!(path)
        push!(path, :rhs); _collect_constant_values!(out, expr.rhs, path); pop!(path)
    end
    return out
end

function _is_function_of(values::Vector{Any}, outer_values::Vector{Tuple}, positions::Vector{Int})
    lookup = Dict{Tuple,Any}()
    for (value, indices) in zip(values, outer_values)
        key = Tuple(indices[position] for position in positions)
        if haskey(lookup, key) && lookup[key] != value
            return false
        end
        lookup[key] = value
    end
    return true
end

function _index_projection(values::Vector{Any}, index_names::Tuple{Vararg{Symbol}},
    outer_values::Vector{Tuple})
    isempty(values) && return nothing
    all(value -> value == first(values), values) && return nothing
    isempty(index_names) && return nothing
    candidates = Tuple{Vararg{Symbol}}[]
    count = length(index_names)
    for width in 1:count
        for mask in 1:(1 << count) - 1
            positions = [position for position in 1:count if (mask & (1 << (position - 1))) != 0]
            length(positions) == width || continue
            _is_function_of(values, outer_values, positions) || continue
            push!(candidates, Tuple(index_names[position] for position in positions))
        end
        !isempty(candidates) && break
    end
    length(candidates) == 1 || return nothing
    return _IndexProjection(only(candidates))
end

function _reference_index_maps(expressions::Vector{<:EquationExpr},
    index_names::Tuple{Vararg{Symbol}}, outer_values::Vector{Tuple})
    values = Dict{String,Vector{Any}}()
    for expression in expressions
        _collect_reference_values!(values, expression)
    end
    mappings = Dict{String,_IndexProjection}()
    for (path, observed) in values
        projection = _index_projection(observed, index_names, outer_values)
        !isnothing(projection) && (mappings[path] = projection)
    end
    return mappings
end

function _parameter_entries(eq)
    payload = eq.payload
    payload isa NamedTuple || return NamedTuple[]
    params = get(payload, :params, nothing)
    params isa NamedTuple || return NamedTuple[]
    entries = NamedTuple[]
    for name in propertynames(params)
        parameter = getproperty(params, name)
        if parameter isa Real
            push!(entries, (name=Symbol(name), key=(), value=Float64(parameter)))
        elseif parameter isa AbstractDict
            for (key, value) in parameter
                value isa Real || continue
                key_tuple = key isa Tuple ? Tuple(key) : (key,)
                push!(entries, (name=Symbol(name), key=key_tuple, value=Float64(value)))
            end
        end
    end
    return entries
end

_same_number(left::Real, right::Real) = left == right || isapprox(left, right; rtol=1.0e-12, atol=0.0)

function _parameter_specification(values::Vector{Float64}, instances::Vector{NamedTuple},
    index_names::Tuple{Vararg{Symbol}}, outer_values::Vector{Tuple})
    isempty(values) && return nothing
    candidate_names = Set(entry.name for entry in _parameter_entries(first(instances)))
    for instance in instances[2:end]
        intersect!(candidate_names, Set(entry.name for entry in _parameter_entries(instance)))
    end
    specifications = NamedTuple[]
    for name in sort!(collect(candidate_names); by=string)
        matches_per_instance = Vector{NamedTuple}()
        valid = true
        for (instance, value) in zip(instances, values)
            matches = [entry for entry in _parameter_entries(instance) if entry.name == name &&
                _same_number(entry.value, value)]
            if length(matches) != 1
                valid = false
                break
            end
            push!(matches_per_instance, only(matches))
        end
        valid || continue
        key_lengths = unique(length(entry.key) for entry in matches_per_instance)
        length(key_lengths) == 1 || continue
        indices = Any[]
        for position in 1:only(key_lengths)
            observed = Any[entry.key[position] for entry in matches_per_instance]
            projection = _index_projection(observed, index_names, outer_values)
            if isnothing(projection)
                all(value -> value == first(observed), observed) || (valid = false; break)
                push!(indices, first(observed))
            else
                push!(indices, projection)
            end
        end
        valid && push!(specifications, (name=name, indices=indices))
    end
    length(specifications) == 1 || return nothing
    return only(specifications)
end

function _generated_coefficient_specification(values::Vector{Float64},
    index_names::Tuple{Vararg{Symbol}}, outer_values::Vector{Tuple}, ordinal::Int)
    projection = _index_projection(Any[values...], index_names, outer_values)
    indices = isnothing(projection) ? Any[] : Any[projection]
    return (name=Symbol("calibration_coefficient_", ordinal), indices=indices)
end

function _constant_parameter_maps(expressions::Vector{<:EquationExpr},
    instances::Vector{NamedTuple}, index_names::Tuple{Vararg{Symbol}},
    outer_values::Vector{Tuple})
    values = Dict{String,Vector{Float64}}()
    for expression in expressions
        _collect_constant_values!(values, expression)
    end
    mappings = Dict{String,NamedTuple}()
    ordinal = 0
    for path in sort!(collect(keys(values)))
        observed = values[path]
        all(value -> value in (-1.0, 0.0, 1.0), observed) && continue
        ordinal += 1
        specification = _parameter_specification(observed, instances, index_names, outer_values)
        mappings[path] = isnothing(specification) ?
            _generated_coefficient_specification(observed, index_names, outer_values, ordinal) :
            specification
    end
    return mappings
end

function _template_expression(expr::EquationExpr, reference_maps, constant_maps,
    path::Vector{Any}=Any[])
    if expr isa EVar || expr isa EParam
        indices = nothing
        if !isnothing(expr.idxs)
            indices = Any[]
            for (position, value) in enumerate(expr.idxs)
                push!(path, :index, position)
                push!(indices, get(reference_maps, _template_path_key(path), value))
                pop!(path); pop!(path)
            end
        end
        return expr isa EVar ? EVar(expr.name, indices) : EParam(expr.name, indices)
    elseif expr isa EConst
        specification = get(constant_maps, _template_path_key(path), nothing)
        return isnothing(specification) ? expr : EParam(specification.name, specification.indices)
    elseif expr isa EAdd
        terms = EquationExpr[]
        for (position, term) in enumerate(expr.terms)
            push!(path, :term, position)
            push!(terms, _template_expression(term, reference_maps, constant_maps, path))
            pop!(path); pop!(path)
        end
        return EAdd(terms)
    elseif expr isa EMul
        factors = EquationExpr[]
        for (position, term) in enumerate(expr.factors)
            push!(path, :term, position)
            push!(factors, _template_expression(term, reference_maps, constant_maps, path))
            pop!(path); pop!(path)
        end
        return EMul(factors)
    elseif expr isa EPow
        push!(path, :base)
        base = _template_expression(expr.base, reference_maps, constant_maps, path)
        pop!(path)
        push!(path, :exponent)
        exponent = _template_expression(expr.exponent, reference_maps, constant_maps, path)
        pop!(path)
        return EPow(base, exponent)
    elseif expr isa EDiv
        push!(path, :numerator)
        numerator = _template_expression(expr.numerator, reference_maps, constant_maps, path)
        pop!(path)
        push!(path, :denominator)
        denominator = _template_expression(expr.denominator, reference_maps, constant_maps, path)
        pop!(path)
        return EDiv(numerator, denominator)
    elseif expr isa ENeg
        push!(path, :expression)
        nested = _template_expression(expr.expr, reference_maps, constant_maps, path)
        pop!(path)
        return ENeg(nested)
    elseif expr isa ELog
        push!(path, :expression)
        nested = _template_expression(expr.expr, reference_maps, constant_maps, path)
        pop!(path)
        return ELog(nested)
    elseif expr isa ESum
        push!(path, :expression)
        nested = _template_expression(expr.expr, reference_maps, constant_maps, path)
        pop!(path)
        return ESum(expr.index, expr.domain, nested)
    elseif expr isa EProd
        push!(path, :expression)
        nested = _template_expression(expr.expr, reference_maps, constant_maps, path)
        pop!(path)
        return EProd(expr.index, expr.domain, nested)
    elseif expr isa EEq
        push!(path, :lhs)
        lhs = _template_expression(expr.lhs, reference_maps, constant_maps, path)
        pop!(path)
        push!(path, :rhs)
        rhs = _template_expression(expr.rhs, reference_maps, constant_maps, path)
        pop!(path)
        return EEq(lhs, rhs)
    elseif expr isa ELe
        push!(path, :lhs)
        lhs = _template_expression(expr.lhs, reference_maps, constant_maps, path)
        pop!(path)
        push!(path, :rhs)
        rhs = _template_expression(expr.rhs, reference_maps, constant_maps, path)
        pop!(path)
        return ELe(lhs, rhs)
    elseif expr isa EGe
        push!(path, :lhs)
        lhs = _template_expression(expr.lhs, reference_maps, constant_maps, path)
        pop!(path)
        push!(path, :rhs)
        rhs = _template_expression(expr.rhs, reference_maps, constant_maps, path)
        pop!(path)
        return EGe(lhs, rhs)
    end
    return expr
end

function _append_equation_templates!(templates::Vector{EquationTemplate},
    instances::Vector{NamedTuple})
    representative = first(instances)
    expression, kind, objective_sense = _equation_expression(representative)
    index_names, _ = _template_index_data(representative)
    if isnothing(expression)
        push!(templates, EquationTemplate(representative.block, representative.tag, kind,
            nothing, objective_sense, _equation_condition_role(representative), instances,
            Pair{String,Vector{String}}[], index_names))
        return templates
    end
    expressions = EquationExpr[_equation_expression(instance)[1] for instance in instances]
    outer_values = Tuple[_template_index_data(instance)[2] for instance in instances]
    reference_maps = _reference_index_maps(expressions, index_names, outer_values)
    constant_maps = _constant_parameter_maps(expressions, instances, index_names, outer_values)
    grouped = Dict{String,Vector{NamedTuple}}()
    rendered_expressions = Dict{String,EquationExpr}()
    for (instance, instance_expression) in zip(instances, expressions)
        template_expression = _template_expression(instance_expression, reference_maps, constant_maps)
        key = repr(template_expression)
        push!(get!(grouped, key, NamedTuple[]), instance)
        rendered_expressions[key] = template_expression
    end
    for key in sort!(collect(keys(grouped)))
        grouped_instances = grouped[key]
        template_expression = rendered_expressions[key]
        push!(templates, EquationTemplate(representative.block, representative.tag, kind,
            template_expression, objective_sense, _equation_condition_role(representative),
            grouped_instances, _collect_domains(template_expression), index_names))
    end
    return templates
end

function _equation_templates(eqs)
    seed_groups = Dict{Tuple,Vector{NamedTuple}}()
    seed_order = Tuple[]
    for eq in eqs
        key = _template_seed_key(eq)
        if !haskey(seed_groups, key)
            seed_groups[key] = NamedTuple[]
            push!(seed_order, key)
        end
        push!(seed_groups[key], eq)
    end
    templates = EquationTemplate[]
    for key in seed_order
        _append_equation_templates!(templates, seed_groups[key])
    end
    return templates
end

function _render_equation_families(families; format::Symbol, level::Symbol,
    show_defs::Bool, show_condition_roles::Bool)
    if level != :block && level != :equation
        error("Unsupported level: $(level). Use :block or :equation")
    end
    _validate_equation_format(format)
    lines = String[]
    if format == :markdown
        push!(lines, "# Equation families")
    elseif format == :latex
        push!(lines, "% Equation families")
    else
        push!(lines, "EQUATION FAMILIES")
    end
    isempty(families) && return join([lines; _format_text(format, "No equations registered.")], "\n")
    if level == :equation
        for family in families
            append!(lines, _render_equation_family(family; format=format, show_defs=show_defs,
                show_condition_roles=show_condition_roles))
        end
    else
        by_block = Dict{Symbol,Vector{EquationFamily}}()
        for family in families
            push!(get!(by_block, family.block, EquationFamily[]), family)
        end
        for (block, block_families) in sort(collect(by_block); by=first)
            append!(lines, _render_family_block_section(block, block_families; format=format,
                show_defs=show_defs, show_condition_roles=show_condition_roles))
        end
    end
    return join(lines, "\n")
end

function _render_equation_report_families(families; format::Symbol, show_defs::Bool,
    show_condition_roles::Bool)
    _validate_equation_format(format)
    objectives = filter(family -> family.kind == :objective, families)
    equations = filter(family -> family.kind != :objective, families)
    lines = _report_heading(format, "Model equation report", 1)
    if isempty(families)
        push!(lines, _format_text(format, "No equations registered."))
        return join(lines, "\n")
    end
    if !isempty(objectives)
        append!(lines, _report_heading(format, "Objective functions", 2))
        append!(lines, _render_report_families(objectives; format=format, show_defs=show_defs,
            show_condition_roles=show_condition_roles))
    end
    if !isempty(equations)
        append!(lines, _report_heading(format, "Equations", 2))
        append!(lines, _render_report_families(equations; format=format, show_defs=show_defs,
            show_condition_roles=show_condition_roles))
    end
    return join(lines, "\n")
end

function _render_equation_report_expanded(eqs; format::Symbol, show_defs::Bool,
    show_condition_roles::Bool)
    _validate_equation_format(format)
    objectives = filter(eq -> _equation_expression(eq)[2] == :objective, eqs)
    equations = filter(eq -> _equation_expression(eq)[2] != :objective, eqs)
    lines = _report_heading(format, "Model equation report", 1)
    if isempty(eqs)
        push!(lines, _format_text(format, "No equations registered."))
        return join(lines, "\n")
    end
    if !isempty(objectives)
        append!(lines, _report_heading(format, "Objective functions", 2))
        append!(lines, _render_report_equations(objectives; format=format, show_defs=show_defs,
            show_condition_roles=show_condition_roles))
    end
    if !isempty(equations)
        append!(lines, _report_heading(format, "Equations", 2))
        append!(lines, _render_report_equations(equations; format=format, show_defs=show_defs,
            show_condition_roles=show_condition_roles))
    end
    return join(lines, "\n")
end

function _report_heading(format::Symbol, text::AbstractString, level::Int)
    if format == :markdown
        return [repeat("#", level) * " " * text]
    elseif format == :latex
        command = level == 1 ? "section" : level == 2 ? "subsection" : "subsubsection"
        return ["\\$(command)*{$(_latex_escape(text))}"]
    end
    return [uppercase(text)]
end

function _render_report_families(families; format::Symbol, show_defs::Bool,
    show_condition_roles::Bool)
    lines = String[]
    by_block = Dict{Symbol,Vector{Any}}()
    for family in families
        push!(get!(by_block, family.block, Any[]), family)
    end
    for (block, block_families) in sort(collect(by_block); by=first)
        append!(lines, _render_family_block_section(block, block_families; format=format,
            show_defs=show_defs, show_condition_roles=show_condition_roles, report=true))
    end
    return lines
end

function _render_report_equations(eqs; format::Symbol, show_defs::Bool,
    show_condition_roles::Bool)
    lines = String[]
    by_block = Dict{Symbol,Vector{NamedTuple}}()
    for eq in eqs
        push!(get!(by_block, eq.block, NamedTuple[]), eq)
    end
    for (block, block_eqs) in sort(collect(by_block); by=first)
        if format == :markdown
            push!(lines, "### Block: $(block)")
        elseif format == :latex
            push!(lines, "\\subsubsection*{Block: $(_latex_escape(string(block)))}")
        else
            push!(lines, "BLOCK: $(block)")
        end
        for eq in block_eqs
            append!(lines, _render_report_equation(eq; format=format, show_defs=show_defs,
                show_condition_roles=show_condition_roles))
        end
    end
    return lines
end

function _render_family_block_section(block, families; format::Symbol, show_defs::Bool,
    show_condition_roles::Bool, report::Bool=false)
    lines = String[]
    if format == :markdown
        push!(lines, report ? "### Block: $(block)" : "## Block: $(block)")
    elseif format == :latex
        command = report ? "subsubsection" : "paragraph"
        push!(lines, "\\$(command)*{Block: $(_latex_escape(string(block)))}")
    else
        push!(lines, "BLOCK: $(block)")
    end
    for family in families
        append!(lines, _render_equation_family(family; format=format, show_defs=show_defs,
            show_condition_roles=show_condition_roles))
    end
    return lines
end

function _render_equation_family(family; format::Symbol, show_defs::Bool,
    show_condition_roles::Bool, label_override::Union{Nothing,String}=nothing)
    lines = String[]
    label = show_defs ? something(label_override,
        _family_label(family; show_condition_role=show_condition_roles)) : ""
    instance_text = "$(length(family.instances)) registered " *
        (length(family.instances) == 1 ? "instance" : "instances")
    if family.expression === nothing
        info, _ = _equation_info(first(family.instances); format=format)
        if format == :latex
            !isempty(label) && push!(lines, "\\paragraph{$(_latex_escape(label))}")
            push!(lines, "\\noindent\\emph{Description only; no equation AST was registered.} " *
                _latex_escape(info) * "\\par")
        elseif format == :markdown
            !isempty(label) && push!(lines, "**$(label)**")
            push!(lines, "Description only; no equation AST was registered: $(info)")
        else
            push!(lines, isempty(label) ? "* $(info)" : "* $(label) $(info)")
        end
        return lines
    end
    rendered = family.kind == :objective ?
        _render_objective_expr(family.expression, family.objective_sense; format=format) :
        render_expr(family.expression; format=format)
    if format == :latex
        !isempty(label) && push!(lines, "\\paragraph{$(_latex_escape(label))}")
        push!(lines, "\\begin{align*}")
        push!(lines, _render_latex_alignment(family.expression, rendered; kind=family.kind))
        push!(lines, "\\end{align*}")
        push!(lines, "\\noindent\\emph{$(instance_text).}\\par")
        append!(lines, _render_family_domains(family.domains; format=:latex))
    elseif format == :markdown
        !isempty(label) && push!(lines, "**$(label)**")
        push!(lines, "\$\$\n$(rendered)\n\$\$")
        push!(lines, "_$(instance_text)._")
        append!(lines, _render_family_domains(family.domains; format=:markdown))
    else
        prefix = isempty(label) ? "*" : "* $(label)"
        push!(lines, "$(prefix) $(rendered) ($(instance_text))")
        append!(lines, _render_family_domains(family.domains; format=:plain))
    end
    return lines
end

function _render_report_equation(eq; format::Symbol, show_defs::Bool,
    show_condition_roles::Bool)
    expression, kind, sense = _equation_expression(eq)
    expression === nothing && return [_render_equation_line(eq; format=format,
        show_defs=show_defs, show_condition_roles=show_condition_roles)]
    family = EquationFamily(eq.block, eq.tag, kind, expression, sense,
        _equation_condition_role(eq), NamedTuple[eq], _collect_domains(expression))
    return _render_equation_family(family; format=format, show_defs=show_defs,
        show_condition_roles=show_condition_roles,
        label_override=_equation_label(eq; show_condition_role=show_condition_roles))
end

function _family_label(family; show_condition_role::Bool=false)
    label = string(family.block, ".", family.tag)
    if show_condition_role && family.condition_role !== nothing
        return string(label, " (", replace(string(family.condition_role), "_" => " "), ")")
    end
    return label
end

function _render_latex_alignment(expression::EquationExpr, rendered::AbstractString; kind::Symbol)
    if kind == :equation
        if expression isa EEq
            return string(render_expr(expression.lhs; format=:latex), " &= ",
                render_expr(expression.rhs; format=:latex))
        elseif expression isa ELe
            return string(render_expr(expression.lhs; format=:latex), " &\\le ",
                render_expr(expression.rhs; format=:latex))
        elseif expression isa EGe
            return string(render_expr(expression.lhs; format=:latex), " &\\ge ",
                render_expr(expression.rhs; format=:latex))
        end
    end
    return string("&\\quad ", rendered)
end

function _render_family_domains(domains; format::Symbol)
    isempty(domains) && return String[]
    unique_domains = unique(domains)
    if format == :latex
        terms = ["$(_latex_escape(index)) \\in \\{$(join(_latex_escape.(domain), ", "))\\}"
            for (index, domain) in unique_domains]
        return ["\\noindent\\emph{Internal sum/product domains:} \\( $(join(terms, "; ")) \\)\\par"]
    elseif format == :markdown
        terms = ["$(index) in { $(join(domain, ", ")) }" for (index, domain) in unique_domains]
        return ["Internal sum/product domains: " * join(terms, "; ")]
    end
    terms = ["$(index) in { $(join(domain, ", ")) }" for (index, domain) in unique_domains]
    return ["  domains: " * join(terms, "; ")]
end

function _validate_equation_format(format::Symbol)
    format in (:markdown, :latex, :plain) ||
        error("Unsupported format: $(format). Use :markdown, :latex, or :plain")
    return nothing
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
    _render_block_section(block, eqs; format, show_defs, show_condition_roles)

Render a block heading followed by its equations.
"""
function _render_block_section(block, eqs; format::Symbol, show_defs::Bool,
    show_condition_roles::Bool)
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
        push!(lines, _render_equation_line(eq; format=format, show_defs=show_defs,
            show_condition_roles=show_condition_roles))
    end
    return lines
end

"""
    _render_equation_line(eq; format, show_defs, show_condition_roles)

Render a single equation line with label and optional domain annotations.
"""
function _render_equation_line(eq; format::Symbol, show_defs::Bool,
    show_condition_roles::Bool)
    info, is_math = _equation_info(eq; format=format)
    domains = Pair{String,Vector{String}}[]
    if format == :markdown
        domains = _extract_domains(eq)
    end
    label = ""
    if show_defs
        label = _equation_label(eq; show_condition_role=show_condition_roles)
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
        if is_math
            label_comment = isempty(label) ? "" : "% $(label)\n"
            return string(label_comment, "\\[\n", info, "\n\\]")
        elseif isempty(label)
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
    elseif expr isa ENeg || expr isa ELog
        _collect_domains!(domains, expr.expr)
    elseif expr isa EEq || expr isa ELe || expr isa EGe
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
        objective_expr = get(payload, :objective_expr, nothing)
        if objective_expr !== nothing
            if objective_expr isa EquationExpr
                sense = get(payload, :objective_sense, :Max)
                return _render_objective_expr(objective_expr, sense; format=format), true
            end
            return string(objective_expr), false
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

function _render_objective_expr(expr::EquationExpr, sense; format::Symbol)
    rendered = render_expr(expr; format=format)
    if format == :latex || format == :markdown
        prefix = _objective_prefix_latex(sense)
        return string(prefix, rendered)
    end
    prefix = _objective_prefix_plain(sense)
    return string(prefix, rendered)
end

function _objective_prefix_latex(sense)
    return _objective_is_min(sense) ? "\\min\\;" : "\\max\\;"
end

function _objective_prefix_plain(sense)
    return _objective_is_min(sense) ? "min " : "max "
end

function _objective_is_min(sense)
    if sense isa Symbol
        return sense in (:Min, :min, :MIN, :Minimize, :minimize, :MINIMIZE)
    elseif sense isa AbstractString
        lowered = lowercase(sense)
        return lowered in ("min", "minimize")
    end
    return false
end

function _equation_label(eq; show_condition_role::Bool=false)
    payload = eq.payload
    idxs = ()
    if payload isa NamedTuple && haskey(payload, :indices)
        idxs = payload.indices
    end
    idx_text = _format_indices(idxs)
    label = isempty(idx_text) ?
        string(eq.block, ".", eq.tag) :
        string(eq.block, ".", eq.tag, "[", idx_text, "]")
    if show_condition_role && payload isa NamedTuple
        role = get(payload, :condition_role, :enforce)
        return string(label, " (", replace(string(role), "_" => " "), ")")
    end
    return label
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
        base = _wrap_power_base(expr.base, _render_expr(expr.base; format=format); format=format)
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
        exp = _wrap_power_exponent(expr.exponent, exp; format=format)
        return string(base, "^", exp)
    elseif expr isa EDiv
        num = _render_expr(expr.numerator; format=format)
        den = _render_expr(expr.denominator; format=format)
        if format == :latex
            return string("\\frac{", num, "}{", den, "}")
        end
        return string(num, " / ", den)
    elseif expr isa ENeg
        inner = _wrap_if_needed(expr.expr, _render_expr(expr.expr; format=format); format=format)
        return string("-", inner)
    elseif expr isa ELog
        inner = _render_expr(expr.expr; format=format)
        if format == :latex
            return string("\\log\\left(", inner, "\\right)")
        end
        return string("log(", inner, ")")
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
    elseif expr isa ELe
        lhs = _render_expr(expr.lhs; format=format)
        rhs = _render_expr(expr.rhs; format=format)
        op = format == :latex ? " \\le " : " <= "
        return string(lhs, op, rhs)
    elseif expr isa EGe
        lhs = _render_expr(expr.lhs; format=format)
        rhs = _render_expr(expr.rhs; format=format)
        op = format == :latex ? " \\ge " : " >= "
        return string(lhs, op, rhs)
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

Render an exponent as mathematics. This preserves generated calibration
coefficient symbols rather than converting them to escaped plain text.
"""
function _render_exponent_expr(expr::EquationExpr)
    return _render_expr(expr; format=:latex)
end

function _latex_escape_exponent(text::AbstractString)
    escaped = replace(text, "\\" => "\\textbackslash{}")
    escaped = replace(escaped, "_" => "\\_", "#" => "\\#", "%" => "\\%", "&" => "\\&", "\$" => "\\\$", "^" => "\\^{}", "~" => "\\~{}")
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

function _wrap_power_base(expr::EquationExpr, rendered::AbstractString; format::Symbol)
    needs_grouping = expr isa EAdd || expr isa EMul || expr isa EDiv ||
        expr isa ENeg || expr isa EPow || expr isa EEq || expr isa ELe || expr isa EGe
    !needs_grouping && return rendered
    if format == :latex
        return string("\\left(", rendered, "\\right)")
    end
    return string("(", rendered, ")")
end

function _wrap_power_exponent(expr::EquationExpr, rendered::AbstractString; format::Symbol)
    needs_grouping = expr isa EAdd || expr isa EMul || expr isa EDiv ||
        expr isa ENeg || expr isa EPow || expr isa EEq || expr isa ELe || expr isa EGe
    !needs_grouping && return rendered
    return string("(", rendered, ")")
end

"""
    _render_symbol(name, idxs; format)

Render a symbol name with optional indices.
"""
function _render_symbol(name::Symbol, idxs::Union{Nothing,Vector{Any}}; format::Symbol)
    coefficient = match(r"^calibration_coefficient_(\d+)$", string(name))
    if !isnothing(coefficient)
        identifier = only(coefficient.captures)
        if format == :latex
            if idxs === nothing || isempty(idxs)
                return string("\\kappa_{", identifier, "}")
            end
            idx_text = join(map(idx -> _render_index(idx; format=format), idxs), ",")
            return string("\\kappa_{", identifier, ",", idx_text, "}")
        end
        suffix = idxs === nothing || isempty(idxs) ? "" :
            "[" * join(map(idx -> _render_index(idx; format=format), idxs), ",") * "]"
        return string("κ", identifier, suffix)
    end
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
    if idx isa _IndexProjection
        labels = _latex_escape.(string.(idx.names))
        if length(labels) == 1
            return only(labels)
        elseif format == :latex
            return "\\langle " * join(labels, ", ") * "\\rangle"
        end
        return "(" * join(labels, ", ") * ")"
    end
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
