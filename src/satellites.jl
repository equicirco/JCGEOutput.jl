"""
Data-driven satellite reporting for quantities outside the monetary CGE core.

Satellite anchors convert a solved model-volume driver into a reported
quantity in a declared unit.  They are reporting objects: they do not add a
constraint or otherwise alter the equilibrium problem.  A satellite balance
can subsequently check a signed physical, environmental, or other quantity
identity over the projected anchors.
"""

struct SatelliteAnchor
    id::Symbol
    unit::String
    base_quantity::Float64
    driver::Symbol
    base_driver::Float64

    function SatelliteAnchor(id::Symbol, unit::AbstractString, base_quantity::Real,
        driver::Symbol, base_driver::Real)
        base_quantity >= 0.0 || error("Satellite anchor $(id) must have a non-negative base quantity.")
        base_driver > 0.0 || error("Satellite anchor $(id) must have a strictly positive base driver.")
        isempty(strip(unit)) && error("Satellite anchor $(id) must declare a unit.")
        new(id, String(unit), Float64(base_quantity), driver, Float64(base_driver))
    end
end

"""
Solved driver levels defining a named satellite reference point.

The usual reference is the solved zero-policy equilibrium.  It may differ
slightly from a rounded monetary calibration input, so using it as the
denominator preserves observed physical quantities exactly at the reported
baseline and gives every scenario a common comparison point.
"""
struct SatelliteReference
    id::Symbol
    drivers::Dict{Symbol,Float64}

    function SatelliteReference(id::Symbol, drivers::AbstractDict{Symbol,<:Real})
        isempty(string(id)) && error("A satellite reference must declare an identifier.")
        normalized = Dict{Symbol,Float64}(key => Float64(value) for (key, value) in drivers)
        all(value -> isfinite(value) && value > 0.0, values(normalized)) ||
            error("Satellite-reference drivers must be finite and strictly positive.")
        new(id, normalized)
    end
end

struct SatelliteBalance
    id::Symbol
    unit::String
    terms::Vector{Pair{Symbol,Float64}}

    function SatelliteBalance(id::Symbol, unit::AbstractString, terms::AbstractVector{<:Pair})
        isempty(terms) && error("Satellite balance $(id) must contain at least one signed term.")
        isempty(strip(unit)) && error("Satellite balance $(id) must declare a unit.")
        normalized = Pair{Symbol,Float64}[
            Symbol(first(term)) => Float64(last(term)) for term in terms
        ]
        all(isfinite(last(term)) for term in normalized) ||
            error("Satellite balance $(id) has a non-finite term coefficient.")
        new(id, String(unit), normalized)
    end
end

"""
    satellite_reference(results, anchors; id=:baseline)

Capture the solved driver levels needed to project an anchor set relative to a
named reference equilibrium.  Supply the returned reference to
[`satellite_projection`](@ref) for each scenario.
"""
function satellite_reference(obj, anchors::AbstractVector{SatelliteAnchor}; id::Symbol=:baseline)
    results = _results(obj)
    drivers = Dict{Symbol,Float64}()
    for anchor in anchors
        value = get(results.primals, anchor.driver, nothing)
        value === nothing && error("Satellite reference $(id) requires solved driver $(anchor.driver).")
        isfinite(value) && value > 0.0 ||
            error("Satellite reference $(id) requires a finite, strictly positive driver $(anchor.driver).")
        drivers[anchor.driver] = value
    end
    return SatelliteReference(id, drivers)
end

"""
    satellite_calibration_report(reference, anchors)

Report the difference between each anchor's declared monetary calibration
driver and the solved driver retained in a satellite reference.  The report is
diagnostic only: it does not modify the physical projection.
"""
function satellite_calibration_report(reference::SatelliteReference,
    anchors::AbstractVector{SatelliteAnchor})
    rows = NamedTuple[]
    for anchor in anchors
        reference_driver = get(reference.drivers, anchor.driver, nothing)
        reference_driver === nothing && error(
            "Satellite reference $(reference.id) is missing driver $(anchor.driver) for anchor $(anchor.id).")
        push!(rows, (
            id = anchor.id,
            driver = anchor.driver,
            calibration_driver = anchor.base_driver,
            reference_driver = reference_driver,
            absolute_difference = reference_driver - anchor.base_driver,
            relative_difference = (reference_driver - anchor.base_driver) / anchor.base_driver,
        ))
    end
    return rows
end

"""
    satellite_projection(results, anchors; reference=nothing, strict=true)

Project each anchor as `base_quantity * solved_driver / reference_driver`.
Passing a [`SatelliteReference`](@ref) constructed from the solved zero-policy
equilibrium preserves every observed quantity at that base point and makes
scenario changes relative to it.  Omitting `reference` retains the direct
calibration-driver projection for backwards compatibility.  With `strict=false`,
missing solved or reference drivers are reported rather than raising an error.
"""
function satellite_projection(obj, anchors::AbstractVector{SatelliteAnchor};
    reference::Union{Nothing,SatelliteReference}=nothing, strict::Bool=true)
    results = _results(obj)
    rows = NamedTuple[]
    for anchor in anchors
        driver_value = get(results.primals, anchor.driver, nothing)
        reference_driver = reference === nothing ? anchor.base_driver :
            get(reference.drivers, anchor.driver, nothing)
        if driver_value === nothing || reference_driver === nothing
            if strict
                driver_value === nothing && error(
                    "Satellite anchor $(anchor.id) requires solved driver $(anchor.driver).")
                error("Satellite reference $(reference.id) requires driver $(anchor.driver) for anchor $(anchor.id).")
            end
            push!(rows, (
                id = anchor.id,
                unit = anchor.unit,
                base_quantity = anchor.base_quantity,
                driver = anchor.driver,
                calibration_driver = anchor.base_driver,
                reference_id = reference === nothing ? :calibration : reference.id,
                reference_driver = reference_driver === nothing ? missing : reference_driver,
                solved_driver = missing,
                volume_index = missing,
                projected_quantity = missing,
                status = driver_value === nothing ? :missing_driver : :missing_reference_driver,
            ))
            continue
        end
        volume_index = driver_value / reference_driver
        push!(rows, (
            id = anchor.id,
            unit = anchor.unit,
            base_quantity = anchor.base_quantity,
            driver = anchor.driver,
            calibration_driver = anchor.base_driver,
            reference_id = reference === nothing ? :calibration : reference.id,
            reference_driver = reference_driver,
            solved_driver = driver_value,
            volume_index = volume_index,
            projected_quantity = anchor.base_quantity * volume_index,
            status = :projected,
        ))
    end
    return rows
end

"""
    satellite_balances(projection, balances; strict=true, tol=1e-8)

Evaluate signed satellite identities.  Each term is `anchor_id => coefficient`;
the balance residual is the sum of coefficient times projected quantity.  All
terms in a balance must carry its declared unit.
"""
function satellite_balances(projection, balances::AbstractVector{SatelliteBalance};
    strict::Bool=true, tol::Real=1.0e-8)
    tol >= 0.0 || error("Satellite-balance tolerance must be non-negative.")
    values = Dict{Symbol,NamedTuple}()
    for row in Tables.rows(projection)
        values[Symbol(row.id)] = (
            unit = String(row.unit),
            quantity = row.projected_quantity,
        )
    end

    rows = NamedTuple[]
    for balance in balances
        missing_terms = Symbol[]
        residual = 0.0
        scale = 0.0
        for term in balance.terms
            anchor_id = first(term)
            coefficient = last(term)
            anchor = get(values, anchor_id, nothing)
            if anchor === nothing || ismissing(anchor.quantity)
                push!(missing_terms, anchor_id)
                continue
            end
            anchor.unit == balance.unit || error(
                "Satellite balance $(balance.id) has unit $(balance.unit), but anchor $(anchor_id) has unit $(anchor.unit).")
            contribution = coefficient * anchor.quantity
            residual += contribution
            scale += abs(contribution)
        end
        if !isempty(missing_terms)
            strict && error("Satellite balance $(balance.id) is missing projected anchors: $(join(string.(missing_terms), ", ")).")
            push!(rows, (
                id = balance.id,
                unit = balance.unit,
                residual = missing,
                scale = missing,
                passes = false,
                status = :missing_anchor,
            ))
            continue
        end
        passes = abs(residual) <= tol * max(1.0, scale)
        push!(rows, (
            id = balance.id,
            unit = balance.unit,
            residual = residual,
            scale = scale,
            passes = passes,
            status = :evaluated,
        ))
    end
    return rows
end
