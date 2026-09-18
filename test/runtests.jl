using JCGECalibrate
using JCGECore
using JCGEOutput
using JCGERuntime
using Test

@testset "JCGEOutput" begin
    struct DummyBlock
        name::Symbol
    end

    @testset "render_equations and render_block" begin
        ctx = KernelContext()
        register_equation!(ctx; tag=:eq1, block=:prod, payload=(indices=(1,), info="x = y", constraint=nothing))
        register_equation!(ctx; tag=:eq2, block=:prod, payload=(indices=(), info="z = w", constraint=nothing))
        register_equation!(ctx; tag=:eq3, block=:market, payload="q = sum(x)")
        register_equation!(ctx; tag=:eq4, block=:prod,
            payload=(indices=(), expr=EAdd([EVar(:x), EConst(1)]), constraint=nothing))
        register_equation!(ctx; tag=:check, block=:market,
            payload=(indices=(), info="accounting identity", constraint=nothing,
                condition_role=:accounting_check))

        rendered = render_equations(ctx; format=:plain, level=:equation)
        @test occursin("prod.eq1", rendered)
        @test occursin("x = y", rendered)
        @test occursin("x + 1", rendered)

        rendered_roles = render_equations(ctx; format=:plain, level=:equation,
            show_condition_roles=true)
        @test occursin("market.check (accounting check)", rendered_roles)

        block_rendered = render_block(ctx, :prod; format=:markdown)
        @test occursin("z = w", block_rendered)
        @test !occursin("q = sum(x)", block_rendered)
    end

    @testset "equation AST rendering" begin
        @test render_expr(ELe(EVar(:x), EConst(2)); format=:plain) == "x <= 2"
        @test render_expr(EGe(ELog(EVar(:x)), EConst(0)); format=:plain) == "log(x) >= 0"
        @test render_expr(ELe(ELog(EAdd([EVar(:x), EConst(1)])), EConst(3)); format=:latex) ==
              "\\log\\left(x + 1\\right) \\le 3"
        @test render_expr(EDiv(EVar(:price), EMul([EConst(1.02), EVar(:input_price)]));
            format=:latex) == "\\frac{price}{1.02 \\cdot input\\_price}"
        @test render_expr(EPow(EVar(:quantity), EParam(:material_share)); format=:latex) ==
              "{quantity}^{material\\_share}"
        @test render_expr(EPow(EMul([EVar(:metal), EVar(:labor)]),
            EAdd([EParam(:alpha), EParam(:beta)])); format=:latex) ==
              "\\left(metal \\cdot labor\\right)^{alpha + beta}"
        @test render_expr(EPow(EMul([EVar(:metal), EVar(:labor)]),
            EAdd([EParam(:alpha), EParam(:beta)])); format=:plain) ==
              "(metal * labor)^(alpha + beta)"

        ctx = KernelContext()
        expr = ELe(ESum(:i, [:a, :b], EVar(:x, [EIndex(:i)])), EConst(10))
        register_equation!(ctx; tag=:limit, block=:market,
            payload=(indices=(), expr=expr, constraint=nothing))

        rendered = render_equations(ctx; format=:markdown, level=:equation)
        @test occursin("\\le", rendered)
        @test occursin("\\sum_{i \\in \\mathcal{D}_{i}}", rendered)
        @test occursin("Domain i in { a, b }", rendered)
    end

    @testset "objective rendering" begin
        ctx = KernelContext()
        register_equation!(ctx; tag=:objective, block=:single_objective,
            payload=(indices=(),
                info="household utility objective under fiscal closure",
                objective_expr=EAdd([ELog(EVar(:x)), ELog(EVar(:y))]),
                objective_sense=:Max,
                constraint=nothing))

        plain = render_equations(ctx; format=:plain, level=:equation)
        @test occursin("single_objective.objective", plain)
        @test occursin("max log(x) + log(y)", plain)

        markdown = render_equations(ctx; format=:markdown, level=:equation)
        @test occursin("\\max\\;", markdown)
        @test occursin("\\log\\left(x\\right) + \\log\\left(y\\right)", markdown)
    end

    @testset "equation families and reports" begin
        ctx = KernelContext()
        market_expr = EEq(EVar(:supply), EVar(:demand))
        register_equation!(ctx; tag=:clearing, block=:market,
            payload=(indices=(:DE,), expr=market_expr, constraint=nothing))
        register_equation!(ctx; tag=:clearing, block=:market,
            payload=(indices=(:FR,), expr=market_expr, constraint=nothing))
        register_equation!(ctx; tag=:clearing, block=:market,
            payload=(indices=(:IT,), expr=EEq(EVar(:supply),
                EAdd([EVar(:demand), EConst(1)])), constraint=nothing))
        register_equation!(ctx; tag=:utility, block=:household,
            payload=(indices=(), objective_expr=ELog(EVar(:utility)),
                objective_sense=:Max, constraint=nothing))
        register_equation!(ctx; tag=:limit, block=:market,
            payload=(indices=(), expr=ELe(ESum(:i, [:a, :b],
                EVar(:use, [EIndex(:i)])), EConst(1)), constraint=nothing))
        register_equation!(ctx; tag=:description, block=:household,
            payload=(indices=(), info="closure description", constraint=nothing))
        register_equation!(ctx; tag=:start, block=:initial_values,
            payload=(indices=(:supply,), info="start supply = 1.0", constraint=nothing))

        families = equation_families(ctx)
        clearing_families = filter(family -> family.block == :market && family.tag == :clearing,
            families)
        @test length(clearing_families) == 2
        clearing = only(filter(family -> length(family.instances) == 2, clearing_families))
        @test clearing.kind == :equation
        @test length(clearing.instances) == 2
        @test clearing.expression == market_expr

        compact = render_equations(ctx; format=:plain, view=:family, level=:equation)
        @test occursin("2 registered instances", compact)
        @test occursin("supply = demand + 1", compact)

        compact_latex = render_equations(ctx; format=:latex, view=:family, level=:equation)
        @test occursin("\\begin{align*}", compact_latex)

        latex = render_equations(ctx; format=:latex, level=:equation)
        @test occursin("\\[", latex)
        @test occursin("supply = demand", latex)

        report = render_equation_report(ctx; format=:latex)
        @test occursin("\\section*{Model equation report}", report)
        @test occursin("\\subsection*{Objective functions}", report)
        @test occursin("\\begin{align*}", report)
        @test occursin("supply &= demand", report)
        @test occursin("\\sum_{i \\in \\mathcal{D}_{i}}", report)
        @test occursin("&\\le 1", report)
        @test occursin("Internal sum/product domains", report)
        @test occursin("2 registered instances", report)
        @test occursin("Description only; no equation AST was registered.", report)
        @test !occursin("start supply = 1.0", report)

        audit_report = render_equation_report(ctx; format=:plain,
            include_solver_annotations=true)
        @test occursin("start supply = 1.0", audit_report)

        expanded = render_equation_report(ctx; format=:markdown, view=:expanded)
        @test occursin("Objective functions", expanded)
        @test occursin("market.clearing[DE]", expanded)
        @test occursin("market.clearing[FR]", expanded)
        @test occursin("market.clearing[IT]", expanded)

        indexed = KernelContext()
        for (i, alpha) in ((:a, 0.3), (:b, 0.7))
            register_equation!(indexed; tag=:demand, block=:example,
                payload=(indices=(i,), index_names=(:i,),
                    params=(alpha=Dict(i => alpha),),
                    expr=EEq(EVar(:x, [i]), EMul([EParam(:alpha, [i]), EVar(:y, [i])])),
                    constraint=nothing))
        end
        templates = equation_templates(indexed)
        @test length(templates) == 1
        @test length(only(templates).instances) == 2
        indexed_report = render_equation_report(indexed; format=:latex, view=:indexed)
        @test occursin("x_{i} &= alpha_{i}", indexed_report)

        compound = KernelContext()
        for (position, (product, region)) in enumerate(((:A, :DE), (:A, :FR),
            (:B, :DE), (:B, :FR)))
            good = Symbol("IND_", region, "_", product)
            weight = 0.1 * position
            register_equation!(compound; tag=:allocation, block=:trade,
                payload=(indices=(product, region), index_names=(:product, :region),
                    params=(share=Dict((product, region) => weight),),
                    expr=EEq(EVar(:purchase, [product, region]),
                        EMul([EConst(weight), EVar(:price, [good])])), constraint=nothing))
        end
        compound_templates = equation_templates(compound)
        @test length(compound_templates) == 1
        compound_report = render_equation_report(compound; format=:latex, view=:indexed)
        @test occursin("share_{product,region}", compound_report)
        @test occursin("\\langle product, region\\rangle", compound_report)

        calibrated = KernelContext()
        for (product, region, coefficient) in ((:A, :DE, 0.25), (:A, :FR, 0.4),
            (:B, :DE, 0.6), (:B, :FR, 0.75))
            register_equation!(calibrated; tag=:production, block=:example,
                payload=(indices=(product, region), index_names=(:product, :region),
                    expr=EEq(EVar(:output, [product, region]),
                        EMul([EConst(coefficient), EPow(EVar(:input, [product, region]),
                            EConst(coefficient))])),
                    constraint=nothing))
        end
        calibrated_templates = equation_templates(calibrated)
        @test length(calibrated_templates) == 1
        calibrated_report = render_equation_report(calibrated; format=:latex, view=:indexed)
        @test occursin("\\kappa_{1,\\langle product, region\\rangle}", calibrated_report)
        @test occursin("\\kappa_{2,\\langle product, region\\rangle}", calibrated_report)
        @test !occursin("0.25", calibrated_report)
        @test !occursin("0.75", calibrated_report)

        unresolved = KernelContext()
        for good in (:IND_DE_A, :IND_DE_B)
            register_equation!(unresolved; tag=:allocation, block=:trade,
                payload=(indices=(:A,), index_names=(:product,),
                    expr=EEq(EVar(:purchase, [:A]), EVar(:price, [good])), constraint=nothing))
        end
        @test length(equation_templates(unresolved)) == 2

        eol = KernelContext()
        eol_coordinates = Dict{Tuple,Tuple}()
        for region in (:DE, :FR), family in (:ELMA, :RATV), route in (:REC, :REP)
            line = Symbol("EOL_", region, "_", family, "_", route)
            eol_coordinates[(line,)] = (region, family, route)
            register_equation!(eol; tag=:choice, block=:encoded_eol,
                payload=(indices=(line,),
                    expr=EEq(EVar(:flow, [line]), EMul([
                        EParam(:share, [line]), EVar(:price, [line])])),
                    constraint=nothing))
        end
        eol_mapping = EquationReportMapping(
            source_block=:encoded_eol,
            source_tag=:choice,
            report_block=:eol_allocation,
            report_tag=:choice,
            index_names=(:region, :family, :route),
            coordinates=eol_coordinates,
        )
        eol_templates = equation_templates(eol; report_mappings=[eol_mapping])
        @test length(eol_templates) == 1
        @test only(eol_templates).block == :eol_allocation
        @test only(eol_templates).index_names == (:region, :family, :route)
        eol_report = render_equation_report(eol; format=:latex, view=:indexed,
            report_mappings=[eol_mapping])
        @test occursin("flow_{\\langle region, family, route\\rangle}", eol_report)
        @test occursin("share_{\\langle region, family, route\\rangle}", eol_report)
        @test !occursin("EOL\\_DE\\_ELMA\\_REC", eol_report)

        sample_eol_source = first(keys(eol_coordinates))
        partial_eol_mapping = EquationReportMapping(
            source_block=:encoded_eol,
            source_tag=:choice,
            index_names=(:region, :family, :route),
            coordinates=Dict(sample_eol_source => eol_coordinates[sample_eol_source]),
        )
        @test_throws ArgumentError equation_templates(eol;
            report_mappings=[partial_eol_mapping])
        @test !isempty(equation_templates(eol;
            report_mappings=[partial_eol_mapping], strict_report_mappings=false))

        unused_eol_coordinates = copy(eol_coordinates)
        unused_eol_coordinates[(:NOT_REGISTERED,)] = (:DE, :ELMA, :REC)
        unused_eol_mapping = EquationReportMapping(
            source_block=:encoded_eol,
            source_tag=:choice,
            index_names=(:region, :family, :route),
            coordinates=unused_eol_coordinates,
        )
        @test_throws ArgumentError equation_templates(eol;
            report_mappings=[unused_eol_mapping])
        @test !isempty(equation_templates(eol;
            report_mappings=[unused_eol_mapping], strict_report_mappings=false))
        @test_throws ArgumentError equation_templates(eol;
            report_mappings=[eol_mapping, eol_mapping])
        @test_throws ErrorException render_equation_report(eol; format=:plain,
            view=:family, report_mappings=[eol_mapping])

        regional_sum = KernelContext()
        for (region, labor, capital) in ((:DE, :FAC_DE_LAB, :FAC_DE_CAP),
            (:FR, :FAC_FR_LAB, :FAC_FR_CAP))
            register_equation!(regional_sum; tag=:income, block=:household_source,
                payload=(indices=(region,),
                    expr=EEq(EVar(:income, [region]), ESum(:factor, [labor, capital],
                        EVar(:wage, [EIndex(:factor)]))), constraint=nothing))
        end
        regional_sum_mapping = EquationReportMapping(
            source_block=:household_source,
            source_tag=:income,
            report_block=:household,
            report_tag=:income,
            index_names=(:region,),
            coordinates=Dict((:DE,) => (:DE,), (:FR,) => (:FR,)),
            domain_values=Dict(
                :FAC_DE_LAB => :LAB,
                :FAC_DE_CAP => :CAP,
                :FAC_FR_LAB => :LAB,
                :FAC_FR_CAP => :CAP,
            ),
        )
        sum_templates = equation_templates(regional_sum; report_mappings=[regional_sum_mapping])
        @test length(sum_templates) == 1
        sum_report = render_equation_report(regional_sum; format=:latex, view=:indexed,
            report_mappings=[regional_sum_mapping])
        @test occursin("\\sum_{factor \\in \\mathcal{D}_{factor}}", sum_report)
        @test occursin("factor \\in \\{LAB, CAP\\}", sum_report)
        @test !occursin("FAC\\_DE\\_LAB", sum_report)

        regional_product = KernelContext()
        for (region, labor, capital) in ((:DE, :FAC_DE_LAB, :FAC_DE_CAP),
            (:FR, :FAC_FR_LAB, :FAC_FR_CAP))
            register_equation!(regional_product; tag=:technology, block=:production_source,
                payload=(indices=(region,),
                    expr=EEq(EVar(:scale, [region]), EProd(:factor, [labor, capital],
                        EParam(:productivity, [EIndex(:factor)]))), constraint=nothing))
        end
        product_mapping = EquationReportMapping(
            source_block=:production_source,
            source_tag=:technology,
            report_block=:production,
            report_tag=:technology,
            index_names=(:region,),
            coordinates=Dict((:DE,) => (:DE,), (:FR,) => (:FR,)),
            domain_values=regional_sum_mapping.domain_values,
        )
        product_report = render_equation_report(regional_product; format=:latex,
            view=:indexed, report_mappings=[product_mapping])
        @test occursin("\\prod_{factor \\in \\mathcal{D}_{factor}}", product_report)
        @test occursin("factor \\in \\{LAB, CAP\\}", product_report)
        @test !occursin("FAC\\_FR\\_CAP", product_report)

        physical_link = KernelContext()
        for (quantity, coordinates) in ((:physical_DE_ELMA_REC, (:DE, :ELMA, :REC)),
            (:physical_FR_RATV_REC, (:FR, :RATV, :REC)))
            register_equation!(physical_link; tag=:quantity_link, block=:physical_source,
                payload=(indices=(quantity,), index_names=(:quantity,),
                    expr=EEq(EVar(:physical_tonnes, [EIndex(:quantity)]),
                        EParam(:coefficient, [EIndex(:quantity)])), constraint=nothing))
        end
        physical_mapping = EquationReportMapping(
            source_block=:physical_source,
            source_tag=:quantity_link,
            report_block=:physical_flow,
            report_tag=:quantity_link,
            index_names=(:region, :family, :route),
            coordinates=Dict(
                (:physical_DE_ELMA_REC,) => (:DE, :ELMA, :REC),
                (:physical_FR_RATV_REC,) => (:FR, :RATV, :REC),
            ),
            index_projections=Dict(:quantity => (:region, :family, :route)),
        )
        physical_report = render_equation_report(physical_link; format=:latex,
            view=:indexed, report_mappings=[physical_mapping])
        @test occursin("{physical\\_tonnes}_{\\langle region, family, route\\rangle}",
            physical_report)
        @test !occursin("_{quantity}", physical_report)

        unused_projection_mapping = EquationReportMapping(
            source_block=:physical_source,
            source_tag=:quantity_link,
            index_names=(:region, :family, :route),
            coordinates=physical_mapping.coordinates,
            index_projections=Dict(:other => (:region,)),
        )
        @test_throws ArgumentError equation_templates(physical_link;
            report_mappings=[unused_projection_mapping])

        unused_domain_mapping = EquationReportMapping(
            source_block=:household_source,
            source_tag=:income,
            index_names=(:region,),
            coordinates=Dict((:DE,) => (:DE,), (:FR,) => (:FR,)),
            domain_values=Dict(:NOT_A_FACTOR => :OTHER),
        )
        @test_throws ArgumentError equation_templates(regional_sum;
            report_mappings=[unused_domain_mapping])

        explicit_reference = KernelContext()
        register_equation!(explicit_reference; tag=:price_link, block=:encoded_reference,
            payload=(indices=(:case_DE_ELMA,),
                expr=EEq(EVar(:output, [:case_DE_ELMA]),
                    EParam(:price, [:PRICE_DE_ELMA])), constraint=nothing))
        reference_mapping = EquationReportMapping(
            source_block=:encoded_reference,
            source_tag=:price_link,
            index_names=(:region, :family),
            coordinates=Dict((:case_DE_ELMA,) => (:DE, :ELMA)),
            reference_indices=Dict(
                (:variable, :output, (:case_DE_ELMA,)) => ((:region, :family),),
                (:parameter, :price, (:PRICE_DE_ELMA,)) => ((:region, :family),),
            ),
        )
        reference_report = render_equation_report(explicit_reference; format=:latex,
            view=:indexed, report_mappings=[reference_mapping])
        @test occursin("output_{\\langle region, family\\rangle}", reference_report)
        @test occursin("price_{\\langle region, family\\rangle}", reference_report)
        @test !occursin("PRICE\\_DE\\_ELMA", reference_report)

        unused_reference_mapping = EquationReportMapping(
            source_block=:encoded_reference,
            source_tag=:price_link,
            index_names=(:region, :family),
            coordinates=Dict((:case_DE_ELMA,) => (:DE, :ELMA)),
            reference_indices=Dict(
                (:variable, :missing, (:MISSING,)) => ((:region,),),
            ),
        )
        @test_throws ArgumentError equation_templates(explicit_reference;
            report_mappings=[unused_reference_mapping])
    end

    @testset "render_symbols and render_blocks" begin
        ctx = KernelContext()
        register_variable!(ctx, :x, 1.0)
        register_variable!(ctx, :y, 2.0)

        symbols = render_symbols(ctx; format=:plain)
        @test occursin("x", symbols)
        @test occursin("y", symbols)

        sections = [
            section(:production, Any[DummyBlock(:prod_a), DummyBlock(:prod_b)]),
            section(:markets, Any[DummyBlock(:mkt)]),
        ]
        rendered_sections = render_sections(sections; format=:markdown)
        @test occursin("production", rendered_sections)
        @test occursin("prod_a", rendered_sections)
    end

    @testset "results container" begin
        ctx = KernelContext()
        register_variable!(ctx, :x, 1.0)
        register_equation!(ctx; tag=:eq1, block=:prod,
            payload=(indices=(), expr=EEq(EVar(:x), EConst(1)), constraint=nothing,
                condition_role=:accounting_check, residual=0.0))
        results = collect_results(ctx; metadata=Dict(:scenario_id => "base"))
        @test results.metadata[:scenario_id] == "base"
        @test only(results.accounting_checks).residual == 0.0

        rows = tidy(results)
        @test rows isa Vector
        @test any(row -> row.kind == :accounting_check, rows)

        json_path = joinpath(mktempdir(), "results.json")
        csv_path = joinpath(mktempdir(), "results.csv")
        to_json(results, json_path)
        to_csv(results, csv_path)
        @test isfile(json_path)
        @test isfile(csv_path)

        roundtrip = results_from_json(json_path)
        @test roundtrip.primals == results.primals
        @test only(roundtrip.accounting_checks).residual == 0.0

        dataset = JCGEOutput.to_dualsignals(results; dataset_id="test", component_type_by_block=Dict(:prod => :sector))
        @test dataset.dataset_id == "test"
        @test any(solution -> solution.slack == 0.0, dataset.constraint_solutions)

        ds_json = joinpath(mktempdir(), "dualsignals.json")
        ds_dir = mktempdir()
        JCGEOutput.write_dualsignals_json(results, ds_json; dataset_id="test")
        JCGEOutput.write_dualsignals_csv(results, ds_dir; dataset_id="test", prefix="test")
        @test isfile(ds_json)
        @test isfile(joinpath(ds_dir, "test_metadata.csv"))

        ctx_dataset = JCGEOutput.to_dualsignals(ctx; dataset_id="ctx")
        @test ctx_dataset.dataset_id == "ctx"
    end

    @testset "results roundtrip formats" begin
        results = Results(Dict(:x => 1.0), Dict{Symbol,Float64}(), NamedTuple[], NamedTuple[], Dict{Symbol,Any}())
        arrow_path = joinpath(mktempdir(), "results.arrow")
        parquet_path = joinpath(mktempdir(), "results.parquet")
        to_arrow(results, arrow_path)
        to_parquet(results, parquet_path)
        @test isfile(arrow_path)
        @test isfile(parquet_path)

        from_arrow = results_from_arrow(arrow_path)
        from_parquet = results_from_parquet(parquet_path)
        @test get(from_arrow.primals, :x, 0.0) == 1.0
        @test get(from_parquet.primals, :x, 0.0) == 1.0
    end

    @testset "satellite reporting" begin
        results = Results(
            Dict(:volume_a => 12.0, :volume_b => 7.2),
            Dict{Symbol,Float64}(),
            NamedTuple[],
            NamedTuple[],
            Dict{Symbol,Any}(),
        )
        anchors = [
            SatelliteAnchor(:a, "tonnes", 6.0, :volume_a, 10.0),
            SatelliteAnchor(:b, "tonnes", 5.0, :volume_b, 5.0),
        ]
        projection = satellite_projection(results, anchors)
        @test projection[1].volume_index == 1.2
        @test isapprox(projection[1].projected_quantity, 7.2)
        @test isapprox(projection[2].projected_quantity, 7.2)
        @test all(row -> row.status == :projected, projection)

        reference = satellite_reference(results, anchors)
        @test reference.id == :baseline
        baseline_projection = satellite_projection(results, anchors; reference)
        @test all(row -> row.volume_index == 1.0, baseline_projection)
        @test all(row -> row.projected_quantity == row.base_quantity, baseline_projection)
        @test all(row -> row.reference_id == :baseline, baseline_projection)
        calibration_report = satellite_calibration_report(reference, anchors)
        @test calibration_report[1].calibration_driver == 10.0
        @test calibration_report[1].reference_driver == 12.0
        @test calibration_report[1].relative_difference == 0.2

        scenario_results = Results(
            Dict(:volume_a => 18.0, :volume_b => 3.6),
            Dict{Symbol,Float64}(),
            NamedTuple[],
            NamedTuple[],
            Dict{Symbol,Any}(),
        )
        scenario_projection = satellite_projection(scenario_results, anchors; reference)
        @test scenario_projection[1].volume_index == 1.5
        @test scenario_projection[1].projected_quantity == 9.0
        @test scenario_projection[2].volume_index == 0.5
        @test scenario_projection[2].projected_quantity == 2.5

        balances = [SatelliteBalance(:mass_check, "tonnes", [:a => 1.0, :b => -1.0])]
        checked = satellite_balances(projection, balances)
        @test only(checked).passes
        @test only(checked).residual == 0.0

        incomplete = satellite_projection(results, [SatelliteAnchor(:missing, "kg", 1.0, :missing_driver, 1.0)]; strict=false)
        @test only(incomplete).status == :missing_driver
        partial_reference = SatelliteReference(:partial, Dict(:volume_a => 12.0))
        missing_reference = satellite_projection(results, anchors; reference=partial_reference, strict=false)
        @test missing_reference[2].status == :missing_reference_driver
        @test_throws ErrorException satellite_projection(results, [SatelliteAnchor(:missing, "kg", 1.0, :missing_driver, 1.0)])
        @test_throws ErrorException SatelliteAnchor(:bad, "", 1.0, :volume_a, 1.0)
        @test_throws ErrorException SatelliteBalance(:bad, "kg", Pair{Symbol,Float64}[])
    end

    @testset "SAM output" begin
        labels = [:BRD, :MLK, :CAP, :LAB, :IDT, :TRF, :HOH, :GOV, :INV, :EXT]
        sam = JCGECalibrate.LabeledMatrix(zeros(length(labels), length(labels)), labels, labels)
        sam_table = JCGECalibrate.SAMTable(
            [:BRD, :MLK],
            [:CAP, :LAB],
            :LAB,
            :IDT,
            :TRF,
            :HOH,
            :GOV,
            :INV,
            :EXT,
            sam,
        )

        results = Results(Dict(
            :X_BRD_BRD => 2.0,
            :pq_BRD => 3.0,
            :F_CAP_BRD => 7.0,
            :pf_CAP => 8.0,
            :Xp_BRD => 4.0,
            :Xg_BRD => 1.0,
            :Xv_BRD => 0.5,
            :E_BRD => 2.0,
            :M_BRD => 1.0,
            :pe_BRD => 6.0,
            :pm_BRD => 5.0,
            :Td => 9.0,
            :Sp => 10.0,
            :Sg => 11.0,
            :Sf => 12.0,
            :Tz_BRD => 13.0,
            :Tm_BRD => 14.0,
            :FF_CAP => 15.0,
        ), Dict{Symbol,Float64}(), NamedTuple[], NamedTuple[], Dict{Symbol,Any}())

        spec = RunSpec(
            "test",
            ModelSpec(Any[], Sets([:BRD, :MLK], [:BRD, :MLK], [:CAP, :LAB], [:HOH, :GOV, :INV, :EXT]),
                Mappings(Dict(:BRD => :BRD, :MLK => :MLK))),
            ClosureSpec(:LAB),
            ScenarioSpec(:base, Dict{Symbol,Any}()),
        )

        out = sam_from_solution(results; spec=spec, sam_table=sam_table, include_quantities=true)
        sam_vals = out.values

        @test sam_vals[:BRD, :BRD] == 6.0
        @test sam_vals[:CAP, :BRD] == 56.0
        @test sam_vals[:BRD, :HOH] == 12.0
        @test sam_vals[:BRD, :GOV] == 3.0
        @test sam_vals[:BRD, :INV] == 1.5
        @test sam_vals[:BRD, :EXT] == 12.0
        @test sam_vals[:EXT, :BRD] == 5.0
        @test sam_vals[:GOV, :HOH] == 9.0
        @test sam_vals[:INV, :HOH] == 10.0
        @test sam_vals[:INV, :GOV] == 11.0
        @test sam_vals[:INV, :EXT] == 12.0
        @test sam_vals[:IDT, :BRD] == 13.0
        @test sam_vals[:TRF, :BRD] == 14.0
        @test sam_vals[:HOH, :CAP] == 120.0

        @test_throws ErrorException sam_from_solution(results; spec=spec, sam_table=sam_table, valuation=:baseline)

        sam_path = joinpath(mktempdir(), "sam.csv")
        write_sam_csv(out.values, sam_path)
        @test isfile(sam_path)
    end
end
