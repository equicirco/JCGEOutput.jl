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

        rendered = render_equations(ctx; format=:plain, level=:equation)
        @test occursin("prod.eq1", rendered)
        @test occursin("x = y", rendered)
        @test occursin("x + 1", rendered)

        block_rendered = render_block(ctx, :prod; format=:markdown)
        @test occursin("z = w", block_rendered)
        @test !occursin("q = sum(x)", block_rendered)
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
            payload=(indices=(), expr=EEq(EVar(:x), EConst(1)), constraint=nothing))
        results = collect_results(ctx; metadata=Dict(:scenario_id => "base"))
        @test results.metadata[:scenario_id] == "base"

        rows = tidy(results)
        @test rows isa Vector

        json_path = joinpath(mktempdir(), "results.json")
        csv_path = joinpath(mktempdir(), "results.csv")
        to_json(results, json_path)
        to_csv(results, csv_path)
        @test isfile(json_path)
        @test isfile(csv_path)

        roundtrip = results_from_json(json_path)
        @test roundtrip.primals == results.primals

        dataset = JCGEOutput.to_dualsignals(results; dataset_id="test", component_type_by_block=Dict(:prod => :sector))
        @test dataset.dataset_id == "test"

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
