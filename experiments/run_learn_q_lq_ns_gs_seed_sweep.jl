include(joinpath(@__DIR__, "learn_q_lq_game.jl"))

const LEARN_Q_LQ_SWEEP_NS = (100, 200, 300, 400, 500)
const LEARN_Q_LQ_SWEEP_GS = (1,)

function run_learn_q_lq_ns_gs_seed_sweep(;
    S::Int = 3,
    seeds = nothing,
    num_iters::Union{Nothing,Int} = nothing,
    q_lr::Union{Nothing,Real} = nothing,
    γ::Union{Nothing,Real} = nothing,
    picnn_weight_scale::Union{Nothing,Real} = nothing,
    picnn_wz_softplus::Union{Nothing,Bool} = nothing,
    picnn_z_gate_softplus::Union{Nothing,Bool} = nothing,
    ns_values = LEARN_Q_LQ_SWEEP_NS,
    gs_values = LEARN_Q_LQ_SWEEP_GS,
)
    seed_list = seeds === nothing ? collect(1:S) : collect(Int, seeds)
    isempty(seed_list) && throw(ArgumentError("seeds must be non-empty"))
    all(s -> s >= 1, seed_list) || throw(ArgumentError("all seeds must be >= 1, got $seed_list"))
    num_iters === nothing || num_iters >= 1 ||
        throw(ArgumentError("num_iters must be >= 1, got $num_iters"))
    q_lr === nothing || q_lr > 0 || throw(ArgumentError("q_lr must be > 0, got $q_lr"))
    γ === nothing || (0 <= γ < 1) || throw(ArgumentError("γ must be in [0,1), got $γ"))
    picnn_weight_scale === nothing || picnn_weight_scale > 0 ||
        throw(ArgumentError("picnn_weight_scale must be > 0, got $picnn_weight_scale"))

    total = length(seed_list) * length(ns_values) * length(gs_values)
    println(
        "learn_q_lq sweep: seeds=$seed_list, ns=$ns_values, gs=$gs_values",
        num_iters === nothing ? "" : ", num_iters=$num_iters",
        q_lr === nothing ? "" : ", q_lr=$q_lr",
        γ === nothing ? "" : ", γ=$γ",
        picnn_weight_scale === nothing ? "" : ", picnn_weight_scale=$picnn_weight_scale",
        picnn_wz_softplus === nothing ? "" : ", picnn_wz_softplus=$picnn_wz_softplus",
        picnn_z_gate_softplus === nothing ? "" : ", picnn_z_gate_softplus=$picnn_z_gate_softplus",
        " → $total runs",
    )
    println("threads=", Threads.nthreads())

    run_idx = 0
    for seed in seed_list
        for ns in ns_values
            for gs in gs_values
                run_idx += 1
                overrides = (;
                    rng_seed = seed,
                    trajectory_rng_seed = seed,
                    num_samples_per_iter = ns,
                    num_grad_steps = gs,
                    csv_skip_first_iter = false,
                )
                if num_iters !== nothing
                    overrides = merge(overrides, (; num_iters = num_iters))
                end
                if q_lr !== nothing
                    overrides = merge(overrides, (; q_lr = float(q_lr)))
                end
                if γ !== nothing
                    overrides = merge(overrides, (; γ = float(γ)))
                end
                if picnn_weight_scale !== nothing
                    overrides = merge(overrides, (; picnn_weight_scale = float(picnn_weight_scale)))
                end
                if picnn_wz_softplus !== nothing
                    overrides = merge(overrides, (; picnn_wz_softplus = picnn_wz_softplus))
                end
                if picnn_z_gate_softplus !== nothing
                    overrides = merge(overrides, (; picnn_z_gate_softplus = picnn_z_gate_softplus))
                end
                cfg = merge(CONFIG, overrides)
                println("\n", "="^72)
                println(
                    "[$run_idx/$total] seed=$seed  ns=$ns  gs=$gs  γ=$(cfg.γ)  num_iters=$(cfg.num_iters)  q_lr=$(cfg.q_lr)  picnn_weight_scale=$(cfg.picnn_weight_scale)  picnn_wz_softplus=$(cfg.picnn_wz_softplus)  picnn_z_gate_softplus=$(cfg.picnn_z_gate_softplus)",
                )
                println("="^72, "\n")
                learn_q_lq(; cfg = cfg)
                println("\nFinished seed=$seed ns=$ns gs=$gs\n")
            end
        end
    end
    println("All $total runs completed.")
    return nothing
end

function _parse_sweep_cli_args(args)
    S = length(args) >= 1 ? parse(Int, args[1]) : 3
    num_iters = length(args) >= 2 ? parse(Int, args[2]) : nothing
    return S, num_iters
end

if abspath(PROGRAM_FILE) == @__FILE__
    S_cli, num_iters_cli = _parse_sweep_cli_args(ARGS)
    run_learn_q_lq_ns_gs_seed_sweep(; S = S_cli, num_iters = num_iters_cli)
end
