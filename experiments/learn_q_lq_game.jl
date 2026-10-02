using LearningValueFunctions
using Dates
using Flux
using LinearAlgebra
using LinearAlgebra: BLAS
using Optimisers
using Printf: @sprintf, @printf
using Random
using Statistics

# ──────────────────────────────────────────────────────────────────────────────
# non_lq: two stacked 4-state double integrators, nonlinear separation cost.

const CONFIG = (
    problem_type = "non_lq",
    nx = 8,
    nu = 2,
    nv = 2,
    R1 = 0.01 .* Matrix{Float64}(I, 2, 2),
    R2 = 0.02 .* Matrix{Float64}(I, 2, 2),
    A_per_player = Float64[
        1.0 0.0 0.1 0.0
        0.0 1.0 0.0 0.1
        0.0 0.0 1.0 0.0
        0.0 0.0 0.0 1.0
    ],
    B_per_player = Float64[
        0.005 0.0
        0.0 0.005
        0.1 0.0
        0.0 0.1
    ],

    # `nothing` → built-in non_lq exponential separation cost. Non-`nothing` overrides.
    stage_cost = nothing,
    non_lq_exp_weight = 0.01,
    non_lq_exp_width_x = 0.1,
    non_lq_exp_width_y = 0.1,

    # TD / sampling (1-step transitions; ICNN path requires N == 1)
    γ = 0.95,
    num_samples_per_iter = 32,
    N = 1,
    num_grad_steps = 1,  # Adam steps on each frozen sample batch
    num_iters = 200,
    α = 0.1, # legacy linear-FA step (unused for ICNN Q)

    # Convex–concave Q network (PICNN §3.2 + MatrixNet C(x))
    # hidden_A = [64, 64],
    # hidden_B = [64, 64],
    # hidden_C = [64],
    hidden_A = [32, 32],
    hidden_B = [32, 32],
    hidden_C = [32],
    # hidden_A = [16, 16],
    # hidden_B = [16, 16],
    # hidden_C = [16],
    q_lr = 1e-2,   # Adam learning rate for ZeroSumQICNN
    q_grad_clip_norm = 5.0,  # global grad-norm clip (Optimisers.ClipNorm) before Adam
    # After this many TD iterations, switch to `q_lr_after` / `q_grad_clip_norm_after` (`nothing` = no switch).
    q_optimizer_change_after_iters = 10000,  # e.g. 1000
    q_lr_after = 1e-5,
    q_grad_clip_norm_after = 5.0,
    picnn_weight_scale = 0.5,  # PICNN A/B init scale (1.0 = unscaled randn/rand; smaller → stabler early TD)
    # `true` → W^{(z)} via icnn_softplus; `false` → abs.(W_z) (default, matches prior runs).
    picnn_wz_softplus = false,
    # `true` → z passthrough gate via icnn_softplus; `false` → ReLU (default).
    picnn_z_gate_softplus = false,
    # Initial state: uniform in box (training uses similar state ranges via position_box_halfwidth)
    x_lo = -2.5,
    x_hi = 2.5,

    rng_seed = 42,

    # Thread-parallel trajectory rollouts (`rollout_zero_sum_q!`); needs `julia -t N` with N>1
    parallel_rollouts = true,
    trajectory_rng_seed = nothing,   # nothing → use `rng_seed` for reproducible parallel draws
    blas_single_thread_parallel = true,

    stage_solver_kwargs = (; verbose = false, max_iterations = 150, step_size = 1.0, tol = 1e-8),

    # Process noise: x' = A x + B₁u + B₂v + σ ξ, ξ~N(0,I). Scale ~0.01 ≈ 10% of max |Δv|
    # from one control step (B gives Δv≲0.1); set 0 for deterministic dynamics.
    dynamics_noise_std = 0.01,

    # SeCoND warm-start σ: z0 = σ * randn(nu+nv). Use `0` to restore zeros init (often yields u=v=0).
    stage_game_z0_noise = 1e-2,

    # Box constraints on inner min–max controls: z=[u;v] clamped to [control_lb, control_ub] each SeCoND step (see `solve_stage_game_Q_SeCoND_fd`).
    control_lb = -1.0,
    control_ub = 1.0,

    # One-step state tube: x' = A x + B [u;v] must lie in [x_lo, x_hi] per state component (uses same `x_lo`/`x_hi` as above).
    use_stage_state_bounds = true,
    # Like `thread_safe_clarabel_qp_matrices` in `lq_game_training.jl`: `nothing` → copy QP matrices before Clarabel iff `Threads.nthreads()>1`.
    thread_safe_stage_qp_matrices = nothing,

    # Stage-cost / reward box `[r_min, r_max]`: drives Q clipping (`/(1-γ)`) and TD metric
    # normalization (`/((r_max-r_min)/(1-γ))`). `nothing` → auto from non_lq.
    r_min = nothing,
    r_max = nothing,
    # TD Q clipping: clamp Q(x,u,v) and bootstrap Q(x',u'★,v'★) to [r_min,r_max]/(1-γ).
    # Does not clamp y=ℓ+γQ'.
    td_clip_q_targets = true,

    log_every = 10,

    # Local metrics CSV. Default path: `runs/<problem>_seed…_ns…_gs….csv`.
    csv_enable = true,
    csv_dir = "runs",
    csv_path = nothing,          # `nothing` → auto name under `csv_dir`
    csv_log_every = 1,
    # `false` logs iter 1 pre-update TD (same start across `num_grad_steps`); `true` skips it.
    csv_skip_first_iter = false,
)

# ──────────────────────────────────────────────────────────────────────────────

function make_q_optimizer(q_model, q_lr::Real, q_grad_clip_norm::Real)
    return Flux.setup(
        Optimisers.OptimiserChain(
            Optimisers.ClipNorm(q_grad_clip_norm),
            Optimisers.Adam(q_lr),
        ),
        q_model,
    )
end

"""Walk a `Flux.setup` tree and retune every `OptimiserChain(ClipNorm, Adam)` leaf (keeps Adam momentum state)."""
function retune_q_optimizer_leaves(opt, new_lr::Real, new_clip::Real)
    function map_opt_leaves(x, f)
        x isa Optimisers.Leaf{<:OptimiserChain} && return f(x)
        x isa NamedTuple && return NamedTuple{keys(x)}(map(v -> map_opt_leaves(v, f), Tuple(x)))
        x isa Tuple && return Tuple(map(v -> map_opt_leaves(v, f), x))
        return x
    end
    retune(leaf) = Optimisers.Leaf(
        Optimisers.OptimiserChain(Optimisers.ClipNorm(new_clip), Optimisers.Adam(new_lr)),
        leaf.state,
    )
    return map_opt_leaves(opt, retune)
end

"""
At iteration `it > q_optimizer_change_after_iters`, retune `q_lr` / `q_grad_clip_norm` once.
Returns `(opt, active_lr, active_clip)`; mutates `applied` when the switch happens.
"""
function maybe_apply_q_optimizer_schedule!(opt, cfg, it::Int, applied::Ref{Bool})
    lr = cfg.q_lr
    clip = cfg.q_grad_clip_norm
    milestone = hasproperty(cfg, :q_optimizer_change_after_iters) ? cfg.q_optimizer_change_after_iters : nothing
    if milestone !== nothing
        if applied[]
            lr = cfg.q_lr_after === nothing ? cfg.q_lr : cfg.q_lr_after
            clip = cfg.q_grad_clip_norm_after === nothing ? cfg.q_grad_clip_norm : cfg.q_grad_clip_norm_after
        elseif it > milestone
            applied[] = true
            lr = cfg.q_lr_after === nothing ? cfg.q_lr : cfg.q_lr_after
            clip = cfg.q_grad_clip_norm_after === nothing ? cfg.q_grad_clip_norm : cfg.q_grad_clip_norm_after
            @printf(
                "  [q optimizer] iter %d: q_lr %.6g -> %.6g, q_grad_clip_norm %.6g -> %.6g\n",
                it,
                cfg.q_lr,
                lr,
                cfg.q_grad_clip_norm,
                clip,
            )
            opt = retune_q_optimizer_leaves(opt, lr, clip)
        end
    end
    return opt, lr, clip
end

"""
non_lq stage cost.

State layout: agent 1 position in `x[1:2]`, agent 2 in `x[5:6]`; control penalties use `R1`, `R2`.
"""
function make_stage_cost_non_lq(
    R1::AbstractMatrix,
    R2::AbstractMatrix;
    w::Real,
    wx::Real,
    wy::Real,
)
    wx > 0 || throw(ArgumentError("non_lq_exp_width_x must be positive"))
    wy > 0 || throw(ArgumentError("non_lq_exp_width_y must be positive"))
    return function stage_cost(x, u, v)
        dx = x[1] - x[5]
        dy = x[2] - x[6]
        c_sep = -w * exp(-(dx * dx + dy * dy) / (2 * wx^2))
        c_u = 0.5 * dot(u, R1, u) - 0.5 * dot(v, R2, v)
        return c_sep + c_u
    end
end

"""Built-in or user-provided scalar stage cost `ℓ(x,u,v)` for TD rollouts."""
function resolve_stage_cost(cfg)
    if hasproperty(cfg, :stage_cost) && cfg.stage_cost !== nothing
        return cfg.stage_cost
    end
    return make_stage_cost_non_lq(
        cfg.R1,
        cfg.R2;
        w = cfg.non_lq_exp_weight,
        wx = cfg.non_lq_exp_width_x,
        wy = cfg.non_lq_exp_width_y,
    )
end

"""Min/max of `0.5 * z' * R * z` over a coordinate box.

Evaluates all corners (correct extrema for indefinite quadratics on a box) and the
box-clamped origin (needed for the PSD minimum when `0` lies inside the box).
"""
function _quad_form_bounds_box(R::AbstractMatrix, z_lo::Real, z_ub::Real)
    n = size(R, 1)
    size(R, 2) == n || throw(DimensionMismatch("R must be square"))
    n <= 12 || throw(ArgumentError("box quadratic bound enumeration supports n<=12, got n=$n"))
    z_lo <= z_ub || throw(ArgumentError("z_lo must be <= z_ub"))
    z = zeros(Float64, n)
    qmin = Inf
    qmax = -Inf
    for mask in 0:(2^n - 1)
        for i in 1:n
            z[i] = ((mask >> (i - 1)) & 1) == 1 ? Float64(z_ub) : Float64(z_lo)
        end
        val = 0.5 * dot(z, R, z)
        qmin = min(qmin, val)
        qmax = max(qmax, val)
    end
    # Interior / face critical point at projected 0 (PSD minimum when 0 ∈ box).
    for i in 1:n
        z[i] = clamp(0.0, Float64(z_lo), Float64(z_ub))
    end
    val0 = 0.5 * dot(z, R, z)
    qmin = min(qmin, val0)
    qmax = max(qmax, val0)
    return qmin, qmax
end

"""
Analytic / box bounds on the non_lq stage cost `r(x,u,v)`.

Returns `(r_min, r_max)`. Prefer `resolve_r_bounds` (honors explicit `CONFIG.r_min`/`r_max`).
"""
function resolve_stage_cost_bounds(cfg)
    if hasproperty(cfg, :stage_cost) && cfg.stage_cost !== nothing
        throw(ArgumentError(
            "custom CONFIG.stage_cost: set r_min and r_max explicitly for TD clipping / metrics",
        ))
    end
    u_lo, u_ub = Float64(cfg.control_lb), Float64(cfg.control_ub)
    w = Float64(cfg.non_lq_exp_weight)
    # c_sep ∈ [-w, 0]; c_u = ½u'R1u − ½v'R2v
    cu_min_u, cu_max_u = _quad_form_bounds_box(cfg.R1, u_lo, u_ub)
    cv_min_v, cv_max_v = _quad_form_bounds_box(cfg.R2, u_lo, u_ub)
    # −½v'R2v ranges over [-cv_max, -cv_min]
    cu_min = cu_min_u - cv_max_v
    cu_max = cu_max_u - cv_min_v
    return (-w + cu_min, 0.0 + cu_max)
end

"""
Resolved stage-cost / reward box `(r_min, r_max)`.

Uses explicit `CONFIG.r_min`/`r_max` when set; otherwise auto from non_lq.
"""
function resolve_r_bounds(cfg)
    r_min = hasproperty(cfg, :r_min) ? cfg.r_min : nothing
    r_max = hasproperty(cfg, :r_max) ? cfg.r_max : nothing
    if r_min === nothing || r_max === nothing
        auto_min, auto_max = resolve_stage_cost_bounds(cfg)
        r_min = r_min === nothing ? auto_min : Float64(r_min)
        r_max = r_max === nothing ? auto_max : Float64(r_max)
    else
        r_min = Float64(r_min)
        r_max = Float64(r_max)
    end
    r_min <= r_max || throw(ArgumentError("r_min ($r_min) must be <= r_max ($r_max)"))
    return (r_min, r_max)
end

"""Scale for reporting TD residuals: `(r_max - r_min) / (1 - γ)` (= Q-box diameter)."""
function td_error_norm_scale(r_min::Real, r_max::Real, γ::Real)
    0 <= γ < 1 || throw(ArgumentError("γ must be in [0,1) for TD norm scale, got $γ"))
    r_min <= r_max || throw(ArgumentError("r_min ($r_min) must be <= r_max ($r_max)"))
    s = (r_max - r_min) / (1 - γ)
    s > 0 || throw(ArgumentError("TD norm scale (r_max-r_min)/(1-γ) must be > 0; got r∈[$r_min,$r_max], γ=$γ"))
    return s
end

"""
Q-value clip interval `[Qmin,Qmax] = [r_min,r_max]/(1-γ)` when `td_clip_q_targets`.
Applied to both prediction `Q(x,u,v)` and bootstrap `Q(x',·)` in the TD residual.

Returns `(q_min, q_max)` or `nothing` if clipping is disabled.
"""
function resolve_q_target_bounds(cfg)
    clip = hasproperty(cfg, :td_clip_q_targets) ? cfg.td_clip_q_targets : false
    clip || return nothing
    γ = Float64(cfg.γ)
    0 <= γ < 1 || throw(ArgumentError("γ must be in [0,1) for Q target clipping, got $γ"))
    r_min, r_max = resolve_r_bounds(cfg)
    q_min = r_min / (1 - γ)
    q_max = r_max / (1 - γ)
    q_min <= q_max || throw(ArgumentError("Q clip bounds invalid from r∈[$r_min,$r_max], γ=$γ"))
    return (q_min, q_max)
end

"""`stage_dynamics(x, u, v, rng) -> x′` with optional isotropic Gaussian process noise."""
function _make_noisy_stage_dynamics(ld, dynamics_noise_std::Real)
    σ = Float64(dynamics_noise_std)
    σ >= 0 || throw(ArgumentError("dynamics_noise_std must be >= 0, got $σ"))
    if σ == 0
        return (x, u, v, rng) -> get_next_state(ld, x, [u, v])
    end
    return function (x, u, v, rng)
        return get_next_state(ld, x, [u, v]) .+ σ .* randn(rng, length(x))
    end
end

"""Build stacked double-integrator [`LinearDynamics`](@ref) and a `(x,u,v) -> x′` closure."""
function make_stage_linear_dynamics(cfg)
    ld = build_linear_dynamics(
        [cfg.A_per_player, cfg.A_per_player],
        [cfg.B_per_player, cfg.B_per_player],
    )
    σ = hasproperty(cfg, :dynamics_noise_std) ? cfg.dynamics_noise_std : 0.0
    return _make_noisy_stage_dynamics(ld, σ), ld
end

function make_sample_initial(x_lo::Real, x_hi::Real, nx::Int)
    return function sample_initial_state(rng::AbstractRNG)
        return x_lo .+ (x_hi - x_lo) .* rand(rng, nx)
    end
end

const CSV_METRIC_HEADER = "iteration,td_mean_abs,td_rms,td_loss,mean_norm_u,mean_norm_v,max_norm_u,max_norm_v,min_param,seco_nd_mean_iters,seco_nd_min_iters,seco_nd_max_iters,q_lr,q_grad_clip_norm"

function default_csv_filename(cfg)
    return @sprintf(
        "%s_seed%d_ns%d_gs%d_gamma%g_%s.csv",
        cfg.problem_type,
        cfg.rng_seed,
        cfg.num_samples_per_iter,
        cfg.num_grad_steps,
        cfg.γ,
        Dates.format(Dates.now(), "yyyymmdd_HHMMSS"),
    )
end

function resolve_csv_path(cfg)
    hasproperty(cfg, :csv_enable) && cfg.csv_enable || return nothing
    if hasproperty(cfg, :csv_path) && cfg.csv_path !== nothing
        path = string(cfg.csv_path)
        isempty(path) && throw(ArgumentError("csv_path must be non-empty when csv_enable=true"))
        return path
    end
    dir = hasproperty(cfg, :csv_dir) ? string(cfg.csv_dir) : "runs"
    isempty(dir) && throw(ArgumentError("csv_dir must be non-empty when csv_path is nothing"))
    return joinpath(dir, default_csv_filename(cfg))
end

function init_csv_logger(cfg)
    path = resolve_csv_path(cfg)
    path === nothing && return nothing
    mkpath(dirname(path))
    io = open(path, "w")
    println(io, CSV_METRIC_HEADER)
    flush(io)
    return (; io, path)
end

function csv_log_row!(io::IO, it::Int, mt, td_loss::Real, td_norm::Real, q_lr::Real, q_clip::Real)
    println(
        io,
        @sprintf(
            "%d,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%d,%d,%.10g,%.10g",
            it,
            mt.mean_abs_td_error / td_norm,
            mt.rms_td_error / td_norm,
            td_loss,
            mt.mean_norm_u,
            mt.mean_norm_v,
            mt.max_norm_u,
            mt.max_norm_v,
            mt.min_param,
            mt.mean_seco_nd_iters,
            mt.min_seco_nd_iters,
            mt.max_seco_nd_iters,
            q_lr,
            q_clip,
        ),
    )
    flush(io)
    return nothing
end

function should_log_metrics(it::Int, num_iters::Int, every::Int, skip_first::Bool)
    skip_first && it == 1 && return false
    return every ≤ 1 || it % every == 0 || it == num_iters
end

function validate_sizes(c)
    nx, nu, nv = c.nx, c.nu, c.nv
    if !hasproperty(c, :stage_cost) || c.stage_cost === nothing
        c.non_lq_exp_width_x > 0 || throw(ArgumentError("non_lq_exp_width_x must be positive"))
        c.non_lq_exp_width_y > 0 || throw(ArgumentError("non_lq_exp_width_y must be positive"))
    end
    @assert size(c.R1) == (nu, nu)
    @assert size(c.R2) == (nv, nv)
    @assert iseven(nx) "non_lq uses two stacked agents; nx must be even"
    n_sub = nx ÷ 2
    @assert size(c.A_per_player) == (n_sub, n_sub)
    @assert size(c.B_per_player) == (n_sub, nu)
    @assert nu == nv "non_lq uses the same B_per_player for both players; nu must equal nv"
    @assert c.control_lb < c.control_ub "control_lb must be < control_ub"
    if hasproperty(c, :dynamics_noise_std)
        c.dynamics_noise_std >= 0 || throw(ArgumentError("dynamics_noise_std must be >= 0"))
    end
    @assert c.q_lr > 0
    if hasproperty(c, :q_optimizer_change_after_iters) && c.q_optimizer_change_after_iters !== nothing
        c.q_optimizer_change_after_iters > 0 ||
            throw(ArgumentError("q_optimizer_change_after_iters must be positive"))
        c.q_optimizer_change_after_iters < c.num_iters ||
            @warn "q_optimizer_change_after_iters >= num_iters; schedule will never apply"
        (c.q_lr_after === nothing && c.q_grad_clip_norm_after === nothing) &&
            throw(ArgumentError(
                "set q_lr_after and/or q_grad_clip_norm_after when q_optimizer_change_after_iters is set",
            ))
        c.q_lr_after !== nothing && @assert c.q_lr_after > 0
        c.q_grad_clip_norm_after !== nothing && @assert c.q_grad_clip_norm_after > 0
    end
    @assert hasproperty(c, :picnn_weight_scale) && c.picnn_weight_scale > 0
    @assert !isempty(c.hidden_A) && !isempty(c.hidden_B) && !isempty(c.hidden_C)
    c.use_stage_state_bounds && @assert c.x_lo < c.x_hi "x_lo must be < x_hi when use_stage_state_bounds"
    @assert c.N == 1 "ICNN 1-step TD requires N=1 (got N=$(c.N))"
    @assert c.num_samples_per_iter >= 1
    @assert c.num_grad_steps >= 1
    r_min, r_max = resolve_r_bounds(c)
    td_error_norm_scale(r_min, r_max, Float64(c.γ))  # validate > 0
    if hasproperty(c, :td_clip_q_targets) && c.td_clip_q_targets
        bounds = resolve_q_target_bounds(c)
        bounds === nothing || bounds[1] <= bounds[2] ||
            throw(ArgumentError("resolved Q clip bounds invalid: $bounds"))
    end
end

function learn_q_lq(; cfg = CONFIG)
    validate_sizes(cfg)
    traj_seed = cfg.trajectory_rng_seed === nothing ? cfg.rng_seed : cfg.trajectory_rng_seed

    prev_blas_threads = nothing
    if cfg.parallel_rollouts && cfg.blas_single_thread_parallel && Threads.nthreads() > 1
        prev_blas_threads = BLAS.get_num_threads()
        BLAS.set_num_threads(1)
    end

    rng = MersenneTwister(cfg.rng_seed)
    q_model = create_zero_sum_q_icnn(
        cfg.nx,
        cfg.nu,
        cfg.nv;
        hidden_A = collect(cfg.hidden_A),
        hidden_B = collect(cfg.hidden_B),
        hidden_C = collect(cfg.hidden_C),
        picnn_weight_scale = cfg.picnn_weight_scale,
        picnn_wz_softplus = hasproperty(cfg, :picnn_wz_softplus) ? cfg.picnn_wz_softplus : false,
        picnn_z_gate_softplus =
            hasproperty(cfg, :picnn_z_gate_softplus) ? cfg.picnn_z_gate_softplus : false,
        rng = rng,
    )
    opt = make_q_optimizer(q_model, cfg.q_lr, cfg.q_grad_clip_norm)
    q_opt_schedule_applied = Ref(false)

    ws = ZeroSumQTrajectoryWorkspace(
        Float64,
        cfg.nx,
        cfg.nu,
        cfg.nv,
        cfg.num_samples_per_iter,
        cfg.N,
    )
    stage_cost = resolve_stage_cost(cfg)
    r_min, r_max = resolve_r_bounds(cfg)
    td_norm = td_error_norm_scale(r_min, r_max, Float64(cfg.γ))
    q_clip_bounds = resolve_q_target_bounds(cfg)

    stage_dynamics, ld = make_stage_linear_dynamics(cfg)

    sample_initial_state = make_sample_initial(cfg.x_lo, cfg.x_hi, cfg.nx)

    Bstack = hcat(ld.Bs[1], ld.Bs[2])
    nz_z = cfg.nu + cfg.nv
    stage_qp_A, stage_qp_P =
        cfg.use_stage_state_bounds ? build_stage_state_control_qp_matrices(Bstack, nz_z) : (nothing, nothing)
    state_bound_kw =
        cfg.use_stage_state_bounds ? (;
            A_state = ld.A,
            B_z = Bstack,
            state_lb = fill(Float64(cfg.x_lo), cfg.nx),
            state_ub = fill(Float64(cfg.x_hi), cfg.nx),
            stage_qp_A = stage_qp_A,
            stage_qp_P = stage_qp_P,
            thread_safe_stage_qp_matrices = cfg.thread_safe_stage_qp_matrices,
        ) : NamedTuple()
    stage_solver_kwargs = merge(
        state_bound_kw,
        merge((; z_lb = cfg.control_lb, z_ub = cfg.control_ub), cfg.stage_solver_kwargs),
    )

    csv_lg = init_csv_logger(cfg)
    csv_lg !== nothing && println("[csv] writing ", csv_lg.path)

    println(
        "Learning Q_icnn (PICNN A/B + MatrixNet C); num_samples_per_iter=",
        cfg.num_samples_per_iter,
        ", N=",
        cfg.N,
        ", num_grad_steps=",
        cfg.num_grad_steps,
        ", iters=",
        cfg.num_iters,
        ", q_lr=",
        cfg.q_lr,
        ", grad_clip=",
        cfg.q_grad_clip_norm,
        hasproperty(cfg, :q_optimizer_change_after_iters) && cfg.q_optimizer_change_after_iters !== nothing ?
        ", schedule_after_iter=$(cfg.q_optimizer_change_after_iters)" : "",
        ", picnn_weight_scale=",
        cfg.picnn_weight_scale,
        ", picnn_wz_softplus=",
        hasproperty(cfg, :picnn_wz_softplus) ? cfg.picnn_wz_softplus : false,
        ", picnn_z_gate_softplus=",
        hasproperty(cfg, :picnn_z_gate_softplus) ? cfg.picnn_z_gate_softplus : false,
    )
    println(
        "  r∈[",
        r_min,
        ", ",
        r_max,
        "]  td_norm=(r_max-r_min)/(1-γ)=",
        td_norm,
        "  (td/mean_abs & td/rms reported / td_norm)",
    )
    if q_clip_bounds !== nothing
        println(
            "  td_clip_q_targets=true  Q∈[",
            q_clip_bounds[1],
            ", ",
            q_clip_bounds[2],
            "] = [r_min,r_max]/(1-γ)",
        )
    else
        println("  td_clip_q_targets=false")
    end
    println(
        "  threads=",
        Threads.nthreads(),
        " parallel_rollouts=",
        cfg.parallel_rollouts,
        " trajectory_rng_seed=",
        traj_seed,
        " dynamics_noise_std=",
        hasproperty(cfg, :dynamics_noise_std) ? cfg.dynamics_noise_std : 0.0,
    )

    try
        last_td_loss = 0.0
        active_q_lr, active_q_clip = cfg.q_lr, cfg.q_grad_clip_norm
        for it in 1:(cfg.num_iters)
            opt, active_q_lr, active_q_clip =
                maybe_apply_q_optimizer_schedule!(opt, cfg, it, q_opt_schedule_applied)
            opt, q_model, td_loss, pre_mt = td_zero_sum_q_icnn_step!(
                q_model,
                opt,
                ws,
                rng;
                γ = cfg.γ,
                stage_cost,
                num_grad_steps = cfg.num_grad_steps,
                q_clip_bounds = q_clip_bounds,
                sample_initial_state,
                stage_dynamics,
                stage_solver_kwargs = stage_solver_kwargs,
                stage_game_z0_noise = cfg.stage_game_z0_noise,
                parallel_rollouts = cfg.parallel_rollouts,
                trajectory_rng_seed = traj_seed,
            )
            last_td_loss = td_loss
            if cfg.log_every > 0 && (it % cfg.log_every == 0 || it == cfg.num_iters)
                println(
                    @sprintf(
                        "  iter %5d  mean|δ|_pre/td_norm = %.6g  td_loss_post = %.6g  min|θ| = %.4g  SeCoND mean/min/max = %.2f / %d / %d",
                        it,
                        pre_mt.mean_abs_td_error / td_norm,
                        last_td_loss,
                        pre_mt.min_param,
                        pre_mt.mean_seco_nd_iters,
                        pre_mt.min_seco_nd_iters,
                        pre_mt.max_seco_nd_iters,
                    ),
                )
            end
            if csv_lg !== nothing
                skip_first = hasproperty(cfg, :csv_skip_first_iter) && cfg.csv_skip_first_iter
                csv_every = hasproperty(cfg, :csv_log_every) ? cfg.csv_log_every : 1
                if should_log_metrics(it, cfg.num_iters, csv_every, skip_first)
                    csv_log_row!(csv_lg.io, it, pre_mt, last_td_loss, td_norm, active_q_lr, active_q_clip)
                end
            end
        end
    finally
        if prev_blas_threads !== nothing
            BLAS.set_num_threads(prev_blas_threads)
        end
        if csv_lg !== nothing
            close(csv_lg.io)
        end
    end

    return (; q_model, opt, ws, cfg)
end

function main()
    learn_q_lq()
    println("Done.")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
