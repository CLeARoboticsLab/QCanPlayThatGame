using Random: AbstractRNG, MersenneTwister, Random, rand
using Statistics: mean
using Zygote: Zygote, ignore_derivatives
using Optimisers: Optimisers

"""
    solve_stage_game_Q_icnn_SeCoND_fd(x, model, nu, nv; z0, z_lb, z_ub, kwargs...)

Solve `min_u max_v Q(x,u,v)` with [`Q_icnn`](@ref) and [`SeCoND_stage_game_forwarddiff`](@ref).
Same constraints / projection interface as [`solve_stage_game_Q_SeCoND_fd`](@ref).
"""
function solve_stage_game_Q_icnn_SeCoND_fd(
    x,
    model::ZeroSumQICNN,
    nu::Int,
    nv::Int;
    z0 = nothing,
    z_lb = nothing,
    z_ub = nothing,
    A_state = nothing,
    B_z = nothing,
    state_lb = nothing,
    state_ub = nothing,
    stage_qp_A = nothing,
    stage_qp_P = nothing,
    thread_safe_stage_qp_matrices::Union{Nothing,Bool} = nothing,
    step_size::Float64 = 0.1,
    max_iterations::Int = 1000,
    kwargs...,
)
    nz = nu + nv
    z_init = z0 === nothing ? zeros(Float64, nz) : collect(z0)
    length(z_init) == nz || throw(ArgumentError("length(z0) must be nu+nv = $nz"))
    if z_lb !== nothing && z_ub !== nothing
        z_init .= clamp.(z_init, z_lb, z_ub)
    elseif z_lb !== nothing || z_ub !== nothing
        throw(ArgumentError("solve_stage_game_Q_icnn_SeCoND_fd: set both z_lb and z_ub, or neither."))
    end
    if _state_projection_active(A_state, B_z, state_lb, state_ub, x)
        (z_lb !== nothing && z_ub !== nothing) ||
            throw(ArgumentError("next-state projection requires both z_lb and z_ub (control box)."))
        ts_qp =
            thread_safe_stage_qp_matrices === nothing ? (Threads.nthreads() > 1) :
            thread_safe_stage_qp_matrices
        project_stage_z_state_control_box!(
            z_init,
            copy(z_init),
            A_state,
            B_z,
            x,
            state_lb,
            state_ub,
            z_lb,
            z_ub;
            qp_A = stage_qp_A,
            qp_P = stage_qp_P,
            thread_safe_qp_matrices = ts_qp,
        )
    end
    objective(z::AbstractVector) =
        Q_icnn(model, x, view(z, 1:nu), view(z, nu + 1:nz))
    z = SeCoND_stage_game_forwarddiff(
        objective,
        z_init,
        nu;
        step_size,
        max_iterations,
        z_lb,
        z_ub,
        stage_x = x,
        A_state,
        B_z,
        state_lb,
        state_ub,
        stage_qp_A,
        stage_qp_P,
        thread_safe_stage_qp_matrices = thread_safe_stage_qp_matrices,
        kwargs...,
    )
    u = z[1:nu]
    v = z[nu + 1:nz]
    return u, v, z
end

_solver_kw(nt::NamedTuple, key::Symbol, default) =
    hasproperty(nt, key) ? getproperty(nt, key) : default

function _stage_rollout_constraints_for_feasibility(sk::NamedTuple)
    z_lb = _solver_kw(sk, :z_lb, nothing)
    z_ub = _solver_kw(sk, :z_ub, nothing)
    A_state = _solver_kw(sk, :A_state, nothing)
    B_z = _solver_kw(sk, :B_z, nothing)
    state_lb = _solver_kw(sk, :state_lb, nothing)
    state_ub = _solver_kw(sk, :state_ub, nothing)
    need_check =
        (z_lb !== nothing && z_ub !== nothing) ||
        (A_state !== nothing &&
         B_z !== nothing &&
         state_lb !== nothing &&
         state_ub !== nothing)
    return need_check, z_lb, z_ub, A_state, B_z, state_lb, state_ub
end

function _control_box_bounds(z_lb, z_ub, nu::Int, nv::Int)
    nz = nu + nv
    lb = z_lb === nothing ? nothing : (z_lb isa Real ? fill(Float64(z_lb), nz) : collect(Float64, z_lb))
    ub = z_ub === nothing ? nothing : (z_ub isa Real ? fill(Float64(z_ub), nz) : collect(Float64, z_ub))
    lb === nothing && ub === nothing && return nothing, nothing, nothing, nothing
    lb === nothing && throw(ArgumentError("z_lb required when z_ub is set for initial control sampling"))
    ub === nothing && throw(ArgumentError("z_ub required when z_lb is set for initial control sampling"))
    length(lb) == length(ub) == nz ||
        throw(ArgumentError("z_lb/z_ub length must be nu+nv=$(nz), got $(length(lb)), $(length(ub))"))
    return view(lb, 1:nu), view(lb, nu + 1:nz), view(ub, 1:nu), view(ub, nu + 1:nz)
end

function _default_sample_initial_controls(
    rng::AbstractRNG,
    nu::Int,
    nv::Int,
    z_lb,
    z_ub,
)
    u_lb, v_lb, u_ub, v_ub = _control_box_bounds(z_lb, z_ub, nu, nv)
    if u_lb === nothing
        return randn(rng, nu), randn(rng, nv)
    end
    u = u_lb .+ (u_ub .- u_lb) .* rand(rng, nu)
    v = v_lb .+ (v_ub .- v_lb) .* rand(rng, nv)
    return u, v
end

"""
Collect one 1-step transition: sample `(x,u,v)`, dynamics → `x'`, SeCoND at `x'` → `(u',v')`.

Requires workspace horizon `N == 1`. Stores exploratory actions in `us`/`vs` and Nash bootstrap
actions in `us_next`/`vs_next`.
"""
function _collect_one_sample_icnn!(
    m::Integer,
    rng_m::AbstractRNG,
    model::ZeroSumQICNN,
    xs,
    us,
    vs,
    us_next,
    vs_next,
    seco_nd_iters,
    sample_initial_state,
    sample_initial_controls,
    stage_dynamics,
    nu::Int,
    nv::Int,
    stage_solver_kwargs::NamedTuple,
    stage_game_z0_noise::Real,
    max_initial_state_resamples::Int,
    initial_state_resample_counter::Union{Nothing,Base.Threads.Atomic{Int}},
)
    nz = nu + nv
    need_check, z_lb, z_ub, A_state, B_z, state_lb, state_ub =
        _stage_rollout_constraints_for_feasibility(stage_solver_kwargs)
    default_initial_controls(rng) =
        _default_sample_initial_controls(rng, nu, nv, z_lb, z_ub)
    draw_initial_controls(rng) =
        sample_initial_controls === nothing ? default_initial_controls(rng) :
        sample_initial_controls(rng)
    attempt = 0
    while true
        attempt += 1
        if need_check && attempt > max_initial_state_resamples
            error(
                "rollout_zero_sum_q_icnn!: could not obtain feasible constrained stage actions after $(max_initial_state_resamples) initial-state resamples",
            )
        end
        x = sample_initial_state(rng_m)
        u, v = draw_initial_controls(rng_m)
        z = vcat(u, v)
        if need_check && !stage_z_feasible(x, z; z_lb, z_ub, A_state, B_z, state_lb, state_ub)
            continue
        end

        x_next = stage_dynamics(x, u, v, rng_m)
        if need_check && state_lb !== nothing && state_ub !== nothing
            # Process noise can leave the state tube; resample (x,u,v) if so.
            out_of_box = false
            for i in eachindex(x_next)
                if !(state_lb[i] <= x_next[i] <= state_ub[i])
                    out_of_box = true
                    break
                end
            end
            out_of_box && continue
        end
        z0_default =
            stage_game_z0_noise > 0 ? stage_game_z0_noise .* randn(rng_m, nz) : zeros(Float64, nz)
        sk = merge((; z0 = z0_default), stage_solver_kwargs)
        iters_ref = Ref(0)
        sk = merge(sk, (; iterations_out = iters_ref))
        u_star, v_star, z_star = solve_stage_game_Q_icnn_SeCoND_fd(x_next, model, nu, nv; sk...)
        if need_check &&
           !stage_z_feasible(x_next, z_star; z_lb, z_ub, A_state, B_z, state_lb, state_ub)
            continue
        end

        @views xs[:, 1, m] .= x
        @views xs[:, 2, m] .= x_next
        @views us[:, 1, m] .= u
        @views vs[:, 1, m] .= v
        @views us_next[:, m] .= u_star
        @views vs_next[:, m] .= v_star
        seco_nd_iters[1, m] = iters_ref[]

        if need_check && initial_state_resample_counter !== nothing
            Base.Threads.atomic_add!(initial_state_resample_counter, attempt - 1)
        end
        return nothing
    end
end

"""
    rollout_zero_sum_q_icnn!(rng, model, workspace; rollout keywords...)

Collect `M` independent **1-step** transitions (`N` must be 1):

1. Sample `(x, u, v)` (exploratory controls).
2. `x' = stage_dynamics(x, u, v, rng)` (optional process noise).
3. Solve `min_u' max_v' Q(x',·)` with SeCoND; store Nash actions in `workspace.us_next` /
   `workspace.vs_next` for the TD bootstrap target.
"""
function rollout_zero_sum_q_icnn!(
    rng::AbstractRNG,
    model::ZeroSumQICNN,
    ws::ZeroSumQTrajectoryWorkspace;
    sample_initial_state,
    sample_initial_controls = nothing,
    stage_dynamics,
    stage_solver_kwargs::NamedTuple = NamedTuple(),
    stage_game_z0_noise::Real = 1e-3,
    parallel_rollouts::Bool = false,
    trajectory_rng_seed::Union{Nothing,Integer} = nothing,
    max_initial_state_resamples::Int = 100,
    initial_state_resample_counter::Union{Nothing,Base.Threads.Atomic{Int}} = nothing,
)
    xs, us, vs, us_next, vs_next, seco_nd_iters =
        ws.xs, ws.us, ws.vs, ws.us_next, ws.vs_next, ws.seco_nd_iters
    _, Np1, M = size(xs)
    N = Np1 - 1
    N == 1 || throw(ArgumentError("ICNN 1-step TD requires workspace N=1, got N=$N"))
    nu, nv = model.nu, model.nv
    @assert size(us, 2) == 1 && size(vs, 2) == 1 && size(seco_nd_iters) == (1, M)
    @assert size(us_next) == (nu, M) && size(vs_next) == (nv, M)

    if parallel_rollouts && M > 1 && Threads.nthreads() > 1
        # Salt with `rng` each collect so outer iters get fresh batches (fixed
        # trajectory_rng_seed alone would replay the same (x,u,v) forever).
        seed_u =
            trajectory_rng_seed === nothing ? UInt64(0x94D495FA96FD7047) : UInt64(trajectory_rng_seed)
        collect_salt = rand(rng, UInt64)
        Threads.@threads for m in 1:M
            rng_m = MersenneTwister(hash(hash(seed_u, UInt64(m)), collect_salt))
            _collect_one_sample_icnn!(
                m,
                rng_m,
                model,
                xs,
                us,
                vs,
                us_next,
                vs_next,
                seco_nd_iters,
                sample_initial_state,
                sample_initial_controls,
                stage_dynamics,
                nu,
                nv,
                stage_solver_kwargs,
                stage_game_z0_noise,
                max_initial_state_resamples,
                initial_state_resample_counter,
            )
        end
    else
        for m in 1:M
            _collect_one_sample_icnn!(
                m,
                rng,
                model,
                xs,
                us,
                vs,
                us_next,
                vs_next,
                seco_nd_iters,
                sample_initial_state,
                sample_initial_controls,
                stage_dynamics,
                nu,
                nv,
                stage_solver_kwargs,
                stage_game_z0_noise,
                max_initial_state_resamples,
                initial_state_resample_counter,
            )
        end
    end
    return ws
end

"""
    td_icnn_mse_loss(model, workspace, γ, stage_cost, q_clip_bounds)

Semi-gradient 1-step TD loss over `M` samples with optional Q-value clipping:
`mean_m (clip(Q(x,u,v)) - y)²` where `y = ℓ + γ clip(Q(x',u'★,v'★))`.
Only Q outputs are clipped; `y` is never clamped. Bootstrap `Q(x',·)` is stop-grad.

Positional form is required for Zygote (kwargs AD segfaults on Julia 1.12 + Zygote).
"""
function td_icnn_mse_loss(
    model::ZeroSumQICNN,
    ws::ZeroSumQTrajectoryWorkspace,
    γ::Real,
    stage_cost,
    q_clip_bounds::Union{Nothing,Tuple{<:Real,<:Real}},
)
    xs, us, vs, us_next, vs_next = ws.xs, ws.us, ws.vs, ws.us_next, ws.vs_next
    _, Np1, M = size(xs)
    N = Np1 - 1
    N == 1 || throw(ArgumentError("ICNN 1-step TD loss requires workspace N=1, got N=$N"))
    q_min, q_max = q_clip_bounds === nothing ? (-Inf, Inf) : q_clip_bounds
    loss = 0.0
    for m in 1:M
        x_t = xs[:, 1, m]
        u_t = us[:, 1, m]
        v_t = vs[:, 1, m]
        x_tp = xs[:, 2, m]
        q_t = clamp(Q_icnn(model, x_t, u_t, v_t), q_min, q_max)
        q_tp = ignore_derivatives(clamp(Q_icnn(model, x_tp, us_next[:, m], vs_next[:, m]), q_min, q_max))
        ℓ = stage_cost(x_t, u_t, v_t)
        y = ℓ + γ * q_tp
        δ = q_t - y
        loss += abs2(δ)
    end
    return loss / M
end

function td_icnn_mse_loss(
    model::ZeroSumQICNN,
    ws::ZeroSumQTrajectoryWorkspace;
    γ::Real,
    stage_cost,
    q_clip_bounds::Union{Nothing,Tuple{<:Real,<:Real}} = nothing,
)
    return td_icnn_mse_loss(model, ws, γ, stage_cost, q_clip_bounds)
end

"""True if every numeric leaf in a gradient tree is finite."""
function icnn_gradients_all_finite(grad_tree)
    grad_tree isa Number && return isfinite(grad_tree)
    grad_tree isa AbstractArray{<:Number} && return all(isfinite, grad_tree)
    grad_tree isa Tuple && return all(icnn_gradients_all_finite, grad_tree)
    grad_tree isa NamedTuple && return all(icnn_gradients_all_finite, values(grad_tree))
    grad_tree isa AbstractDict && return all(icnn_gradients_all_finite, values(grad_tree))
    return true
end

"""Collect paths in a Zygote grad tree that contain non-finite values (for skip diagnostics)."""
function icnn_nonfinite_grad_paths(grad_tree; max_paths::Int = 8)
    paths = String[]
    function walk(node, path::String)
        length(paths) >= max_paths && return
        if node isa Number
            if !isfinite(node)
                push!(paths, "$path=$(node)")
            end
        elseif node isa AbstractArray{<:Number}
            n_nan = count(isnan, node)
            n_inf = count(isinf, node)
            if n_nan > 0 || n_inf > 0
                push!(paths, "$path: size=$(size(node)) nan=$n_nan inf=$n_inf")
            end
        elseif node isa NamedTuple
            for (k, v) in pairs(node)
                walk(v, isempty(path) ? String(k) : "$path.$k")
                length(paths) >= max_paths && return
            end
        elseif node isa Tuple
            for (i, v) in enumerate(node)
                walk(v, "$path[$i]")
                length(paths) >= max_paths && return
            end
        elseif node isa AbstractDict
            for (k, v) in node
                walk(v, "$path[$k]")
                length(paths) >= max_paths && return
            end
        elseif node === nothing
            return
        end
    end
    walk(grad_tree, "")
    return paths
end

function icnn_td_skip_reason(loss, grad_tree)
    loss_bad = !isfinite(loss)
    grads_bad = grad_tree === nothing || !icnn_gradients_all_finite(grad_tree)
    parts = String[]
    if loss_bad
        push!(parts, "non-finite loss=$loss")
    else
        push!(parts, "finite loss=$loss")
    end
    if grads_bad
        if grad_tree === nothing
            push!(parts, "gradients=nothing")
        else
            bad = icnn_nonfinite_grad_paths(grad_tree)
            detail = isempty(bad) ? "non-finite gradients" : "non-finite gradients at [" * join(bad, "; ") * "]"
            push!(parts, detail)
        end
    else
        push!(parts, "gradients finite")
    end
    return join(parts, "; ")
end

"""Fill `workspace.td_errors` from the current `model` and stored 1-step samples (no AD)."""
function fill_td_errors_icnn!(
    model::ZeroSumQICNN,
    ws::ZeroSumQTrajectoryWorkspace;
    γ::Real,
    stage_cost,
    q_clip_bounds::Union{Nothing,Tuple{<:Real,<:Real}} = nothing,
)
    xs, us, vs, us_next, vs_next, δbuf =
        ws.xs, ws.us, ws.vs, ws.us_next, ws.vs_next, ws.td_errors
    _, Np1, M = size(xs)
    N = Np1 - 1
    N == 1 || throw(ArgumentError("ICNN 1-step TD errors require workspace N=1, got N=$N"))
    q_min, q_max = q_clip_bounds === nothing ? (-Inf, Inf) : q_clip_bounds
    for m in 1:M
        x_t = xs[:, 1, m]
        u_t = us[:, 1, m]
        v_t = vs[:, 1, m]
        x_tp = xs[:, 2, m]
        q_t = clamp(Q_icnn(model, x_t, u_t, v_t), q_min, q_max)
        q_tp = clamp(Q_icnn(model, x_tp, us_next[:, m], vs_next[:, m]), q_min, q_max)
        ℓ = stage_cost(x_t, u_t, v_t)
        y = ℓ + γ * q_tp
        δbuf[1, m] = q_t - y
    end
    return ws
end

"""
    td_icnn_gradient_update!(model, opt, workspace; γ, stage_cost, q_clip_bounds)

Apply one Adam step on the TD MSE loss after samples are stored in `workspace`.

If the loss or gradients are non-finite, the optimizer step is **skipped** (returns unchanged
`opt`, `model`) and a diagnostic is printed so long runs do not abort after W&B sync.
Does not refresh `workspace.td_errors` (caller may keep pre-update residuals for logging).
"""
function td_icnn_gradient_update!(
    model::ZeroSumQICNN,
    opt,
    ws::ZeroSumQTrajectoryWorkspace;
    γ::Real,
    stage_cost,
    q_clip_bounds::Union{Nothing,Tuple{<:Real,<:Real}} = nothing,
)
    loss_fn(m) = td_icnn_mse_loss(m, ws, γ, stage_cost, q_clip_bounds)
    l, gs = Zygote.withgradient(loss_fn, model)
    g = gs[1]
    if !isfinite(l) || g === nothing || !icnn_gradients_all_finite(g)
        println(stderr, "[ICNN TD] Skipping update: ", icnn_td_skip_reason(l, g))
        return opt, model, l
    end
    opt, model = Optimisers.update!(opt, model, g)
    return opt, model, l
end

"""
    td_zero_sum_q_icnn_step!(model, opt, workspace, rng; γ, stage_cost, num_grad_steps, ...)

Collect `M` 1-step transitions (exploratory `(u,v)`, SeCoND Nash at `x'` for targets), then take
`num_grad_steps` semi-gradient Adam updates on that frozen batch.

Returns `(opt, model, post_loss, pre_metrics)` where `pre_metrics` are TD / rollout stats on the
batch **before** Adam (so different `num_grad_steps` share the same pre-update TD curve start).
`post_loss` is the MSE after the last Adam step.
"""
function td_zero_sum_q_icnn_step!(
    model::ZeroSumQICNN,
    opt,
    ws::ZeroSumQTrajectoryWorkspace,
    rng::AbstractRNG;
    γ::Real,
    stage_cost,
    num_grad_steps::Integer = 1,
    q_clip_bounds::Union{Nothing,Tuple{<:Real,<:Real}} = nothing,
    sample_initial_state,
    sample_initial_controls = nothing,
    stage_dynamics,
    stage_solver_kwargs::NamedTuple = NamedTuple(),
    stage_game_z0_noise::Real = 1e-3,
    parallel_rollouts::Bool = false,
    trajectory_rng_seed::Union{Nothing,Integer} = nothing,
    max_initial_state_resamples::Int = 100,
    initial_state_resample_counter::Union{Nothing,Base.Threads.Atomic{Int}} = nothing,
)
    num_grad_steps >= 1 || throw(ArgumentError("num_grad_steps must be >= 1, got $num_grad_steps"))
    rollout_zero_sum_q_icnn!(
        rng,
        model,
        ws;
        sample_initial_state,
        sample_initial_controls,
        stage_dynamics,
        stage_solver_kwargs,
        stage_game_z0_noise,
        parallel_rollouts,
        trajectory_rng_seed,
        max_initial_state_resamples,
        initial_state_resample_counter,
    )
    fill_td_errors_icnn!(
        model,
        ws;
        γ,
        stage_cost,
        q_clip_bounds,
    )
    pre_metrics = rollout_training_metrics_icnn(ws, model)
    loss = 0.0
    for _ in 1:num_grad_steps
        opt, model, loss = td_icnn_gradient_update!(
            model,
            opt,
            ws;
            γ,
            stage_cost,
            q_clip_bounds,
        )
    end
    return opt, model, loss, pre_metrics
end

"""Metrics from last ICNN sample batch (uses `workspace.td_errors`)."""
function rollout_training_metrics_icnn(ws::ZeroSumQTrajectoryWorkspace, model::ZeroSumQICNN)
    δ = ws.td_errors
    mean_abs = mean(abs, δ)
    rms = sqrt(mean(abs2, δ))
    us, vs = ws.us, ws.vs
    Nst, M = size(us, 2), size(us, 3)
    su = Vector{Float64}(undef, M * Nst)
    sv = Vector{Float64}(undef, M * Nst)
    k = 0
    for m in 1:M, t in 1:Nst
        k += 1
        su[k] = norm(us[:, t, m])
        sv[k] = norm(vs[:, t, m])
    end
    it = ws.seco_nd_iters
    min_param = minimum(minimum(abs, p) for p in Flux.trainables(model))
    return (
        mean_abs_td_error = mean_abs,
        rms_td_error = rms,
        mean_norm_u = mean(su),
        mean_norm_v = mean(sv),
        max_norm_u = maximum(su),
        max_norm_v = maximum(sv),
        min_param = min_param,
        mean_seco_nd_iters = mean(it),
        min_seco_nd_iters = minimum(it),
        max_seco_nd_iters = maximum(it),
    )
end
