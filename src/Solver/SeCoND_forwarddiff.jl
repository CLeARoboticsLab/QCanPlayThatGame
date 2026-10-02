# Stage-game SeCoND with ForwardDiff ∇f / ∇²f, control box, and optional next-state Clarabel projection.

export SeCoND_stage_game_forwarddiff,
    project_stage_z_state_control_box!,
    build_stage_state_control_qp_matrices,
    stage_z_feasible

using Base.Threads: Threads
using Clarabel
using ForwardDiff
using LinearAlgebra: I, norm
using SparseArrays: SparseMatrixCSC, sparse

"""Project stacked decision `update` into `[z_lb, z_ub]` elementwise; scalars or length-`nz` vectors."""
function _project_stage_z!(z_dest::AbstractVector, update::AbstractVector, z_lb, z_ub)
    if z_lb === nothing && z_ub === nothing
        z_dest .= update
        return z_dest
    end
    (z_lb !== nothing && z_ub !== nothing) ||
        throw(ArgumentError("SeCoND stage game: pass both z_lb and z_ub, or neither (unconstrained)."))
    z_dest .= clamp.(update, z_lb, z_ub)
    return z_dest
end

function _vec_bounds(val, n::Int, ::Type{T}) where {T<:Real}
    if val isa AbstractVector
        length(val) == n || throw(ArgumentError("bound vector length $(length(val)) ≠ $n"))
        return collect(T, val)
    end
    return fill(T(val), n)
end

"""
    build_stage_state_control_qp_matrices(B_z, nz::Integer)

Fixed QP matrices `(A_ineq, P)` for [`project_stage_z_state_control_box!`](@ref) with constant `B_z`:
`A_ineq = [B; -B; I; -I]`, `P = 2I`, both as sparse `SparseMatrixCSC`. Pass as `qp_A`, `qp_P`
to avoid rebuilding each call.
"""
function build_stage_state_control_qp_matrices(B_z::AbstractMatrix, nz::Integer)
    nx = size(B_z, 1)
    size(B_z, 2) == nz ||
        throw(DimensionMismatch("B_z has $(size(B_z,2)) columns, expected nz=$nz"))
    T = float(eltype(B_z))
    B = Matrix{T}(B_z)
    I_nz = Matrix{T}(I, nz, nz)
    A_ineq = sparse(vcat(B, -B, I_nz, -I_nz))
    P = sparse(T(2.0) * I, nz, nz)
    return A_ineq, P
end

"""
    project_stage_z_state_control_box!(z_out, z0, A, B, x, state_lb, state_ub, z_lb, z_ub; kwargs...)

Closest feasible Euclidean projection of `z0` onto `state_lb .<= A*x+B*z .<= state_ub` and `z_lb .<= z .<= z_ub`.

Keywords:
- `qp_A`, `qp_P`: from [`build_stage_state_control_qp_matrices`](@ref); omit to assemble from `B` each call.
- `thread_safe_qp_matrices`: copy `P` and `A` before `setup!` when `true` (default `Threads.nthreads() > 1`).
"""
function project_stage_z_state_control_box!(
    z_out::AbstractVector{T},
    z0::AbstractVector{T},
    A::AbstractMatrix,
    B::AbstractMatrix,
    x::AbstractVector,
    state_lb,
    state_ub,
    z_lb,
    z_ub;
    qp_A = nothing,
    qp_P = nothing,
    thread_safe_qp_matrices::Bool = Threads.nthreads() > 1,
) where {T<:Real}
    nx = size(A, 1)
    nz = size(B, 2)
    @assert size(A, 2) == length(x) && length(z0) == nz && size(B, 1) == nx "state projection dimension mismatch"
    rhs = A * x
    sl = _vec_bounds(state_lb, nx, T)
    su = _vec_bounds(state_ub, nx, T)
    zlv = _vec_bounds(z_lb, nz, T)
    zuv = _vec_bounds(z_ub, nz, T)

    x_next = rhs + B * z0
    if all(sl .<= x_next .<= su) && all(zlv .<= z0 .<= zuv)
        z_out .= z0
        return z_out
    end

    I_nz = Matrix{T}(I, nz, nz)
    A_qp =
        qp_A === nothing ? sparse(vcat(Matrix(B), Matrix(-B), I_nz, -I_nz)) : qp_A
    P_mat = qp_P === nothing ? sparse(T(2.0) * I, nz, nz) : qp_P
    if qp_A !== nothing
        size(A_qp, 1) == 2 * nx + 2 * nz ||
            throw(DimensionMismatch("qp_A row count $(size(A_qp,1)) ≠ $(2*nx+2*nz)"))
        size(A_qp, 2) == nz || throw(DimensionMismatch("qp_A columns $(size(A_qp,2)) ≠ $nz"))
    end
    if qp_P !== nothing
        size(P_mat, 1) == nz || throw(DimensionMismatch("qp_P dimension mismatch"))
    end

    b_qp = vcat(su .- rhs, rhs .- sl, zuv, .-zlv)
    q = -2.0 .* z0

    P_setup = thread_safe_qp_matrices ? copy(P_mat) : P_mat
    A_setup = thread_safe_qp_matrices ? copy(A_qp) : A_qp
    P_setup isa SparseMatrixCSC || (P_setup = sparse(P_setup))
    A_setup isa SparseMatrixCSC || (A_setup = sparse(A_setup))
    cones = [Clarabel.NonnegativeConeT(length(b_qp))]
    settings = Clarabel.Settings(verbose = false)
    solver = Clarabel.Solver()
    Clarabel.setup!(solver, P_setup, Vector(q), A_setup, collect(b_qp), cones, settings)
    Clarabel.solve!(solver)
    if solver.solution.status == Clarabel.SOLVED
        z_out .= solver.solution.x
    else
        _project_stage_z!(z_out, z0, z_lb, z_ub)
    end
    return z_out
end

function _state_projection_active(A_state, B_z, state_lb, state_ub, stage_x)
    return A_state !== nothing &&
           B_z !== nothing &&
           state_lb !== nothing &&
           state_ub !== nothing &&
           stage_x !== nothing
end

"""
    stage_z_feasible(x, z; z_lb, z_ub, A_state, B_z, state_lb, state_ub)

Return `true` if stacked controls `z` lie in the control box (when `z_lb`/`z_ub` are set) and, when a
one-step state tube is configured, if `state_lb .<= A_state * x + B_z * z .<= state_ub`.
"""
function stage_z_feasible(
    x::AbstractVector,
    z::AbstractVector;
    z_lb = nothing,
    z_ub = nothing,
    A_state = nothing,
    B_z = nothing,
    state_lb = nothing,
    state_ub = nothing,
)
    nz = length(z)
    T = float(eltype(z))
    if z_lb !== nothing && z_ub !== nothing
        zlv = _vec_bounds(z_lb, nz, T)
        zuv = _vec_bounds(z_ub, nz, T)
        for i in 1:nz
            (zlv[i] <= z[i] <= zuv[i]) || return false
        end
    end
    if _state_projection_active(A_state, B_z, state_lb, state_ub, x)
        xn = A_state * x + B_z * z
        nx = length(xn)
        sl = _vec_bounds(state_lb, nx, T)
        su = _vec_bounds(state_ub, nx, T)
        for i in 1:nx
            (sl[i] <= xn[i] <= su[i]) || return false
        end
    end
    return true
end

function _project_stage_step!(
    z_dest::AbstractVector,
    update::AbstractVector,
    z_lb,
    z_ub,
    A_state,
    B_z,
    state_lb,
    state_ub,
    stage_x;
    stage_qp_A = nothing,
    stage_qp_P = nothing,
    thread_safe_qp_matrices::Bool = Threads.nthreads() > 1,
)
    if _state_projection_active(A_state, B_z, state_lb, state_ub, stage_x)
        project_stage_z_state_control_box!(
            z_dest,
            update,
            A_state,
            B_z,
            stage_x,
            state_lb,
            state_ub,
            z_lb,
            z_ub;
            qp_A = stage_qp_A,
            qp_P = stage_qp_P,
            thread_safe_qp_matrices = thread_safe_qp_matrices,
        )
    else
        _project_stage_z!(z_dest, update, z_lb, z_ub)
    end
    return z_dest
end

"""
    SeCoND_stage_game_forwarddiff(objective, initial_guess, n_minimizer; kwargs...)

Single-stage zero-sum game in stacked `z = [u; v]`: player one minimizes and player two maximizes
`objective(z)`. First `n_minimizer` components are minimizer controls.

Optional box: `z_lb`, `z_ub`. Optional next-state box: `stage_x`, `A_state`, `B_z`, `state_lb`,
`state_ub` with `x' = A_state * stage_x + B_z * z`, projected by Clarabel.

Typical objective: `Q_icnn(model, x, u, v)` with fixed `x`.
"""
function SeCoND_stage_game_forwarddiff(
    objective::Function,
    initial_guess::AbstractVector,
    n_minimizer::Integer;
    step_size::Float64,
    max_iterations::Int,
    tol::Float64 = 1e-6,
    ball_tol::Float64 = 1e-9,
    quasi_coeff::Float64 = 1e-3,
    z_lb = nothing,
    z_ub = nothing,
    stage_x = nothing,
    A_state = nothing,
    B_z = nothing,
    state_lb = nothing,
    state_ub = nothing,
    stage_qp_A = nothing,
    stage_qp_P = nothing,
    thread_safe_stage_qp_matrices::Union{Nothing,Bool} = nothing,
    verbose::Bool = true,
    iterations_out::Union{Nothing,Ref{Int}} = nothing,
)
    ts_qp =
        thread_safe_stage_qp_matrices === nothing ? (Threads.nthreads() > 1) : thread_safe_stage_qp_matrices
    z = copy(initial_guess)
    z_old = copy(z)
    error = 1.0
    iterations = 0
    while error > tol && iterations < max_iterations
        ω = ForwardDiff.gradient(objective, z)
        J = ForwardDiff.hessian(objective, z)
        @views ω[n_minimizer+1:end] .*= -1
        @views J[n_minimizer+1:end, :] .*= -1
        if norm(z - z_old) < ball_tol
            update = SeCoND_regularization(ω, J, n_minimizer)
            update = z - step_size * update
        else
            update = quasi_newton_update(ω, J, quasi_coeff)
            update = z - step_size * update
        end
        z_old .= z
        _project_stage_step!(
            z,
            update,
            z_lb,
            z_ub,
            A_state,
            B_z,
            state_lb,
            state_ub,
            stage_x;
            stage_qp_A,
            stage_qp_P,
            thread_safe_qp_matrices = ts_qp,
        )
        error = norm(z - z_old)
        iterations += 1
    end
    if error > tol
        verbose && println("SeCoND (stage game) did not converge, reached max iterations of $max_iterations")
    else
        verbose && println("SeCoND (stage game) converged in $iterations iterations")
    end
    if iterations_out !== nothing
        iterations_out[] = iterations
    end
    return z
end
