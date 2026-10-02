using ForwardDiff: ForwardDiff
using LinearAlgebra: Symmetric, dot, eigvals
using Random: AbstractRNG, Random

# Zero-sum Q representation:
#   Q(x, u, v) = A(x, u) - B(x, v) + u' C(x) v
#
# - `A`, `B`: PICNN (Amos et al. §3.2, Eq. 3) — convex in `u` / `v`, **not** convex in `x`
# - `C`: MatrixNet — arbitrary dependence on `x`

"""
    ZeroSumQICNN{A,B,C}

Decomposed Q-network for zero-sum stage games.

# Fields
- `A_net`: `A(x, u)` — PICNN, convex in `u` only
- `B_net`: `B(x, v)` — PICNN, convex in `v` only (`-B` concave in `v`)
- `C_net`: `C(x) ∈ ℝ^{nu×nv}` — matrix-valued MLP
- `nu`, `nv`, `nx`: dimensions
"""
struct ZeroSumQICNN{A,B,C}
    A_net::A
    B_net::B
    C_net::C
    nx::Int
    nu::Int
    nv::Int
end

Flux.@layer ZeroSumQICNN

function ZeroSumQICNN(
    nx::Int,
    nu::Int,
    nv::Int;
    hidden_A::AbstractVector{<:Integer} = [32, 32],
    hidden_B::AbstractVector{<:Integer} = [32, 32],
    hidden_C::AbstractVector{<:Integer} = [32, 32],
    T::Type = Float64,
    rng::AbstractRNG = Random.default_rng(),
    picnn_weight_scale::Union{Nothing,Real} = nothing,
    picnn_wz_softplus::Bool = false,
    picnn_z_gate_softplus::Bool = false,
    kwargs...
)
    # Back-compat / experiment convenience: `picnn_weight_scale` is the experiment flag, but
    # `create_picnn` expects `weight_scale`.
    picnn_kw = (;
        wz_softplus = picnn_wz_softplus,
        z_gate_softplus = picnn_z_gate_softplus,
        kwargs...,
    )
    if picnn_weight_scale === nothing
        A_net = create_picnn(nx, nu, hidden_A; T = T, rng = rng, picnn_kw...)
        B_net = create_picnn(nx, nv, hidden_B; T = T, rng = rng, picnn_kw...)
    else
        A_net = create_picnn(
            nx,
            nu,
            hidden_A;
            T = T,
            rng = rng,
            weight_scale = float(picnn_weight_scale),
            picnn_kw...,
        )
        B_net = create_picnn(
            nx,
            nv,
            hidden_B;
            T = T,
            rng = rng,
            weight_scale = float(picnn_weight_scale),
            picnn_kw...,
        )
    end
    C_net = MatrixNet(nx, nu, nv, hidden_C; T = T, rng = rng)
    return ZeroSumQICNN(A_net, B_net, C_net, nx, nu, nv)
end

"""
    Q_icnn(model, x, u, v) -> scalar

Evaluate `Q(x,u,v) = A(x,u) - B(x,v) + u' C(x) v`.
"""
function Q_icnn(model::ZeroSumQICNN, x::AbstractVecOrMat, u::AbstractVecOrMat, v::AbstractVecOrMat)
    x = vec(x)
    u = vec(u)
    v = vec(v)
    @assert length(x) == model.nx
    @assert length(u) == model.nu
    @assert length(v) == model.nv
    A_val = picnn_scalar(model.A_net, x, u)
    B_val = picnn_scalar(model.B_net, x, v)
    C_mat = model.C_net(x)
    return A_val - B_val + dot(u, C_mat * v)
end

"""
    create_zero_sum_q_icnn(nx, nu, nv; hidden_A, hidden_B, hidden_C, kwargs...)

Construct [`ZeroSumQICNN`](@ref) with default ICNN / MLP hidden sizes.
"""
function create_zero_sum_q_icnn(
    nx::Int,
    nu::Int,
    nv::Int;
    hidden_A::AbstractVector{<:Integer} = [32, 32],
    hidden_B::AbstractVector{<:Integer} = [32, 32],
    hidden_C::AbstractVector{<:Integer} = [32, 32],
    kwargs...,
)
    return ZeroSumQICNN(
        nx,
        nu,
        nv;
        hidden_A = hidden_A,
        hidden_B = hidden_B,
        hidden_C = hidden_C,
        kwargs...,
    )
end

"""Hessian of `u ↦ Q(x,u,v)` for fixed `x`, `v`."""
function Q_icnn_hessian_u(model::ZeroSumQICNN, x::AbstractVector, u::AbstractVector, v::AbstractVector)
    f = uvec -> Q_icnn(model, x, uvec, v)
    return ForwardDiff.hessian(f, u)
end

"""Hessian of `v ↦ Q(x,u,v)` for fixed `x`, `u`."""
function Q_icnn_hessian_v(model::ZeroSumQICNN, x::AbstractVector, u::AbstractVector, v::AbstractVector)
    f = vvec -> Q_icnn(model, x, u, vvec)
    return ForwardDiff.hessian(f, v)
end

function Q_icnn_convex_concave_checks(
    model::ZeroSumQICNN,
    x::AbstractVector,
    u::AbstractVector,
    v::AbstractVector;
    rtol::Real = 1e-10,
    eig_tol::Real = -1e-10,
)
    sym_u, spd_u, Hu, λu = picnn_convex_in_y(model.A_net, x, u; rtol = rtol, eig_tol = eig_tol)
    sym_v, spd_v, Hv, λv = picnn_convex_in_y(model.B_net, x, v; rtol = rtol, eig_tol = eig_tol)
    not_conv_x_A = picnn_not_convex_in_context(model.A_net, x, u)
    not_conv_x_B = picnn_not_convex_in_context(model.B_net, x, v)
    # Q concave in v iff B convex in v (Hessian of -B w.r.t. v is -H_B)
    Hq_v = Q_icnn_hessian_v(model, x, u, v)
    concave_v = all(eigvals(Symmetric(Hq_v)) .< -eig_tol)
    return (;
        A_convex_u = spd_u,
        A_sym_u = sym_u,
        B_convex_v = spd_v,
        B_sym_v = sym_v,
        A_not_convex_in_x = not_conv_x_A,
        B_not_convex_in_x = not_conv_x_B,
        Q_concave_v = concave_v,
        Hu,
        Hv,
        Hq_v,
        λu,
        λv,
    )
end
