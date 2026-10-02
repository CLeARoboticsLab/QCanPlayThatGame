using ForwardDiff: ForwardDiff
using LinearAlgebra: Symmetric, eigvals
using Random: AbstractRNG, Random

# Partially Input Convex Neural Network (PICNN), Amos et al. §3.2, Eq. (3):
#   https://arxiv.org/abs/1609.07152
#
#   f(context, y) is convex in y (minimizer control u or maximizer v) for ALL context,
#   and is **not** required to be convex or concave in context (state x).
#
# Paper notation (§3.2): non-convex input x_ctx, convex input y.
#   u_{i+1} = g̃_i(W̃_i u_i + b̃_i),  u_0 = x_ctx
#   z_{i+1} = g_i( W^{(z)}_i ( z_i ∘ [W^{(zu)}_i u_i + b^{(z)}_i]_+ )
#                 + W^{(y)}_i ( y ∘ (W^{(yu)}_i u_i + b^{(y)}_i) )
#                 + W^{(u)}_i u_i + b_i )
#   f = z_k
#
# Only W^{(z)} is constrained nonnegative in the forward pass (`abs` or `icnn_softplus`
# via `wz_softplus`). All x-path and y/context couplings are unconstrained → rich
# non-convex dependence on context.

"""Reshape to column `(d, 1)` for matrix multiply."""
function _qicnn_col(v::AbstractVecOrMat)
    v = vec(v)
    return reshape(v, :, 1)
end

"""
    PICNNBlock{T,F,G}

One PICNN layer (Eq. 3 in Amos et al., ICML 2017).

`wz_softplus=true` → effective `W^{(z)} = softplus(W_z)`; `false` → `abs.(W_z)`.
`z_gate_softplus=true` → `z` passthrough gate is `icnn_softplus`; `false` → ReLU.
"""
struct PICNNBlock{T<:Real,F,G}
    W_tilde::Matrix{T}
    b_tilde::Vector{T}
    W_z::Matrix{T}
    W_zu::Matrix{T}
    b_z::Vector{T}
    W_y::Matrix{T}
    W_yu::Matrix{T}
    b_y::Vector{T}
    W_u::Matrix{T}
    b::Vector{T}
    act_u::F
    act_z::G
    wz_softplus::Bool
    z_gate_softplus::Bool
end

Flux.@layer PICNNBlock

function PICNNBlock(
    n_u_in::Int,
    n_u_out::Int,
    n_z_in::Int,
    n_z_out::Int,
    n_y::Int;
    act_u = relu,
    act_z = icnn_softplus,
    wz_softplus::Bool = false,
    z_gate_softplus::Bool = false,
    T::Type = Float64,
    rng::AbstractRNG = Random.default_rng(),
    weight_scale::Real = 1.0,
)
    s = T(weight_scale)
    return PICNNBlock{T,typeof(act_u),typeof(act_z)}(
        s .* randn(rng, T, n_u_out, n_u_in),
        zeros(T, n_u_out),
        s .* rand(rng, T, n_z_out, n_z_in),
        s .* randn(rng, T, n_z_in, n_u_in),
        zeros(T, n_z_in),
        s .* randn(rng, T, n_z_out, n_y),
        s .* randn(rng, T, n_y, n_u_in),
        zeros(T, n_y),
        s .* randn(rng, T, n_z_out, n_u_in),
        zeros(T, n_z_out),
        act_u,
        act_z,
        wz_softplus,
        z_gate_softplus,
    )
end

"""Single PICNN recurrence; `y` is the fixed convex input (passthrough at every layer)."""
function (blk::PICNNBlock)(y::AbstractVecOrMat, u::AbstractVecOrMat, z::AbstractVecOrMat)
    y = _qicnn_col(y)
    u = _qicnn_col(u)
    z = _qicnn_col(z)
    u_new = blk.act_u(blk.W_tilde * u .+ blk.b_tilde)
    pre_gate = blk.W_zu * u .+ blk.b_z
    gate = blk.z_gate_softplus ? icnn_softplus(pre_gate) : max.(pre_gate, 0)
    z_gated = z .* gate
    Wz = blk.wz_softplus ? icnn_softplus(blk.W_z) : abs.(blk.W_z)
    y_gate = y .* (blk.W_yu * u .+ blk.b_y)
    z_new = blk.act_z(Wz * z_gated .+ blk.W_y * y_gate .+ blk.W_u * u .+ blk.b)
    return u_new, z_new
end

"""
    PICNNNet{L}

PICNN output `f(context, y)`; convex in `y`, unrestricted in `context`.
"""
struct PICNNNet{L<:Tuple,F}
    layers::L
    n_context::Int
    n_convex::Int
    final_quad_d::Vector{Float64}
    final_quad::Bool
    final_activation::F
end

Flux.@layer PICNNNet

function (net::PICNNNet)(context::AbstractVecOrMat, y::AbstractVecOrMat)
    u = _qicnn_col(context)
    y = _qicnn_col(y)
    z = zeros(eltype(u), size(net.layers[1].W_z, 2), 1)
    for layer in net.layers
        u, z = layer(y, u, z)
    end
    out = net.final_activation.(z)
    if net.final_quad
        quad = sum(y .^ 2 .* huber_nonneg(net.final_quad_d); dims = 1)
        out = out .+ quad
    end
    return out
end

function Base.show(io::IO, net::PICNNNet)
    print(io, "PICNNNet(n_context=$(net.n_context), n_convex=$(net.n_convex), layers=$(length(net.layers)))")
end

"""
    create_picnn(n_context, n_convex, hidden_dims=[32, 32]; ...)

Build PICNN §3.2 (Eq. 3): **convex in `n_convex` (y)**, **not convex in `n_context` (x)**.

Optional `final_quadratic_in_y=true` adds `sum_i ρ(d_i) y_i²` with `ρ = huber_nonneg` (FICNN-style
head on the convex input only; does not constrain context).

`weight_scale` multiplies all random PICNN weight / `final_quad_d` draws (biases stay zero).
Use `weight_scale < 1` (e.g. `0.1`) for smaller initial `Q` and stabler early TD gradients.

`wz_softplus=true` reparameterizes `W^{(z)}` with `icnn_softplus` instead of `abs` (still ≥ 0).
`z_gate_softplus=true` uses `icnn_softplus` for the `z` passthrough gate instead of ReLU (still ≥ 0).
"""
function create_picnn(
    n_context::Int,
    n_convex::Int,
    hidden_dims::AbstractVector{<:Integer} = [32, 32];
    T::Type = Float64,
    rng::AbstractRNG = Random.default_rng(),
    act_u = relu,
    act_z = icnn_softplus,
    final_activation = identity,
    final_quadratic_in_y::Bool = true,
    weight_scale::Real = 1.0,
    wz_softplus::Bool = false,
    z_gate_softplus::Bool = false,
    use_float64::Bool = true,
)
    @assert !isempty(hidden_dims)
    @assert weight_scale > 0
    s = T(weight_scale)
    layers = PICNNBlock[]
    for (i, h) in enumerate(hidden_dims)
        n_u_in = i == 1 ? n_context : hidden_dims[i - 1]
        n_z_in = i == 1 ? 1 : hidden_dims[i - 1]
        n_z_out = i == length(hidden_dims) ? 1 : h
        push!(
            layers,
            PICNNBlock(
                n_u_in,
                h,
                n_z_in,
                n_z_out,
                n_convex;
                act_u,
                act_z,
                wz_softplus,
                z_gate_softplus,
                T,
                rng,
                weight_scale = s,
            ),
        )
    end
    d = final_quadratic_in_y ? s .* rand(rng, T, n_convex) : zeros(T, n_convex)
    net = PICNNNet(
        tuple(layers...),
        n_context,
        n_convex,
        d,
        final_quadratic_in_y,
        final_activation,
    )
    return use_float64 && T === Float64 ? Flux.f64(net) : net
end

"""Scalar PICNN output."""
function picnn_scalar(net, context::AbstractVecOrMat, y::AbstractVecOrMat)
    return only(vec(net(context, y)))
end

function picnn_hessian_convex_input(net, context::AbstractVector, y::AbstractVector)
    f = yvec -> picnn_scalar(net, context, yvec)
    return ForwardDiff.hessian(f, y)
end

function picnn_hessian_context(net, context::AbstractVector, y::AbstractVector)
    f = xvec -> picnn_scalar(net, xvec, y)
    return ForwardDiff.hessian(f, context)
end

function picnn_convex_in_y(net, context::AbstractVector, y::AbstractVector; rtol::Real = 1e-10, eig_tol::Real = -1e-10)
    H = picnn_hessian_convex_input(net, context, y)
    sym_ok = isapprox(H, H'; rtol = rtol)
    λ = eigvals(Symmetric(H))
    return sym_ok, all(λ .> eig_tol), H, λ
end

"""True if Hessian w.r.t. context is **not** PSD (generic non-convexity in x)."""
function picnn_not_convex_in_context(
    net,
    context::AbstractVector,
    y::AbstractVector;
    eig_tol::Real = 1e-8,
)
    H = picnn_hessian_context(net, context, y)
    λ = eigvals(Symmetric(H))
    return !all(λ .> eig_tol)
end

# Backward-compatible names (A/B nets in Q decomposition)
const ContextICNNChain = PICNNNet
create_context_icnn(convex_dims, context_dims; kwargs...) =
    create_picnn(context_dims, convex_dims; kwargs...)
context_icnn_scalar(net, ctx, y) = picnn_scalar(net, ctx, y)
context_icnn_hessian_control(net, ctx, y) = picnn_hessian_convex_input(net, ctx, y)
function context_icnn_convex_in_control(net, ctx, y; kwargs...)
    sym, spd, H, λ = picnn_convex_in_y(net, ctx, y; kwargs...)
    return sym, spd, H, λ
end
