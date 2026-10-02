using LinearAlgebra: dot
using Random: AbstractRNG, Random

"""
    MatrixNet{L}

State-conditioned matrix `C(x) ∈ ℝ^{nu×nv}` from an MLP: `x ↦ vec(C(x))`.
"""
struct MatrixNet{L}
    chain::L
    nu::Int
    nv::Int
end

Flux.@layer MatrixNet

function MatrixNet(
    nx::Int,
    nu::Int,
    nv::Int,
    hidden_dims::AbstractVector{<:Integer} = [32, 32];
    T::Type = Float64,
    rng::AbstractRNG = Random.default_rng(),
    activation = relu,
    use_float64::Bool = true,
)
    out_dim = nu * nv
    layers = Any[]
    d_prev = nx
    for h in hidden_dims
        push!(layers, Dense(d_prev => h, activation))
        d_prev = h
    end
    push!(layers, Dense(d_prev => out_dim))
    chain = Chain(
        x -> vec(reshape(x, :)),
        layers...,
    )
    net = MatrixNet(chain, nu, nv)
    return use_float64 && T === Float64 ? Flux.f64(net) : net
end

function (net::MatrixNet)(x::AbstractVecOrMat)
    v = net.chain(x)
    return reshape(vec(v), net.nu, net.nv)
end

"""Bilinear coupling `u' C(x) v` with `C = net(x)`."""
function bilinear_coupling(net::MatrixNet, x::AbstractVecOrMat, u::AbstractVecOrMat, v::AbstractVecOrMat)
    C = net(x)
    return dot(vec(u), C * vec(v))
end
