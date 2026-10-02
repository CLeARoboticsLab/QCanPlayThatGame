export ZeroSumQTrajectoryWorkspace

"""
    ZeroSumQTrajectoryWorkspace(T, nx, nu, nv, M, N)

Preallocated storage for `M` rollouts of length `N` (each rollout has `N` action pairs and
states `x₁,…,x_{N+1}`).

Fields:
- `xs`: `(nx, N+1, M)` — state trajectories
- `us`: `(nu, N, M)` — minimizer controls on the sampled transitions
- `vs`: `(nv, N, M)` — maximizer controls on the sampled transitions
- `us_next`: `(nu, M)` — minimizer Nash controls at `x'` (1-step ICNN TD bootstrap)
- `vs_next`: `(nv, M)` — maximizer Nash controls at `x'` (1-step ICNN TD bootstrap)
- `td_errors`: `(N, M)` — TD residual buffer
- `seco_nd_iters`: `(N, M)` — SeCoND inner-loop iteration counts from the last stage solve per `(t, m)`

ICNN 1-step TD stores exploratory `(u,v)` in `us`/`vs` and SeCoND Nash at `x'` in `us_next`/`vs_next`.
"""
struct ZeroSumQTrajectoryWorkspace{Xs,Us,Vs,Usn,Vsn,D,I}
    xs::Xs
    us::Us
    vs::Vs
    us_next::Usn
    vs_next::Vsn
    td_errors::D
    seco_nd_iters::I
end

function ZeroSumQTrajectoryWorkspace(
    ::Type{T},
    nx::Integer,
    nu::Integer,
    nv::Integer,
    M::Integer,
    N::Integer,
) where {T<:Real}
    xs = zeros(T, nx, N + 1, M)
    us = zeros(T, nu, N, M)
    vs = zeros(T, nv, N, M)
    us_next = zeros(T, nu, M)
    vs_next = zeros(T, nv, M)
    td_errors = zeros(T, N, M)
    seco_nd_iters = zeros(Int, N, M)
    return ZeroSumQTrajectoryWorkspace(xs, us, vs, us_next, vs_next, td_errors, seco_nd_iters)
end
