using ForwardDiff: ForwardDiff
using Random: AbstractRNG, Random

# Input Convex Neural Network (ICNN) in Flux.
#
# Architecture and fixes follow the Lux.jl discussion:
# https://discourse.julialang.org/t/input-convex-neural-network-is-not-convex-at-origin-in-lux-jl/133565
#
# - Hidden passthrough weights `W_z` are constrained nonnegative (via `abs` in the forward pass).
# - First hidden layer uses `relu` (not `softplus`) so the Hessian at the origin is SPD.
# - Later layers use `icnn_softplus` (`log1p(exp(x))`), not NNlib's `softplus`, which is
#   unstable for second derivatives at the origin.
# - The final layer uses optional convex quadratic terms in `x` with nonnegative coefficients
#   (`huber_nonneg` on diagonal weights `d`).

"""
    huber_nonneg(x)

Elementwise Huber-like map used for ICNN quadratic coefficients: `0.5*x^2` for `|x|<1`, else `|x|-0.5`.
Applied to raw parameters `d` so the effective weights `huber_nonneg(d)` are nonnegative.
"""
huber_nonneg(x) = @. ifelse(abs(x) < 1.0, 0.5 * x^2, abs(x) - 0.5)

const ICNN_SOFTPLUS_CUTOFF = 20.0

"""
    icnn_softplus(x)

Stable softplus for ICNN hidden layers. Uses `log1p(exp(x))` for moderate `x` and the
asymptote `x` for `x > ICNN_SOFTPLUS_CUTOFF` to avoid `exp` overflow during training.
Prefer this over `NNlib.softplus` / Flux's default (`log1p(exp(-abs(x))) + relu(x)`).
"""
icnn_softplus(x) = @. ifelse(x > ICNN_SOFTPLUS_CUTOFF, x, log1p(exp(x)))

"""Reshape vector or matrix input to `(in_dims, batch)`."""
function _icnn_input_col(x::AbstractVecOrMat)
    x = vec(x)
    return reshape(x, :, 1)
end

"""
    ICNNFirstLayer{T,F}

First ICNN layer: `z = σ(W x + b)` with unconstrained `W`.
"""
struct ICNNFirstLayer{T<:Real,F}
    weight::Matrix{T}
    bias::Vector{T}
    activation::F
    use_bias::Bool
end

function ICNNFirstLayer(
    in_dims::Int,
    out_dims::Int,
    activation = relu;
    use_bias::Bool = true,
    T::Type = Float64,
    rng::AbstractRNG = Random.default_rng(),
)
    weight = randn(rng, T, out_dims, in_dims)
    bias = zeros(T, out_dims)
    return ICNNFirstLayer{T,typeof(activation)}(weight, bias, activation, use_bias)
end

Flux.@layer ICNNFirstLayer

function (layer::ICNNFirstLayer)(x::AbstractVecOrMat)
    x = _icnn_input_col(x)
    y = layer.weight * x
    if layer.use_bias
        y = y .+ layer.bias
    end
    return layer.activation.(y)
end

"""
    ICNNLayer{T,F}

ICNN hidden layer: `z' = σ(W_z^+ z + W_x x + b)` with `W_z^+ = abs(W_z)` (nonnegative).
"""
struct ICNNLayer{T<:Real,F}
    W_z::Matrix{T}
    W_x::Matrix{T}
    bias::Vector{T}
    activation::F
    use_bias::Bool
end

function ICNNLayer(
    in_dims::Int,
    hidden_dims::Int,
    out_dims::Int,
    activation = icnn_softplus;
    use_bias::Bool = true,
    T::Type = Float64,
    rng::AbstractRNG = Random.default_rng(),
)
    W_z = rand(rng, T, out_dims, hidden_dims)
    W_x = randn(rng, T, out_dims, in_dims)
    bias = zeros(T, out_dims)
    return ICNNLayer{T,typeof(activation)}(W_z, W_x, bias, activation, use_bias)
end

Flux.@layer ICNNLayer

function (layer::ICNNLayer)((x, z)::Tuple{<:AbstractVecOrMat,<:AbstractVecOrMat})
    x = _icnn_input_col(x)
    z = _icnn_input_col(z)
    Wz = abs.(layer.W_z)
    y = Wz * z .+ layer.W_x * x
    if layer.use_bias
        y = y .+ layer.bias
    end
    return layer.activation.(y)
end

"""
    FinalICNNLayer{T,F}

ICNN output layer: `y = W_z^+ z [+ sum_i huber_nonneg(d_i) x_i^2] [+ W_x x] [+ b]`, then `σ(y)`.
"""
struct FinalICNNLayer{T<:Real,F}
    W_z::Matrix{T}
    W_x::Matrix{T}
    d::Vector{T}
    bias::Vector{T}
    activation::F
    use_bias::Bool
    use_quadratic::Bool
end

function FinalICNNLayer(
    in_dims::Int,
    hidden_dims::Int,
    out_dims::Int = 1;
    activation = identity,
    use_bias::Bool = false,
    use_quadratic::Bool = true,
    T::Type = Float64,
    rng::AbstractRNG = Random.default_rng(),
)
    W_z = rand(rng, T, out_dims, hidden_dims)
    W_x = use_quadratic ? randn(rng, T, out_dims, in_dims) : zeros(T, out_dims, in_dims)
    d = use_quadratic ? rand(rng, T, in_dims) : zeros(T, in_dims)
    bias = zeros(T, out_dims)
    return FinalICNNLayer{T,typeof(activation)}(
        W_z,
        W_x,
        d,
        bias,
        activation,
        use_bias,
        use_quadratic,
    )
end

Flux.@layer FinalICNNLayer

function (layer::FinalICNNLayer)((x, z)::Tuple{<:AbstractVecOrMat,<:AbstractVecOrMat})
    x = _icnn_input_col(x)
    z = _icnn_input_col(z)
    Wz = abs.(layer.W_z)
    y = Wz * z
    if layer.use_quadratic
        quad = sum(x .^ 2 .* huber_nonneg(layer.d); dims = 1)
        y = y .+ quad .+ layer.W_x * x
    end
    if layer.use_bias
        y = y .+ layer.bias
    end
    return layer.activation.(y)
end

"""
    ICNNChain{L}

Chain of ICNN layers. Forward pass threads the input `x` through all layers.
"""
struct ICNNChain{L<:Tuple}
    layers::L
end

ICNNChain(layers...) = ICNNChain(layers)

Flux.@layer ICNNChain

function (chain::ICNNChain)(x::AbstractVecOrMat)
    z = chain.layers[1](x)
    for i in 2:length(chain.layers)
        z = chain.layers[i]((x, z))
    end
    return z
end

function Base.show(io::IO, chain::ICNNChain)
    print(io, "ICNNChain(")
    for (i, layer) in enumerate(chain.layers)
        i > 1 && print(io, ", ")
        show(io, layer)
    end
    print(io, ")")
end

"""
    create_icnn(n_vars, hidden_dims=[32, 32]; T=Float64, rng, use_float64=true)

Build an Input Convex Neural Network `f(x)` that is convex in `x`.

Default activations match the corrected Lux recipe from the Discourse thread:
- first layer: `relu`
- hidden layers: `icnn_softplus`
- final layer: `identity` with `use_quadratic=true`

Set `use_float64=false` to keep parameter type `T` (e.g. `Float32`).
"""
function create_icnn(
    n_vars::Int,
    hidden_dims::AbstractVector{<:Integer} = [32, 32];
    T::Type = Float64,
    rng::AbstractRNG = Random.default_rng(),
    first_activation = relu,
    hidden_activation = icnn_softplus,
    final_activation = identity,
    final_use_quadratic::Bool = true,
    final_use_bias::Bool = false,
    use_float64::Bool = true,
)
    @assert !isempty(hidden_dims) "hidden_dims cannot be empty"

    layers = Any[]
    push!(layers, ICNNFirstLayer(n_vars, hidden_dims[1], first_activation; T = T, rng = rng))
    for i in 1:(length(hidden_dims) - 1)
        push!(
            layers,
            ICNNLayer(
                n_vars,
                hidden_dims[i],
                hidden_dims[i + 1],
                hidden_activation;
                T = T,
                rng = rng,
            ),
        )
    end
    push!(
        layers,
        FinalICNNLayer(
            n_vars,
            hidden_dims[end],
            1;
            activation = final_activation,
            use_bias = final_use_bias,
            use_quadratic = final_use_quadratic,
            T = T,
            rng = rng,
        ),
    )
    model = ICNNChain(tuple(layers...)...)
    return use_float64 && T === Float64 ? Flux.f64(model) : model
end

"""
    icnn_scalar_output(model, x)

Evaluate `model(x)` and return a scalar (first output element).
"""
function icnn_scalar_output(model, x::AbstractVecOrMat)
    y = model(x)
    return only(vec(y))
end

"""
    icnn_hessian(model, x)

Hessian of the scalar ICNN output w.r.t. `x` (via ForwardDiff).
"""
function icnn_hessian(model, x::AbstractVector)
    f = xvec -> icnn_scalar_output(model, xvec)
    return ForwardDiff.hessian(f, x)
end

"""
    icnn_hessian_spd(model, x; rtol=1e-10, eig_tol=-1e-10)

Check symmetry and positive semidefiniteness of the Hessian at `x` (for tests / diagnostics).
"""
function icnn_hessian_spd(model, x::AbstractVector; rtol::Real = 1e-10, eig_tol::Real = -1e-10)
    H = icnn_hessian(model, x)
    sym_ok = isapprox(H, H'; rtol = rtol)
    λ = eigvals(Symmetric(H))
    spd_ok = all(λ .> eig_tol)
    return sym_ok, spd_ok, H, λ
end
