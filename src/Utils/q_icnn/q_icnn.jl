# Q-function via ICNN / PICNN decomposition (Amos et al. §3.2):
#
#   Q(x, u, v) = A(x, u) - B(x, v) + u' C(x) v
#
# - A, B: PICNN — convex in u / v respectively, **unrestricted in x**
# - C: MatrixNet — arbitrary MLP in x (no convexity constraint)

include(joinpath(@__DIR__, "picnn_flux.jl"))
include("matrix_net.jl")
include("zero_sum_q_icnn.jl")
