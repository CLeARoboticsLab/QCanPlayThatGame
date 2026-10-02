module LearningValueFunctions

using Zygote
using LinearAlgebra: LinearAlgebra, norm_sqr, norm, isposdef
using Random: Random, rand
using Flux
using Statistics

include("Utils/dynamics.jl")
include("Utils/icnn_flux.jl")
include("Solver/SeCoND.jl")
include("Solver/SeCoND_forwarddiff.jl")
include("Utils/q_icnn/q_icnn.jl")
include("Utils/td_zero_sum_q_template.jl")
include("Utils/q_icnn/td_zero_sum_q_icnn.jl")

export create_icnn,
    ICNNChain,
    ICNNFirstLayer,
    ICNNLayer,
    FinalICNNLayer,
    icnn_softplus,
    huber_nonneg,
    icnn_scalar_output,
    icnn_hessian,
    icnn_hessian_spd,
    create_picnn,
    PICNNNet,
    PICNNBlock,
    picnn_scalar,
    picnn_convex_in_y,
    picnn_not_convex_in_context,
    create_context_icnn,
    ContextICNNChain,
    context_icnn_scalar,
    MatrixNet,
    ZeroSumQICNN,
    create_zero_sum_q_icnn,
    Q_icnn,
    Q_icnn_hessian_u,
    Q_icnn_hessian_v,
    Q_icnn_convex_concave_checks,
    solve_stage_game_Q_icnn_SeCoND_fd,
    rollout_zero_sum_q_icnn!,
    td_zero_sum_q_icnn_step!,
    td_icnn_mse_loss,
    rollout_training_metrics_icnn,
    ZeroSumQTrajectoryWorkspace,
    build_linear_dynamics,
    LinearDynamics,
    SeCoND_stage_game_forwarddiff,
    build_stage_state_control_qp_matrices

end # module
