export LinearDynamics, build_linear_dynamics, get_next_state

"""
Store key information of a linear dynamics object, defined by:
1. A: A matrix that maps the game's state at time t to the next state at time t+1
2. Bs: A vector of B matrices that map each agent's control at time t to the next state at time t+1

Linear dynamics are defined as:
    x_{tt+1} = A * x_tt + sum_{ii=1}^N B^ii * u^ii_tt
"""
struct LinearDynamics
    A::Any
    Bs::Any
end

"""Helper function to build a LinearDynamics object from a vector of A matrices and a vector of B matrices."""
function build_linear_dynamics(As, Bs)
    """
    As stores the A matrices for each player, where each A matrix maps a single player's state to the next state.

    Bs stores the B matrices for each player, where each B matrix maps a single player's control to the next state.

    """

    state_dimensions = [size(A, 1) for A in As]
    control_dimensions = [size(B', 1) for B in Bs]
    N = length(As)

    A_dynamics = zeros(sum(state_dimensions), sum(state_dimensions))
    Bs_dynamics = [zeros(sum(state_dimensions), control_dimensions[ii]) for ii in 1:N]
    for ii in 1:N
        A_dynamics[
            (sum(state_dimensions[1:(ii - 1)]) + 1):sum(state_dimensions[1:ii]),
            (sum(state_dimensions[1:(ii - 1)]) + 1):sum(state_dimensions[1:ii]),
        ] .= As[ii]
        Bs_dynamics[ii][
            (sum(state_dimensions[1:(ii - 1)]) + 1):sum(state_dimensions[1:ii]),
            1:control_dimensions[ii],
        ] .= Bs[ii]
    end
    LinearDynamics(A_dynamics, Bs_dynamics)
end

"""Helper function to get the next state of the game from the current state and controls."""
function get_next_state(dynamics::LinearDynamics, x, u)
    dynamics.A * x + sum([dynamics.Bs[ii] * u[ii] for ii in eachindex(u)])
end
