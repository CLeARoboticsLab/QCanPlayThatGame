export SeCoND_regularization, quasi_newton_update

using LinearAlgebra: Symmetric, eigmin, I, norm

"""Regularized Newton step for a stage-game SeCoND update (`n_minimizer` = dim of `u`)."""
function SeCoND_regularization(ω::Any, J::Any, n_minimizer::Integer)
    # Thread-safety: Arpack.eigs is not safe under Threads.@threads; eigmin(Symmetric(...)) uses LAPACK per call.
    λmin = eigmin(Symmetric(J[1:n_minimizer, 1:n_minimizer]))
    λmax = eigmin(Symmetric(J[n_minimizer+1:end, n_minimizer+1:end]))
    J_plus_J_top = zeros(size(J))
    @views J_plus_J_top[1:n_minimizer, 1:n_minimizer] = 2 * J[1:n_minimizer, 1:n_minimizer]
    @views J_plus_J_top[n_minimizer+1:end, n_minimizer+1:end] = 2 * J[n_minimizer+1:end, n_minimizer+1:end]
    if sign(real(λmin)) == 1
        @views J_plus_J_top[1:n_minimizer, 1:n_minimizer] += I
    end
    if sign(real(λmax)) == 1
        @views J_plus_J_top[n_minimizer+1:end, n_minimizer+1:end] += I
    end
    return (J' * J * J_plus_J_top + 1e-3 * norm(ω)^2 * I) \ (J' * ω)
end

function quasi_newton_update(ω::Any, J::Any, quasi_coeff)
    return (J' * J + quasi_coeff * I) \ (J' * ω)
end
