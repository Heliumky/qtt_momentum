"""Exact diagonal MPOs for derivatives in the Fourier basis."""

"""Two's-complement bit weights for an `N`-site Fourier index."""
function frequency_weights(N::Integer)
    N >= 1 || throw(ArgumentError("N must be positive"))
    weights = Float64[2.0^power for power in 0:(N - 2)]
    push!(weights, -2.0^(N - 1))
    return weights
end

"""
    diag_power_mpo(weights, order; prefactor=1)

Construct the exact diagonal MPO
`prefactor * (sum(weights[k] * bit[k]))^order`.

The link state carries all partial powers from zero through `order`, so the
maximum bond dimension is `order + 1`. No TCI or numerical compression is
used.
"""
function diag_power_mpo(weights::AbstractVector{<:Number}, order::Integer;
                        prefactor::Number=1)
    isempty(weights) && throw(ArgumentError("weights cannot be empty"))
    order >= 0 || throw(ArgumentError("order must be nonnegative"))

    element_type = promote_type(eltype(weights), typeof(prefactor), Float64)
    bond_dimension = order + 1
    mpo = Vector{Array{element_type,4}}(undef, length(weights))

    for (site, weight) in enumerate(weights)
        tensor = zeros(element_type, bond_dimension, 2, 2, bond_dimension)
        for bit in 0:1
            increment = weight * bit
            for old_power in 0:order, new_power in old_power:order
                tensor[old_power + 1, bit + 1, bit + 1, new_power + 1] =
                    binomial(new_power, old_power) * increment^(new_power - old_power)
            end
        end

        site == 1 && (tensor = tensor[1:1, :, :, :])
        site == length(weights) &&
            (tensor = tensor[:, :, :, bond_dimension:bond_dimension] .* prefactor)
        mpo[site] = tensor
    end
    return mpo
end

"""Compatibility helper for `prefactor * diag((sum(weights .* bits))^2)`."""
function diag_quadratic_mpo(weights::AbstractVector{<:Number}, prefactor::Number=1)
    return diag_power_mpo(weights, 2; prefactor=prefactor)
end

"""
    differential_mpo(order, N, L; coefficient=1)

Return the exact Fourier-basis MPO for
`coefficient * d^order/dx^order`. Since
`d^order exp(i*pi*n*x/L) = (i*pi*n/L)^order exp(i*pi*n*x/L)`, the operator is
diagonal and has maximum bond dimension `order + 1`.
"""
function differential_mpo(order::Integer, N::Integer, L::Real;
                          coefficient::Number=1)
    order >= 0 || throw(ArgumentError("order must be nonnegative"))
    L > 0 || throw(ArgumentError("L must be positive"))
    prefactor = coefficient * (im * pi / L)^order
    return diag_power_mpo(frequency_weights(N), order; prefactor=prefactor)
end

"""Kinetic energy `-1/2 d^2/dx^2` in the Fourier basis."""
function kinetic_mpo(N::Integer, L::Real)
    L > 0 || throw(ArgumentError("L must be positive"))
    return diag_power_mpo(
        frequency_weights(N),
        2;
        prefactor=0.5 * (pi / L)^2,
    )
end
