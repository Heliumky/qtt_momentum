"""Conversion from separable multi-dimensional Fourier-coefficient MPS data
to real-space functions."""

"""Basis matrix `Phi[point, mode] = fourier_basis(mode, x[point], L)`, with
modes ordered as `freqs(N)` (QTT/two's-complement storage order)."""
function _fourier_basis_matrix(N::Integer, x::AbstractVector, L::Real)
    modes = freqs(N)
    return reduce(hcat, (fourier_basis(mode, x, L) for mode in modes))
end

"""
    coeffs_to_realspace_nd(coefficients, grids, Ls)

Evaluate a separable `D`-dimensional Fourier series
`sum_{n_1,...,n_D} c_{n_1,...,n_D} * prod_d exp(i*pi*n_d*x_d/L_d)/sqrt(2*L_d)`
on the outer product of `grids`. `coefficients` is a `D`-dimensional array
whose axis `d` has length `2^N_d` and is indexed in QTT mode order (see
`freqs`); axis `d` of the output has length `length(grids[d])`.
"""
function coeffs_to_realspace_nd(coefficients::AbstractArray{<:Number},
                                grids::AbstractVector{<:AbstractVector{<:Real}},
                                Ls::AbstractVector{<:Real})
    D = ndims(coefficients)
    (length(grids) == D && length(Ls) == D) ||
        throw(DimensionMismatch("one grid and one L are required per dimension"))

    result = ComplexF64.(coefficients)
    for axis in 1:D
        M = size(result, axis)
        ispow2(M) || throw(ArgumentError("axis $axis length must be a power of two"))
        basis = _fourier_basis_matrix(trailing_zeros(M), grids[axis], Ls[axis])

        perm = (axis, setdiff(1:D, axis)...)
        moved = permutedims(result, perm)
        transformed = basis * reshape(moved, M, :)
        result = permutedims(
            reshape(transformed, length(grids[axis]), size(moved)[2:end]...),
            invperm(perm),
        )
    end
    return result
end

"""
    mps_to_realspace_nd(state, sites, qubits_per_dim, half_width_per_dim;
                        normalize_coefficients=false, points_per_dim=nothing)

Contract a joint Fourier-coefficient MPS (dimension `d` occupying
`qubits_per_dim[d]` contiguous sites) and reconstruct its real-space values
on the outer product of per-dimension uniform grids. `points_per_dim`
defaults to the native `2^{qubits_per_dim}` grid; any other point count is
evaluated by a direct Fourier sum along that axis.
"""
function mps_to_realspace_nd(state::MPS, sites::AbstractVector{<:Index},
                             qubits_per_dim::AbstractVector{<:Integer},
                             half_width_per_dim::AbstractVector{<:Real};
                             normalize_coefficients::Bool=false,
                             points_per_dim::Union{Nothing,AbstractVector{<:Integer}}=nothing)
    length(qubits_per_dim) == length(half_width_per_dim) ||
        throw(DimensionMismatch("one half-width is required per dimension"))
    sum(qubits_per_dim) == length(sites) ||
        throw(DimensionMismatch("qubits_per_dim must sum to the number of sites"))

    coefficients = itensor_mps_to_vec(state, sites)
    if normalize_coefficients
        coefficient_norm = sqrt(sum(abs2, coefficients))
        iszero(coefficient_norm) && throw(ArgumentError("cannot normalize a zero state"))
        coefficients = coefficients ./ coefficient_norm
    end
    coefficient_array = reshape(coefficients, Tuple(2 .^ qubits_per_dim))

    point_counts = isnothing(points_per_dim) ? (2 .^ qubits_per_dim) : collect(points_per_dim)
    length(point_counts) == length(qubits_per_dim) ||
        throw(DimensionMismatch("one point count is required per dimension"))
    grids = [realspace_grid(point_counts[d], half_width_per_dim[d]) for d in eachindex(point_counts)]

    values = coeffs_to_realspace_nd(coefficient_array, grids, half_width_per_dim)
    return (; grids, values, coefficients=coefficient_array)
end
