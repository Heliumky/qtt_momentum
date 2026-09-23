"""Conversion from multi-dimensional Fourier-coefficient MPS data to real-space
functions."""

"""Basis matrix `Phi[point, mode] = fourier_basis(mode, x[point], L)`, with
modes ordered as `freqs(N)` (QTT/two's-complement storage order)."""
function _fourier_basis_matrix(N::Integer, x::AbstractVector, L::Real)
    modes = freqs(N)
    return reduce(hcat, (fourier_basis(mode, x, L) for mode in modes))
end

"""
    coeffs_to_realspace_nd(coefficients, grids, Ls)

Evaluate a `D`-dimensional Fourier series
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
    register_to_axes(values, layout; register=:fourier)

Reorder a dense vector over a joint register (column-major, site 1 fastest)
into a `D`-dimensional array with axis `d` of length `2^N_d`. With
`register=:fourier` axis `d` is in QTT mode order (`freqs(N_d)`); with
`register=:realspace` it is in natural grid order.
"""
function register_to_axes(values::AbstractVector, layout; register::Symbol=:fourier)
    layout = layout isa RegisterLayout ? layout : RegisterLayout(layout)
    register in (:fourier, :realspace) ||
        throw(ArgumentError("register must be :fourier or :realspace"))
    N = nsites(layout)
    length(values) == 2^N || throw(DimensionMismatch("expected 2^N values"))

    counts = layout.qubits_per_dim
    D = length(counts)
    result = similar(values, Tuple(2 .^ counts))
    for c in 0:(2^N - 1)
        axis_index = zeros(Int, D)
        for (site, (d, level)) in enumerate(layout.owner)
            bit = (c >> (site - 1)) & 1
            weight_power = register === :fourier ? level - 1 : counts[d] - level
            axis_index[d] += bit << weight_power
        end
        result[CartesianIndex(Tuple(axis_index .+ 1))] = values[c + 1]
    end
    return result
end

"""
    mps_to_realspace_nd(state, sites, qubits_per_dim, half_width_per_dim;
                        normalize_coefficients=false, points_per_dim=nothing,
                        offsets=zeros(D), cutoff=1e-26)

Reconstruct the real-space values of a joint Fourier-coefficient MPS on the
outer product of per-dimension uniform grids. `qubits_per_dim` may be a
[`RegisterLayout`](@ref) (block layout otherwise).

On the native grid (`points_per_dim=nothing`) the joint inverse DFT MPO is
applied to the MPS, `psi(x) = prod_d sqrt(M_d/2L_d) * (U† c)(x)`, and the
real-space MPS is contracted (`method = :dft_mpo`). Other point counts use a
direct separable Fourier sum (`method = :direct`).
"""
function mps_to_realspace_nd(state::MPS, sites::AbstractVector{<:Index},
                             qubits_per_dim,
                             half_width_per_dim::AbstractVector{<:Real};
                             normalize_coefficients::Bool=false,
                             points_per_dim::Union{Nothing,AbstractVector{<:Integer}}=nothing,
                             offsets::AbstractVector{<:Real}=zeros(length(half_width_per_dim)),
                             cutoff::Real=1e-26)
    layout = qubits_per_dim isa RegisterLayout ? qubits_per_dim : RegisterLayout(qubits_per_dim)
    counts = layout.qubits_per_dim
    D = length(counts)
    length(half_width_per_dim) == D ||
        throw(DimensionMismatch("one half-width is required per dimension"))
    nsites(layout) == length(sites) ||
        throw(DimensionMismatch("qubits_per_dim must sum to the number of sites"))

    if normalize_coefficients
        state_norm = norm(state)
        iszero(state_norm) && throw(ArgumentError("cannot normalize a zero state"))
        state = state / state_norm
    end
    coefficient_array = register_to_axes(itensor_mps_to_vec(state, sites), layout)

    native = isnothing(points_per_dim) || collect(points_per_dim) == 2 .^ counts
    point_counts = isnothing(points_per_dim) ? (2 .^ counts) : collect(points_per_dim)
    length(point_counts) == D ||
        throw(DimensionMismatch("one point count is required per dimension"))
    grids = [realspace_grid(point_counts[d], half_width_per_dim[d]; offset=offsets[d]) for d in 1:D]

    if native
        transform = fourier_transform_mpo_nd(layout, half_width_per_dim; offsets)
        inverse = array_mpo_to_itensor(sites, adjoint_mpo(transform))
        realspace = apply(inverse, state; cutoff=cutoff)
        scale = prod(sqrt(2^counts[d] / 2half_width_per_dim[d]) for d in 1:D)
        values = scale .* register_to_axes(itensor_mps_to_vec(realspace, sites), layout;
                                           register=:realspace)
        method = :dft_mpo
    else
        values = coeffs_to_realspace_nd(coefficient_array, grids, half_width_per_dim)
        method = :direct
    end
    return (; grids, values, coefficients=coefficient_array, method)
end
