"""Conversion from Fourier-coefficient MPS data to real-space functions."""

"""Contract an ITensor MPS to a dense vector in column-major site order (site 1 fastest)."""
function itensor_mps_to_vec(state::MPS, sites::AbstractVector{<:Index})
    length(state) == length(sites) || throw(DimensionMismatch("one site index is required per MPS tensor"))
    contracted = state[1]
    for site in 2:length(state)
        contracted *= state[site]
    end
    return vec(Array(contracted, sites...))
end

"""Normalized Fourier basis function `exp(i*pi*n*x/L) / sqrt(2L)`."""
function fourier_basis(n::Integer, x, L::Real)
    L > 0 || throw(ArgumentError("L must be positive"))
    return exp.(im * pi * n .* x ./ L) ./ sqrt(2L)
end

"""
    evaluate_fourier_series(coefficients, x, L)

Evaluate `sum_n c_n*exp(i*pi*n*x/L)/sqrt(2L)` at arbitrary real-space points.
Coefficients must be in two's-complement/QTT order.
"""
function evaluate_fourier_series(coefficients::AbstractVector{<:Number},
                                 x::AbstractVector, L::Real)
    M = length(coefficients)
    ispow2(M) || throw(ArgumentError("the coefficient count must be a power of two"))
    values = zeros(ComplexF64, length(x))
    for (coefficient, mode) in zip(coefficients, freqs(trailing_zeros(M)))
        values .+= coefficient .* fourier_basis(mode, x, L)
    end
    return values
end

"""
    fourier_to_realspace_mps(state, sites, L; offset=0, cutoff=1e-26,
                             maxdim=typemax(Int), transform_arrays=nothing)

Map a Fourier-coefficient MPS to the real-space MPS of `psi(x_j)` on the
native grid `realspace_grid(2^N, L; offset)` by applying the inverse DFT MPO,
`psi(x_j) = sqrt(M/2L) * (U† c)_j`. The result lives in the real-space
register (site 1 = MSB of `j`) and is never contracted to a dense vector.
Pass `transform_arrays` to reuse an already built DFT MPO.
"""
function fourier_to_realspace_mps(state::MPS, sites::AbstractVector{<:Index}, L::Real;
                                  offset::Real=0,
                                  cutoff::Real=1e-26,
                                  maxdim::Integer=typemax(Int),
                                  transform_arrays=nothing)
    N = length(sites)
    length(state) == N || throw(DimensionMismatch("site count does not match MPS"))
    transform = isnothing(transform_arrays) ? fourier_transform_mpo(N, L; offset) : transform_arrays
    inverse = array_mpo_to_itensor(sites, adjoint_mpo(transform))
    realspace = apply(inverse, state; cutoff=cutoff, maxdim=maxdim)
    return realspace * sqrt(2^N / 2L)
end

"""
    realspace_mps_to_vec(state, sites)

Contract a real-space-register MPS (site 1 = MSB) to a dense vector in natural
grid order.
"""
function realspace_mps_to_vec(state::MPS, sites::AbstractVector{<:Index})
    values = itensor_mps_to_vec(state, sites)
    ordered = similar(values)
    ordered[grid_register_order(length(sites))] = values
    return ordered
end

"""
    mps_to_realspace(state, sites, L; normalize_coefficients=false,
                     x=nothing, npoints=nothing, offset=0)

Reconstruct the real-space values of a Fourier-coefficient MPS. Returns a
named tuple `(x, values, coefficients, method)`.

With neither `x` nor `npoints`, values are evaluated on the native `2^N` grid
by the inverse DFT MPO ([`fourier_to_realspace_mps`](@ref)) and the resulting
real-space MPS is contracted (`method = :dft_mpo`). Set `npoints` for an
independently sized uniform grid, or pass arbitrary coordinates through `x`;
non-native grids are evaluated by a direct Fourier sum (`method = :direct`).
"""
function mps_to_realspace(state::MPS, sites::AbstractVector{<:Index}, L::Real;
                          normalize_coefficients::Bool=false,
                          x::Union{Nothing,AbstractVector}=nothing,
                          npoints::Union{Nothing,Integer}=nothing,
                          offset::Real=0)
    !isnothing(x) && !isnothing(npoints) &&
        throw(ArgumentError("pass either x or npoints, not both"))
    !isnothing(npoints) && npoints < 2 &&
        throw(ArgumentError("npoints must be at least two"))

    if normalize_coefficients
        state_norm = norm(state)
        iszero(state_norm) && throw(ArgumentError("cannot normalize a zero state"))
        state = state / state_norm
    end
    coefficients = itensor_mps_to_vec(state, sites)
    native_points = 2^length(sites)

    if isnothing(x) && (isnothing(npoints) || npoints == native_points)
        grid = realspace_grid(native_points, L; offset)
        values = realspace_mps_to_vec(fourier_to_realspace_mps(state, sites, L; offset), sites)
        method = :dft_mpo
    else
        grid = isnothing(x) ? realspace_grid(npoints, L; offset) : collect(x)
        values = evaluate_fourier_series(coefficients, grid, L)
        method = :direct
    end
    return (
        x=grid,
        values=values,
        coefficients=coefficients,
        method=method,
    )
end
