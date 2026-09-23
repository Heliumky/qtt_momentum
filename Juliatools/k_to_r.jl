"""Conversion from Fourier-coefficient MPS data to real-space functions."""

"""Contract an ITensor MPS to coefficients in little-endian QTT order."""
function itensor_mps_to_vec(state::MPS, sites::AbstractVector{<:Index})
    length(state) == length(sites) || throw(DimensionMismatch("one site index is required per MPS tensor"))
    contracted = state[1]
    for site in 2:length(state)
        contracted *= state[site]
    end
    return vec(Array(contracted, sites...))
end

"""Uniform periodic grid `x_j = -L + 2L*j/M`, excluding the duplicated endpoint."""
function realspace_grid(M::Integer, L::Real)
    M >= 1 || throw(ArgumentError("M must be positive"))
    L > 0 || throw(ArgumentError("L must be positive"))
    return -L .+ (0:(M - 1)) .* (2L / M)
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
    coeffs_to_realspace(coefficients, L)

Evaluate Fourier coefficients on their matching `M`-point periodic grid using
an inverse FFT. This is the fast-grid equivalent of [`evaluate_fourier_series`](@ref).
"""
function coeffs_to_realspace(coefficients::AbstractVector{<:Number}, L::Real)
    M = length(coefficients)
    ispow2(M) || throw(ArgumentError("the coefficient count must be a power of two"))
    L > 0 || throw(ArgumentError("L must be positive"))
    mode_numbers = freqs(trailing_zeros(M))
    shifted_coefficients = coefficients .* (-1.0) .^ mode_numbers
    return M .* ifft(shifted_coefficients) ./ sqrt(2L)
end

"""
    mps_to_realspace(state, sites, L; normalize_coefficients=false,
                     x=nothing, npoints=nothing)

Contract a Fourier-coefficient MPS and reconstruct its real-space values.
Returns a named tuple `(x, values, coefficients)` so plotting and diagnostics
do not need to repeat the contraction.

With neither `x` nor `npoints`, values are evaluated on the native `2^N` grid
with an inverse FFT. Set `npoints` for an independently sized uniform grid, or
pass arbitrary coordinates through `x`; non-native grids are evaluated by a
direct Fourier sum.
"""
function mps_to_realspace(state::MPS, sites::AbstractVector{<:Index}, L::Real;
                          normalize_coefficients::Bool=false,
                          x::Union{Nothing,AbstractVector}=nothing,
                          npoints::Union{Nothing,Integer}=nothing)
    !isnothing(x) && !isnothing(npoints) &&
        throw(ArgumentError("pass either x or npoints, not both"))
    !isnothing(npoints) && npoints < 2 &&
        throw(ArgumentError("npoints must be at least two"))

    coefficients = itensor_mps_to_vec(state, sites)
    if normalize_coefficients
        coefficient_norm = sqrt(sum(abs2, coefficients))
        iszero(coefficient_norm) && throw(ArgumentError("cannot normalize a zero state"))
        coefficients = coefficients ./ coefficient_norm
    end

    if !isnothing(x)
        grid = collect(x)
        values = evaluate_fourier_series(coefficients, grid, L)
        method = :direct
    else
        point_count = isnothing(npoints) ? length(coefficients) : npoints
        grid = realspace_grid(point_count, L)
        if point_count == length(coefficients)
            values = coeffs_to_realspace(coefficients, L)
            method = :ifft
        else
            values = evaluate_fourier_series(coefficients, grid, L)
            method = :direct
        end
    end
    return (
        x=grid,
        values=values,
        coefficients=coefficients,
        method=method,
    )
end
