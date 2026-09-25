"""Discrete Fourier transform as a low-rank MPO and Fourier-space potentials.

For the normalized basis `phi_n(x) = exp(i*pi*n*x/L)/sqrt(2L)` on the grid
`x_j = -L + (j + offset)*2L/M`, the unitary that maps grid samples to Fourier
coefficients is

```
U[n, j] = exp(-i*pi*n*x_j/L) / sqrt(M)
        = Phi(n) * exp(-2*pi*i*k*j/M) / sqrt(M),        k = n mod M,
Phi(n)  = prod_l exp(i*pi*w_l*b_l*(1 - 2*offset/M)),
```

where `n = sum_l w_l*b_l` with two's-complement bit weights `w_l`. The middle
factor is the standard DFT, built by `QuanticsTCI.quanticsfouriermpo`
(Chen–Lindsey interpolative construction, bond dimension ~10-15 at 1e-14).
Its output legs are LSB-first and its input legs MSB-first, which is exactly
the Fourier register / real-space register pair of this package, so no bit
reversal is needed. `Phi` is a product of one-site phases (bond dimension 1)
and is absorbed into the output legs.

A multiplication operator by `V(x)` in the Fourier basis is then
`V_k = U * diag(V(x_j)) * U^dagger`, whose matrix elements are
`V_k[n_out, n_in] = (1/M) * sum_j V(x_j) * exp(i*pi*(n_in - n_out)*x_j/L)`,
the discrete (grid-quadrature) analogue of the analytic coefficient
`(1/2L) * integral(V(x) exp(i*pi*(n_in - n_out)*x/L), x)`.
"""

"""
    fourier_transform_mpo(N, L; offset=0, tolerance=1e-14, maxbonddim=32, K=25)

Array MPO of the unitary `U` described in the file header, with tensors of
shape `(left_bond, output_dim=2, input_dim=2, right_bond)`: output legs are
Fourier-register bits (site 1 = LSB), input legs are real-space-register bits
(site 1 = MSB). `tolerance`, `maxbonddim`, and `K` are passed to
`quanticsfouriermpo`; the default reaches machine precision.
"""
function fourier_transform_mpo(N::Integer, L::Real;
                               offset::Real=0,
                               tolerance::Real=1e-14,
                               maxbonddim::Integer=32,
                               K::Integer=25)
    N >= 2 || throw(ArgumentError("the DFT MPO needs N >= 2"))
    L > 0 || throw(ArgumentError("L must be positive"))
    0 <= offset < 1 || throw(ArgumentError("offset must lie in [0, 1)"))

    dft = quanticsfouriermpo(
        Int(N);
        sign=-1.0,
        tolerance=Float64(tolerance),
        maxbonddim=Int(maxbonddim),
        K=Int(K),
        normalize=true,
    )
    tensors = [Array{ComplexF64,4}(tensor) for tensor in TCI.sitetensors(dft)]
    M = 2^N
    for (site, weight) in enumerate(frequency_weights(N))
        phase = cis(pi * weight * (1 - 2offset / M))
        tensors[site][:, 2, :, :] .*= phase
    end
    return tensors
end

"""Adjoint of an array MPO: `W†[l, a, b, r] = conj(W[l, b, a, r])`."""
adjoint_mpo(tensors::AbstractVector{<:AbstractArray{<:Number,4}}) =
    [conj(permutedims(tensor, (1, 3, 2, 4))) for tensor in tensors]

"""
    grid_register_order(N)

Permutation `p` with `p[c] = j + 1`, where `c` is the column-major linear
index of the real-space register bits (site 1 fastest) and `j` the grid index
(site 1 = MSB). Dense vectors/matrices over the real-space register are
brought to natural grid order with `v[invperm(p)]`, i.e. `v_grid[p] = v`.
"""
function grid_register_order(N::Integer)
    N >= 1 || throw(ArgumentError("N must be positive"))
    return [
        sum((((c >> (site - 1)) & 1) << (N - site) for site in 1:N); init=0) + 1
        for c in 0:(2^N - 1)
    ]
end

"""
    dense_fourier_matrix(N, L; offset=0)

Dense `U[n, j]` with rows in Fourier-register order (`freqs(N)`) and columns
in natural grid order. Exponential in `N`; intended for tests.
"""
function dense_fourier_matrix(N::Integer, L::Real; offset::Real=0)
    x = realspace_grid(2^N, L; offset)
    return [exp(-im * pi * n * xj / L) / sqrt(2^N) for n in freqs(N), xj in x]
end

"""
    fourier_space_operator(sites, realspace_arrays, transform_arrays;
                           cutoff=1e-26, maxdim=typemax(Int))

Return the ITensor MPO `U * A * U†` for a real-space array MPO `A` and an
array MPO `U` (e.g. [`fourier_transform_mpo`](@ref) or its multi-dimensional
analogue). Both products are compressed with the relative SVD `cutoff`.
"""
function fourier_space_operator(sites::AbstractVector{<:Index},
                                realspace_arrays::AbstractVector{<:AbstractArray{<:Number,4}},
                                transform_arrays::AbstractVector{<:AbstractArray{<:Number,4}};
                                cutoff::Real=1e-26,
                                maxdim::Integer=typemax(Int))
    transform = array_mpo_to_itensor(sites, transform_arrays)
    inverse = array_mpo_to_itensor(sites, adjoint_mpo(transform_arrays))
    realspace = array_mpo_to_itensor(sites, realspace_arrays)
    right = apply(realspace, inverse; cutoff=cutoff, maxdim=maxdim)
    return apply(transform, right; cutoff=cutoff, maxdim=maxdim)
end

"""
    potential_mpo_dft(sites, V, N, L; offset=0, pivot_points=Real[],
                      tci_tolerance=1e-12, tci_maxbonddim=200,
                      dft_tolerance=1e-14, dft_maxbonddim=32,
                      cutoff=1e-26, maxdim=typemax(Int), tci_kwargs...)

Fourier-basis multiplication MPO of an arbitrary real-space potential `V(x)`:

```
V(x_j)  --TCI-->  real-space MPS  -->  diagonal MPO D
U = fourier_transform_mpo(N, L; offset)
V_k = U * D * U†                        (compressed with `cutoff`)
```

No analytic Fourier coefficient and no dense grid is required; only the grid
points selected by TCI are evaluated. Extra keywords (for instance
`nsearchglobalpivot`, `threaded`) are forwarded to
[`fit_tensor_train`](@ref). Returns a named tuple with the Fourier-space MPO
`potential`, the real-space diagonal array MPO `realspace_arrays`, the DFT
array MPO `transform_arrays`, and the TCI diagnostics.
"""
function potential_mpo_dft(sites::AbstractVector{<:Index}, V, N::Integer, L::Real;
                           offset::Real=0,
                           pivot_points::AbstractVector{<:Real}=Real[],
                           tci_tolerance::Real=1e-12,
                           tci_maxbonddim::Integer=200,
                           dft_tolerance::Real=1e-14,
                           dft_maxbonddim::Integer=32,
                           cutoff::Real=1e-26,
                           maxdim::Integer=typemax(Int),
                           tci_kwargs...)
    length(sites) == N || throw(DimensionMismatch("N must equal the number of sites"))
    fit = realspace_function_mps(
        V, N, L;
        offset=offset,
        pivot_points=pivot_points,
        tolerance=tci_tolerance,
        maxbonddim=tci_maxbonddim,
        return_diagnostics=true,
        tci_kwargs...,
    )
    realspace_arrays = mps_to_diagonal_mpo(fit.tensors)
    transform_arrays = fourier_transform_mpo(
        N, L; offset, tolerance=dft_tolerance, maxbonddim=dft_maxbonddim,
    )
    potential = fourier_space_operator(
        sites, realspace_arrays, transform_arrays; cutoff, maxdim,
    )
    return (;
        potential,
        realspace_arrays,
        transform_arrays,
        realspace_mps_tensors=fit.tensors,
        tci_diagnostics=fit.diagnostics,
    )
end

"""Fourier-basis MPO of `scale*x^m` via [`potential_mpo_dft`](@ref)."""
function power_potential_mpo_dft(sites::AbstractVector{<:Index}, m::Integer,
                                 N::Integer, L::Real; scale::Number=1, kwargs...)
    m >= 0 || throw(ArgumentError("m must be nonnegative"))
    return potential_mpo_dft(sites, x -> scale * x^m, N, L; kwargs...)
end
