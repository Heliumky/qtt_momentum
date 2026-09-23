"""Non-separable multi-dimensional potential MPOs.

For a separable potential `sum_d f_d(x_d)`, `build_separable_hamiltonian` in
`multi_dim.jl` is exact and only ever fits 1D TCI problems. A genuinely
multi-dimensional potential such as `-1/|r|` does not factor, so its
Fourier-basis multiplication operator must be fit directly over the whole
joint register.
"""

"""
    grid_potential_fourier_coefficients(V, qubits_per_dim, half_width_per_dim)

Sample a real-space function `V(x_1,...,x_D)` on its native
`prod(2 .^ qubits_per_dim)`-point grid and return its Fourier series
coefficients as a `D`-dimensional array in QTT/two's-complement mode order
(axis `d` indexed as `freqs(qubits_per_dim[d])`), using the convention
`coefficient(delta) = (1/(2L)) integral(V(x)*exp(i*pi*delta*x/L), x)` (per
dimension) that this package uses for multiplication-operator matrix
elements. This is the exact discrete analogue of that integral for the
sampled function; it is *not* an approximation beyond ordinary grid
aliasing, which is negligible whenever the grid already resolves `V`.
"""
function grid_potential_fourier_coefficients(V, qubits_per_dim::AbstractVector{<:Integer},
                                             half_width_per_dim::AbstractVector{<:Real})
    D = length(qubits_per_dim)
    length(half_width_per_dim) == D ||
        throw(DimensionMismatch("one half-width is required per dimension"))

    grids = [realspace_grid(2^qubits_per_dim[d], half_width_per_dim[d]) for d in 1:D]
    samples = [
        ComplexF64(V(ntuple(d -> grids[d][point[d]], D)...))
        for point in CartesianIndices(Tuple(2 .^ qubits_per_dim))
    ]
    coefficients = ifft(samples)
    for axis in 1:D
        M = size(coefficients, axis)
        shift = reshape((-1.0) .^ freqs(trailing_zeros(M)), ntuple(i -> i == axis ? M : 1, D))
        coefficients = coefficients .* shift
    end
    return coefficients
end

"""
    coefficient_mpo_tci_nd(coefficient, qubits_per_dim; tolerance=1e-10,
                           maxbonddim=200, return_diagnostics=false)

Directly fit a `D`-dimensional multiplication operator
`W[n_out,n_in] = coefficient(n_in_1-n_out_1, ..., n_in_D-n_out_D)` as an MPO
over the joint `sum(qubits_per_dim)`-site register (dimension `d` occupying
`qubits_per_dim[d]` contiguous sites, dimension 1 first, site 1 within each
block its LSB — the same layout `build_separable_hamiltonian`/`embed_mpo`
use). Each site uses a paired local `(output_bit,input_bit)` index of
dimension four, exactly as in `coefficient_mpo_tci`.
"""
function coefficient_mpo_tci_nd(coefficient, qubits_per_dim::AbstractVector{<:Integer};
                                tolerance::Real=1e-10,
                                maxbonddim::Integer=200,
                                return_diagnostics::Bool=false)
    D = length(qubits_per_dim)
    D >= 1 || throw(ArgumentError("qubits_per_dim cannot be empty"))
    all(>(0), qubits_per_dim) || throw(ArgumentError("qubits_per_dim entries must be positive"))

    N = sum(qubits_per_dim)
    block_ends = cumsum(qubits_per_dim)
    block_starts = [1; block_ends[1:(end - 1)] .+ 1]

    coefficient_cache = Dict{NTuple{D,Int},ComplexF64}()
    cached_coefficient(delta) = get!(
        () -> ComplexF64(coefficient(delta...)),
        coefficient_cache,
        delta,
    )

    function operator_element(paired_indices::Vector{Int})
        output_sigma = Vector{Int}(undef, N)
        input_sigma = Vector{Int}(undef, N)
        for site in 1:N
            paired_digit = paired_indices[site] - 1
            output_sigma[site] = mod(paired_digit, 2) + 1
            input_sigma[site] = div(paired_digit, 2) + 1
        end
        delta = ntuple(D) do d
            block = block_starts[d]:block_ends[d]
            sigma_to_n(input_sigma[block]) - sigma_to_n(output_sigma[block])
        end
        return cached_coefficient(delta)
    end

    fit = fit_tensor_train(
        operator_element,
        fill(4, N);
        tolerance=tolerance,
        maxbonddim=maxbonddim,
        return_diagnostics=true,
    )
    tensors = [
        reshape(tensor, size(tensor, 1), 2, 2, size(tensor, 3))
        for tensor in fit.tensors
    ]
    return return_diagnostics ? (; tensors, diagnostics=fit.diagnostics) : tensors
end

"""
    potential_mpo_tci_nd(V, qubits_per_dim, half_width_per_dim; kwargs...)

Fit the Fourier-basis multiplication MPO for a real-space potential
`V(x_1,...,x_D)` by combining `grid_potential_fourier_coefficients` (exact
grid-based Fourier coefficients, computed once by FFT) with
`coefficient_mpo_tci_nd` (adaptive TCI fit of the resulting Toeplitz
operator). Unlike a per-dimension analytic recurrence, this works for any
sampleable `V`, not just separable ones.
"""
function potential_mpo_tci_nd(V, qubits_per_dim::AbstractVector{<:Integer},
                              half_width_per_dim::AbstractVector{<:Real};
                              tolerance::Real=1e-10,
                              maxbonddim::Integer=200,
                              return_diagnostics::Bool=false)
    coefficients = grid_potential_fourier_coefficients(V, qubits_per_dim, half_width_per_dim)
    bounds = 2 .^ qubits_per_dim
    coefficient(delta::Vararg{Integer}) = coefficients[
        CartesianIndex(ntuple(d -> mod(delta[d], bounds[d]) + 1, length(qubits_per_dim)))
    ]
    return coefficient_mpo_tci_nd(
        coefficient, qubits_per_dim;
        tolerance=tolerance, maxbonddim=maxbonddim, return_diagnostics=return_diagnostics,
    )
end
