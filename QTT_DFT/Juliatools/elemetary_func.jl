"""Index conventions, real-space grids, and TCI fits of real-space functions.

Two registers are used throughout this package:

- **Fourier (k) register**: site 1 is the least-significant bit of the mode
  number and the last site is the two's-complement sign bit, exactly as in the
  analytic `QTTFourier` package. All Hamiltonians and ground states live here.
- **Real-space (x) register**: site 1 is the most-significant bit of the grid
  index `j`, `x_j = -L + (j + offset) * 2L/M`. Diagonal real-space operators
  such as a potential `V(x_j)` are fitted by TCI in this register, where smooth
  functions have low quantics rank. The DFT MPO in `dft_mpo.jl` maps this
  register onto the Fourier register without any bit reversal.
"""

"""Convert an unsigned `N`-bit integer to its signed two's-complement value."""
function twos_complement(index::Integer, N::Integer)
    N >= 1 || throw(ArgumentError("N must be positive"))
    0 <= index < 2^N || throw(ArgumentError("index must lie in 0:$(2^N - 1)"))
    return index >= 2^(N - 1) ? index - 2^N : index
end

"""Fourier mode numbers in QTT storage order (site 1 is the LSB)."""
freqs(N::Integer) = [twos_complement(index, N) for index in 0:(2^N - 1)]

"""Map 1-based binary physical indices (site 1 = LSB) to a signed Fourier mode."""
function sigma_to_n(sigma::AbstractVector{<:Integer})
    isempty(sigma) && throw(ArgumentError("sigma cannot be empty"))
    all(index -> index == 1 || index == 2, sigma) ||
        throw(ArgumentError("every physical index must be 1 or 2"))

    N = length(sigma)
    unsigned_part = sum(
        (sigma[site] - 1) * (1 << (site - 1)) for site in 1:(N - 1);
        init=0,
    )
    sign_part = (sigma[N] - 1) * (1 << (N - 1))
    return unsigned_part - sign_part
end

"""Map 1-based binary physical indices (site 1 = MSB) to a grid index `0:2^N-1`."""
function sigma_to_grid_index(sigma::AbstractVector{<:Integer})
    isempty(sigma) && throw(ArgumentError("sigma cannot be empty"))
    index = 0
    for bit in sigma
        (bit == 1 || bit == 2) || throw(ArgumentError("every physical index must be 1 or 2"))
        index = 2index + (bit - 1)
    end
    return index
end

"""Inverse of [`sigma_to_grid_index`](@ref): MSB-first 1-based bits of `index`."""
function grid_index_to_sigma(index::Integer, N::Integer)
    0 <= index < 2^N || throw(ArgumentError("index must lie in 0:$(2^N - 1)"))
    return [((index >> (N - site)) & 1) + 1 for site in 1:N]
end

"""Reshape a dense length-`2^N` vector into `N` binary physical dimensions."""
function vec_to_site_tensor(values::AbstractVector, N::Integer)
    length(values) == 2^N || throw(DimensionMismatch("expected a vector of length 2^N"))
    return reshape(values, ntuple(_ -> 2, N))
end

"""
    realspace_grid(M, L; offset=0)

Uniform periodic grid `x_j = -L + (j + offset) * 2L/M`, `j = 0:M-1`. With
`offset=0` the grid contains `x=0` (for even `M`); `offset=0.5` gives the
cell-centred grid, which avoids sampling a singularity at the origin.
"""
function realspace_grid(M::Integer, L::Real; offset::Real=0)
    M >= 1 || throw(ArgumentError("M must be positive"))
    L > 0 || throw(ArgumentError("L must be positive"))
    0 <= offset < 1 || throw(ArgumentError("offset must lie in [0, 1)"))
    return -L .+ ((0:(M - 1)) .+ offset) .* (2L / M)
end

"""Grid coordinate for MSB-first bits `sigma` of an `N`-qubit axis."""
grid_coordinate(sigma::AbstractVector{<:Integer}, L::Real; offset::Real=0) =
    -L + (sigma_to_grid_index(sigma) + offset) * (2L / 2^length(sigma))

"""MSB-first bits of the grid point closest to the coordinate `x`."""
function nearest_grid_sigma(x::Real, N::Integer, L::Real; offset::Real=0)
    M = 2^N
    index = mod(round(Int, (x + L) * M / (2L) - offset), M)
    return grid_index_to_sigma(index, N)
end

"""
    fit_tensor_train(target, local_dimensions; tolerance=1e-10,
                     maxbonddim=200, initialpivots=nothing,
                     nsearchglobalpivot=5, maxnglobalpivot=5,
                     tolmarginglobalsearch=10.0, normalizeerror=true,
                     pivotsearch=:full, threaded=false,
                     ntrueerrorsearch=100, return_diagnostics=false)

Fit a scalar function of a discrete index string as a tensor train with
`TCI.crossinterpolate2`. TCI chooses the queried entries; the full tensor is
never formed. Returned tensors have shape `(left_bond, physical_dim,
right_bond)`.

Pivot handling:

- Every entry of `initialpivots` (default: the all-ones index) is first
  improved by `TCI.optfirstpivot`; zero-valued and duplicate pivots are
  dropped, and all remaining pivots seed the TCI. Supplying pivots near the
  features of the target (maxima, singularities, separate peaks) prevents
  TCI from missing them.
- `nsearchglobalpivot`, `maxnglobalpivot`, and `tolmarginglobalsearch` control
  TCI's global pivot search, which adds index strings where the current
  interpolation error is large even if the local sweeps did not find them.
- `normalizeerror=true` makes `tolerance` relative to the largest sampled
  `|target|`; `false` makes it absolute.

Evaluation: by default the target is wrapped in `TCI.CachedFunction`, so
repeated TCI queries never re-evaluate it. With `threaded=true` it is wrapped
in `TCI.ThreadedBatchEvaluator` instead (no cache, multithreaded batch
evaluation), which pays off for expensive, thread-safe targets.

With `return_diagnostics=true`, return `(; tensors, diagnostics)`. Besides
TCI's own pivot-error history, the diagnostics contain a global error
estimate from `TCI.estimatetrueerror` (`ntrueerrorsearch` random greedy
searches; set it to 0 to skip), normalized in the same way as the tolerance.
"""
function fit_tensor_train(target, local_dimensions::AbstractVector{<:Integer};
                          tolerance::Real=1e-10,
                          maxbonddim::Integer=200,
                          initialpivots::Union{Nothing,AbstractVector{<:AbstractVector{<:Integer}}}=nothing,
                          nsearchglobalpivot::Integer=5,
                          maxnglobalpivot::Integer=5,
                          tolmarginglobalsearch::Real=10.0,
                          normalizeerror::Bool=true,
                          pivotsearch::Symbol=:full,
                          threaded::Bool=false,
                          ntrueerrorsearch::Integer=100,
                          return_diagnostics::Bool=false)
    isempty(local_dimensions) && throw(ArgumentError("local_dimensions cannot be empty"))
    all(>(0), local_dimensions) || throw(ArgumentError("local dimensions must be positive"))
    tolerance > 0 || throw(ArgumentError("tolerance must be positive"))
    maxbonddim >= 1 || throw(ArgumentError("maxbonddim must be positive"))
    ntrueerrorsearch >= 0 || throw(ArgumentError("ntrueerrorsearch must be nonnegative"))

    dimensions = Int.(local_dimensions)
    N = length(dimensions)
    scalar_target(indices::Vector{Int}) = ComplexF64(target(indices))
    evaluations = Threads.Atomic{Int}(0)
    counted_target(indices::Vector{Int}) = (Threads.atomic_add!(evaluations, 1); scalar_target(indices))
    black_box = threaded ?
        TCI.ThreadedBatchEvaluator{ComplexF64}(counted_target, dimensions) :
        TCI.CachedFunction{ComplexF64}(counted_target, dimensions)

    starting_pivots = isnothing(initialpivots) ? [ones(Int, N)] :
                      [Int.(pivot) for pivot in initialpivots]
    isempty(starting_pivots) && throw(ArgumentError("initialpivots cannot be empty"))
    for pivot in starting_pivots
        length(pivot) == N || throw(DimensionMismatch("every initial pivot must have length N"))
        all(1 <= pivot[site] <= dimensions[site] for site in 1:N) ||
            throw(ArgumentError("initial pivot entries exceed their local dimensions"))
    end
    optimized_pivots = unique([
        TCI.optfirstpivot(black_box, dimensions, pivot) for pivot in starting_pivots
    ])
    filter!(pivot -> !iszero(black_box(pivot)), optimized_pivots)
    isempty(optimized_pivots) && throw(ArgumentError(
        "TCI could not find a nonzero pivot; pass nonzero initialpivots explicitly",
    ))

    tci, ranks, errors = TCI.crossinterpolate2(
        ComplexF64,
        black_box,
        dimensions,
        optimized_pivots;
        tolerance=Float64(tolerance),
        maxbonddim=Int(maxbonddim),
        nsearchglobalpivot=Int(nsearchglobalpivot),
        maxnglobalpivot=Int(maxnglobalpivot),
        tolmarginglobalsearch=Float64(tolmarginglobalsearch),
        normalizeerror=normalizeerror,
        pivotsearch=pivotsearch,
    )
    tensors = TCI.sitetensors(tci)
    return_diagnostics || return tensors

    scale = normalizeerror ? max(tci.maxsamplevalue, eps()) : 1.0
    true_error = if ntrueerrorsearch == 0
        NaN
    else
        found = TCI.estimatetrueerror(TCI.TensorTrain(tci), scalar_target; nsearch=Int(ntrueerrorsearch))
        (isempty(found) ? 0.0 : first(found)[2]) / scale
    end
    diagnostics = (
        requested_tolerance=Float64(tolerance),
        estimated_relative_error=isempty(errors) ? 0.0 : last(errors),
        global_error_estimate=true_error,
        error_history=collect(errors),
        bond_dimension_history=collect(ranks),
        initial_pivots=optimized_pivots,
        max_sample_value=Float64(tci.maxsamplevalue),
        target_evaluations=evaluations[],
    )
    return (; tensors, diagnostics)
end

"""Fit a function of `N` binary indices as a QTT/MPS."""
function fit_qtt_mps(target, N::Integer; kwargs...)
    N >= 1 || throw(ArgumentError("N must be positive"))
    return fit_tensor_train(target, fill(2, N); kwargs...)
end

"""
    realspace_function_mps(f, N, L; offset=0, pivot_points=Real[], kwargs...)

Fit `f(x_j)` on the `2^N`-point grid [`realspace_grid`](@ref) as an MPS in the
real-space register (site 1 = MSB of `j`). Only the grid points TCI asks for
are evaluated. The grid points nearest to each coordinate in `pivot_points`
are added to the initial pivots (for example `0.0` for a potential that is
extremal at the origin); the boundary point `x=-L` is always included.
Remaining keywords go to [`fit_tensor_train`](@ref).
"""
function realspace_function_mps(f, N::Integer, L::Real;
                                offset::Real=0,
                                pivot_points::AbstractVector{<:Real}=Real[],
                                kwargs...)
    N >= 1 || throw(ArgumentError("N must be positive"))
    L > 0 || throw(ArgumentError("L must be positive"))
    pivots = Vector{Int}[ones(Int, N)]
    append!(pivots, [nearest_grid_sigma(x, N, L; offset) for x in pivot_points])
    return fit_qtt_mps(
        sigma -> f(grid_coordinate(sigma, L; offset)),
        N;
        initialpivots=unique(pivots),
        kwargs...,
    )
end

"""
    power_potential_mps(m, N, L; scale=1, offset=0, kwargs...)

Real-space MPS of `scale*x^m` sampled on the grid. A degree-`m` polynomial has
quantics rank at most `m+1`, so TCI reproduces it exactly at that rank.
"""
function power_potential_mps(m::Integer, N::Integer, L::Real;
                             scale::Number=1, offset::Real=0, kwargs...)
    m >= 0 || throw(ArgumentError("m must be nonnegative"))
    return realspace_function_mps(x -> scale * x^m, N, L; offset, kwargs...)
end

"""
    mps_to_diagonal_mpo(tensors)

Promote real-space MPS tensors `A[l,s,r]` to the diagonal MPO
`W[l,s,s,r] = A[l,s,r]` (shape `(left_bond, output_dim, input_dim,
right_bond)`), i.e. the multiplication operator by the fitted function.
"""
function mps_to_diagonal_mpo(tensors::AbstractVector{<:AbstractArray{T,3}}) where {T}
    return map(tensors) do tensor
        left, physical, right = size(tensor)
        diagonal = zeros(T, left, physical, physical, right)
        for s in 1:physical
            diagonal[:, s, s, :] = tensor[:, s, :]
        end
        diagonal
    end
end
