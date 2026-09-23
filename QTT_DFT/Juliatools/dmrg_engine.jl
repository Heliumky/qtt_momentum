"""ITensor conversion, Hamiltonian assembly, and DMRG utilities."""

"""Convert array MPO tensors to an ITensor `MPO`, dropping boundary links."""
function array_mpo_to_itensor(sites::AbstractVector{<:Index},
                              tensors::AbstractVector{<:AbstractArray{T,4}}) where {T}
    N = length(sites)
    length(tensors) == N || throw(DimensionMismatch("one MPO tensor is required per site"))
    links = [Index(size(tensors[site], 4), "Link,l=$site") for site in 1:(N - 1)]
    itensors = Vector{ITensor}(undef, N)

    for site in 1:N
        tensor = tensors[site]
        size(tensor, 2) == dim(sites[site]) == size(tensor, 3) ||
            throw(DimensionMismatch("MPO physical dimensions do not match site $site"))
        if N == 1
            itensors[site] = ITensor(dropdims(tensor; dims=(1, 4)), sites[site]', sites[site])
        elseif site == 1
            itensors[site] = ITensor(dropdims(tensor; dims=1), sites[site]', sites[site], links[1])
        elseif site == N
            itensors[site] = ITensor(dropdims(tensor; dims=4), links[N - 1], sites[site]', sites[site])
        else
            itensors[site] = ITensor(tensor, links[site - 1], sites[site]', sites[site], links[site])
        end
    end
    return MPO(itensors)
end

"""Convert array MPS tensors to an ITensor `MPS`, dropping boundary links."""
function array_mps_to_itensor(sites::AbstractVector{<:Index},
                              tensors::AbstractVector{<:AbstractArray{T,3}}) where {T}
    N = length(sites)
    length(tensors) == N || throw(DimensionMismatch("one MPS tensor is required per site"))
    links = [Index(size(tensors[site], 3), "Link,l=$site") for site in 1:(N - 1)]
    itensors = Vector{ITensor}(undef, N)

    for site in 1:N
        tensor = tensors[site]
        size(tensor, 2) == dim(sites[site]) ||
            throw(DimensionMismatch("MPS physical dimension does not match site $site"))
        if N == 1
            itensors[site] = ITensor(dropdims(tensor; dims=(1, 3)), sites[site])
        elseif site == 1
            itensors[site] = ITensor(dropdims(tensor; dims=1), sites[site], links[1])
        elseif site == N
            itensors[site] = ITensor(dropdims(tensor; dims=3), links[N - 1], sites[site])
        else
            itensors[site] = ITensor(tensor, links[site - 1], sites[site], links[site])
        end
    end
    return MPS(itensors)
end

"""Convert an ITensor `MPO` back to array tensors `(left_bond, output_dim, input_dim, right_bond)`."""
function itensor_mpo_to_arrays(operator::MPO, sites::AbstractVector{<:Index})
    N = length(sites)
    length(operator) == N || throw(DimensionMismatch("site count does not match MPO"))
    return map(1:N) do site
        tensor = operator[site]
        left = site == 1 ? nothing : commonind(operator[site - 1], tensor)
        right = site == N ? nothing : commonind(tensor, operator[site + 1])
        output, input = sites[site]', sites[site]
        if N == 1
            data = Array(tensor, output, input)
        elseif site == 1
            data = Array(tensor, output, input, right)
        elseif site == N
            data = Array(tensor, left, output, input)
        else
            data = Array(tensor, left, output, input, right)
        end
        left_dim = isnothing(left) ? 1 : dim(left)
        right_dim = isnothing(right) ? 1 : dim(right)
        reshape(ComplexF64.(data), left_dim, dim(output), dim(input), right_dim)
    end
end

# Shape conventions used throughout the project:
# MPS A[i] = (left_bond, physical_dim, right_bond)
# MPO W[i] = (left_bond, output_dim, input_dim, right_bond)

"""Shapes of array MPS tensors as `(left_bond, physical_dim, right_bond)`."""
mps_shapes(tensors::AbstractVector{<:AbstractArray{<:Number,3}}) = Tuple.(size.(tensors))

"""Shapes of array MPO tensors as `(left_bond, output_dim, input_dim, right_bond)`."""
mpo_shapes(tensors::AbstractVector{<:AbstractArray{<:Number,4}}) = Tuple.(size.(tensors))

"""Internal bond dimensions of an array MPS."""
mps_bond_dims(tensors::AbstractVector{<:AbstractArray{<:Number,3}}) =
    [size(tensors[site], 3) for site in 1:(length(tensors) - 1)]

"""Internal bond dimensions of an array MPO."""
mpo_bond_dims(tensors::AbstractVector{<:AbstractArray{<:Number,4}}) =
    [size(tensors[site], 4) for site in 1:(length(tensors) - 1)]

mps_bond_dims(state::MPS) = collect(linkdims(state))
mpo_bond_dims(operator::MPO) = collect(linkdims(operator))

function mps_shapes(state::MPS, sites::AbstractVector{<:Index})
    length(state) == length(sites) || throw(DimensionMismatch("site count does not match MPS"))
    bonds = mps_bond_dims(state)
    N = length(state)
    return [
        (
            site == 1 ? 1 : bonds[site - 1],
            dim(sites[site]),
            site == N ? 1 : bonds[site],
        )
        for site in 1:N
    ]
end

function mpo_shapes(operator::MPO, sites::AbstractVector{<:Index})
    length(operator) == length(sites) || throw(DimensionMismatch("site count does not match MPO"))
    bonds = mpo_bond_dims(operator)
    N = length(operator)
    return [
        (
            site == 1 ? 1 : bonds[site - 1],
            dim(sites[site]),
            dim(sites[site]),
            site == N ? 1 : bonds[site],
        )
        for site in 1:N
    ]
end


max_bond_dim(tensors::AbstractVector{<:AbstractArray{<:Number,3}}) =
    maximum(mps_bond_dims(tensors); init=1)
max_bond_dim(tensors::AbstractVector{<:AbstractArray{<:Number,4}}) =
    maximum(mpo_bond_dims(tensors); init=1)
max_bond_dim(state::MPS) = maximum(mps_bond_dims(state); init=1)
max_bond_dim(operator::MPO) = maximum(mpo_bond_dims(operator); init=1)

"""
    build_hamiltonian(sites, potential, N, L)

Combine an already constructed Fourier-space potential (an ITensor `MPO` or an
array MPO) with the exact Fourier kinetic MPO. Returns a named tuple
containing the Hamiltonian and both components.
"""
function build_hamiltonian(sites::AbstractVector{<:Index}, potential::MPO,
                           N::Integer, L::Real)
    length(sites) == N || throw(DimensionMismatch("N must equal the number of sites"))
    length(potential) == N ||
        throw(DimensionMismatch("one potential MPO tensor is required per site"))
    kinetic_arrays = kinetic_mpo(N, L)
    kinetic = array_mpo_to_itensor(sites, kinetic_arrays)

    # A direct sum is exact. Relative SVD truncation can otherwise erase the
    # potential when the high-frequency kinetic scale is much larger.
    hamiltonian = +(kinetic, potential; alg="directsum")
    return (; hamiltonian, kinetic, potential, kinetic_arrays)
end

function build_hamiltonian(sites::AbstractVector{<:Index},
                           potential_arrays::AbstractVector{<:AbstractArray{<:Number,4}},
                           N::Integer, L::Real)
    length(potential_arrays) == N ||
        throw(DimensionMismatch("one potential MPO tensor is required per site"))
    potential = array_mpo_to_itensor(sites, potential_arrays)
    return (; build_hamiltonian(sites, potential, N, L)..., potential_arrays)
end

"""
    build_dft_hamiltonian(sites, V, N, L; kwargs...)

Build `-1/2*d^2/dx^2 + V(x)` in the Fourier basis. `V(x_j)` is fitted by TCI
in real space and rotated to Fourier space with the DFT MPO
([`potential_mpo_dft`](@ref), which receives all keywords).
"""
function build_dft_hamiltonian(sites::AbstractVector{<:Index}, V, N::Integer, L::Real;
                               kwargs...)
    fit = potential_mpo_dft(sites, V, N, L; kwargs...)
    model = build_hamiltonian(sites, fit.potential, N, L)
    return (;
        model...,
        realspace_potential_arrays=fit.realspace_arrays,
        transform_arrays=fit.transform_arrays,
        tci_diagnostics=fit.tci_diagnostics,
    )
end

"""Build `-1/2*d^2/dx^2 + scale*x^m` with the DFT potential path."""
function build_power_hamiltonian(sites::AbstractVector{<:Index}, m::Integer,
                                 N::Integer, L::Real; scale::Number=1, kwargs...)
    m >= 0 || throw(ArgumentError("m must be nonnegative"))
    return build_dft_hamiltonian(sites, x -> scale * x^m, N, L; kwargs...)
end

"""
    ground_state(H, sites; kwargs...)

Run two-site DMRG from a seeded random MPS, or from `initial_state` if given
(for example a physical guess built in real space with
[`realspace_state_mps_nd`](@ref)). `noise` is passed to ITensor's `dmrg`
(default: no noise). With `return_diagnostics=true`, a third return value
contains the requested cutoff and the observer's maximum truncation error for
each sweep.

Random starts can get stuck: for the `[10, 10]`-qubit 2D Coulomb problem
(kinetic scale ~5e3 against a binding energy ~2) a random start converges to
E = -0.673 in a bond-5 local minimum, while an `exp(-r)` real-space guess
reaches the ground state E = -1.885.
"""
function ground_state(H::MPO, sites::AbstractVector{<:Index};
                      nsweeps::Integer=15,
                      maxdim::Integer=64,
                      cutoff::Real=1e-12,
                      seed::Integer=1,
                      eltype::Type=ComplexF64,
                      outputlevel::Integer=1,
                      initial_state::Union{Nothing,MPS}=nothing,
                      noise=0.0,
                      return_diagnostics::Bool=false)
    nsweeps >= 1 || throw(ArgumentError("nsweeps must be positive"))
    maxdim >= 1 || throw(ArgumentError("maxdim must be positive"))
    seed!(seed)
    initial = isnothing(initial_state) ? random_mps(eltype, sites; linkdims=2) : initial_state
    observer = return_diagnostics ? DMRGObserver() : NoObserver()
    energy, state = dmrg(
        H,
        initial;
        nsweeps=nsweeps,
        maxdim=maxdim,
        cutoff=cutoff,
        outputlevel=outputlevel,
        observer=observer,
        noise=noise,
    )
    if !return_diagnostics
        return energy, state
    end
    errors = collect(truncerrors(observer))
    diagnostics = (
        requested_cutoff=Float64(cutoff),
        truncation_error_by_sweep=errors,
        max_truncation_error=maximum(errors; init=0.0),
    )
    return energy, state, diagnostics
end

"""Expectation value `<psi|H|psi>` for a normalized MPS."""
expect_mpo(psi::MPS, operator::MPO) = real(inner(psi', operator, psi))

# Dense helpers below are intentionally kept for tests and small-system
# diagnostics. Production QTT paths should remain compressed.

function _contract_bond(left::AbstractArray, right::AbstractArray{T,3}) where {T}
    left_shape = size(left)
    matrix = reshape(left, prod(left_shape[1:end-1]), left_shape[end])
    left_rank, physical_dim, right_rank = size(right)
    size(matrix, 2) == left_rank || throw(DimensionMismatch("adjacent links do not match"))
    product = matrix * reshape(right, left_rank, physical_dim * right_rank)
    return reshape(product, left_shape[1:end-1]..., physical_dim, right_rank)
end

"""Factor a dense site tensor into an array MPS by successive SVDs."""
function dense_to_mps(tensor::AbstractArray{T,N};
                      tol::Real=1e-13, chi_max::Integer=512) where {T,N}
    all(==(2), size(tensor)) || throw(DimensionMismatch("all physical dimensions must equal 2"))
    mps = Vector{Array{T,3}}(undef, N)
    remainder = reshape(tensor, 1, :)
    left_rank = 1

    for site in 1:(N - 1)
        remainder = reshape(remainder, left_rank * 2, :)
        factorization = svd(remainder)
        threshold = isempty(factorization.S) ? 0 : tol * factorization.S[1]
        rank = min(max(1, count(>(threshold), factorization.S)), chi_max)
        mps[site] = reshape(Array(factorization.U[:, 1:rank]), left_rank, 2, rank)
        remainder = Diagonal(factorization.S[1:rank]) * factorization.Vt[1:rank, :]
        left_rank = rank
    end
    mps[N] = reshape(Array(remainder), left_rank, 2, 1)
    return mps
end

"""Contract an array MPS to a dense site tensor. Exponential in site count."""
function mps_to_dense(mps::AbstractVector{<:AbstractArray{T,3}}) where {T}
    isempty(mps) && throw(ArgumentError("mps cannot be empty"))
    result = mps[1]
    for tensor in mps[2:end]
        result = _contract_bond(result, tensor)
    end
    return dropdims(result; dims=(1, ndims(result)))
end

"""Contract an array MPO to a dense matrix. Exponential in site count."""
function mpo_to_dense(mpo::AbstractVector{<:AbstractArray{T,4}}) where {T}
    isempty(mpo) && throw(ArgumentError("mpo cannot be empty"))
    N = length(mpo)
    result = mpo[1]
    for tensor in mpo[2:end]
        old_shape = size(result)
        matrix = reshape(result, prod(old_shape[1:end-1]), old_shape[end])
        left_rank, output_dim, input_dim, right_rank = size(tensor)
        size(matrix, 2) == left_rank || throw(DimensionMismatch("adjacent links do not match"))
        product = matrix * reshape(tensor, left_rank, output_dim * input_dim * right_rank)
        result = reshape(product, old_shape[1:end-1]..., output_dim, input_dim, right_rank)
    end

    result = dropdims(result; dims=(1, ndims(result)))
    outputs = 1:2:(2N)
    inputs = 2:2:(2N)
    return reshape(permutedims(result, (outputs..., inputs...)), 2^N, 2^N)
end

"""Exact direct sum of two compatible array MPOs."""
function mpo_add(first::AbstractVector{<:AbstractArray{Ta,4}},
                 second::AbstractVector{<:AbstractArray{Tb,4}}) where {Ta,Tb}
    length(first) == length(second) || throw(DimensionMismatch("MPO lengths must match"))
    isempty(first) && throw(ArgumentError("MPOs cannot be empty"))
    element_type = promote_type(Ta, Tb)
    N = length(first)
    result = Vector{Array{element_type,4}}(undef, N)

    for site in 1:N
        a, b = first[site], second[site]
        (size(a, 2), size(a, 3)) == (size(b, 2), size(b, 3)) ||
            throw(DimensionMismatch("physical dimensions must match"))
        a_left, output_dim, input_dim, a_right = size(a)
        b_left, _, _, b_right = size(b)
        if N == 1
            result[site] = a + b
        elseif site == 1
            tensor = zeros(element_type, 1, output_dim, input_dim, a_right + b_right)
            tensor[:, :, :, 1:a_right] = a
            tensor[:, :, :, (a_right + 1):end] = b
            result[site] = tensor
        elseif site == N
            tensor = zeros(element_type, a_left + b_left, output_dim, input_dim, 1)
            tensor[1:a_left, :, :, :] = a
            tensor[(a_left + 1):end, :, :, :] = b
            result[site] = tensor
        else
            tensor = zeros(
                element_type,
                a_left + b_left,
                output_dim,
                input_dim,
                a_right + b_right,
            )
            tensor[1:a_left, :, :, 1:a_right] = a
            tensor[(a_left + 1):end, :, :, (a_right + 1):end] = b
            result[site] = tensor
        end
    end
    return result
end
