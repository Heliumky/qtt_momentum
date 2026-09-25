"""Register layouts and separable multi-dimensional Hamiltonians.

A `D`-dimensional problem uses one joint register. Each dimension `d` owns
`qubits_per_dim[d]` sites; "level" `l` of dimension `d` is the Fourier bit of
weight `2^(l-1)` (sign bit at `l = N_d`) in the Fourier register and the
real-space bit of weight `2^(N_d-l)` in the real-space register, so the DFT
MPO of each dimension acts level-by-level on the same sites. Two site orders
are supported:

- `:block` — dimension 1's levels first, then dimension 2's, ... (the layout
  of the analytic `QTTFourier` package). Tensor products between dimensions
  are exact and cost no bond dimension.
- `:interleaved` — level 1 of every dimension, then level 2, ... Different
  dimensions at the same length scale are neighbours. This can lower TCI
  ranks for potentials with strongly coupled scales, but the joint DFT MPO
  `U_1 ⊗ U_2` then has the *product* of the 1D DFT bonds (~144 for two 8-qubit
  axes instead of ~12), which makes `U V U†` far more expensive. For the 2D
  Coulomb potential on `[8, 8]` qubits the block layout also has the lower
  TCI rank (37 vs 52), so `:block` is the default.
"""

"""
    RegisterLayout(qubits_per_dim; scheme=:block)

Site assignment of a joint `D`-dimensional register. `positions[d][l]` is the
site of level `l` of dimension `d`; `owner[site] = (d, l)`.
"""
struct RegisterLayout
    qubits_per_dim::Vector{Int}
    scheme::Symbol
    positions::Vector{Vector{Int}}
    owner::Vector{Tuple{Int,Int}}
end

function RegisterLayout(qubits_per_dim::AbstractVector{<:Integer}; scheme::Symbol=:block)
    isempty(qubits_per_dim) && throw(ArgumentError("qubits_per_dim cannot be empty"))
    all(>(0), qubits_per_dim) || throw(ArgumentError("qubits_per_dim entries must be positive"))
    scheme in (:block, :interleaved) ||
        throw(ArgumentError("scheme must be :block or :interleaved"))

    counts = Int.(qubits_per_dim)
    owner = Tuple{Int,Int}[]
    if scheme === :block
        for d in eachindex(counts), level in 1:counts[d]
            push!(owner, (d, level))
        end
    else
        for level in 1:maximum(counts), d in eachindex(counts)
            level <= counts[d] && push!(owner, (d, level))
        end
    end
    positions = [Int[] for _ in counts]
    for (site, (d, level)) in enumerate(owner)
        push!(positions[d], site)
        @assert length(positions[d]) == level
    end
    return RegisterLayout(counts, scheme, positions, owner)
end

RegisterLayout(layout::RegisterLayout) = layout

"""Total number of sites of a layout."""
nsites(layout::RegisterLayout) = length(layout.owner)

"""Sites owned by dimension `d`, ordered by level."""
dimension_sites(layout::RegisterLayout, d::Integer) = layout.positions[d]

"""Identity MPO of `N` qubit sites (bond dimension 1)."""
function identity_mpo(N::Integer; eltype::Type=ComplexF64)
    N >= 1 || throw(ArgumentError("N must be positive"))
    tensor = zeros(eltype, 1, 2, 2, 1)
    tensor[1, 1, 1, 1] = 1
    tensor[1, 2, 2, 1] = 1
    return [copy(tensor) for _ in 1:N]
end

"""Identity on one qubit that carries a bond of dimension `bond` through unchanged."""
function _passthrough_tensor(bond::Integer, eltype::Type)
    tensor = zeros(eltype, bond, 2, 2, bond)
    for a in 1:bond, s in 1:2
        tensor[a, s, s, a] = 1
    end
    return tensor
end

"""
    embed_mpo(component, dim, layout; eltype=ComplexF64)

Embed a single-dimension array MPO `component` (one tensor per level of
dimension `dim`) into the joint register as `I ⊗ ... ⊗ component ⊗ ... ⊗ I`.
Sites of other dimensions become identities; between two sites of `dim` they
carry the component's bond unchanged, so the embedding is exact and has the
component's bond dimensions. `layout` is a [`RegisterLayout`](@ref) or a
`qubits_per_dim` vector (block layout).
"""
function embed_mpo(component::AbstractVector{<:AbstractArray{<:Number,4}},
                   dim::Integer, layout;
                   eltype::Type=ComplexF64)
    layout = layout isa RegisterLayout ? layout : RegisterLayout(layout)
    1 <= dim <= length(layout.qubits_per_dim) || throw(ArgumentError("dim is out of range"))
    length(component) == layout.qubits_per_dim[dim] ||
        throw(DimensionMismatch("component must have qubits_per_dim[dim] sites"))

    result = Vector{Array{eltype,4}}(undef, nsites(layout))
    bond = 1
    for (site, (d, level)) in enumerate(layout.owner)
        if d == dim
            tensor = Array{eltype,4}(component[level])
            size(tensor, 1) == bond ||
                throw(DimensionMismatch("component bonds do not match at level $level"))
            result[site] = tensor
            bond = size(tensor, 4)
        else
            result[site] = _passthrough_tensor(bond, eltype)
        end
    end
    bond == 1 || throw(DimensionMismatch("component must end with a unit bond"))
    return result
end

"""Exact (uncompressed) product `A*B` of two array MPOs on the same sites."""
function mpo_product_exact(first::AbstractVector{<:AbstractArray{Ta,4}},
                           second::AbstractVector{<:AbstractArray{Tb,4}}) where {Ta,Tb}
    length(first) == length(second) || throw(DimensionMismatch("MPO lengths must match"))
    element_type = promote_type(Ta, Tb)
    return map(zip(first, second)) do (a, b)
        al, ao, ai, ar = size(a)
        bl, bo, bi, br = size(b)
        ai == bo || throw(DimensionMismatch("physical dimensions must match"))
        product = zeros(element_type, bl, al, ao, bi, br, ar)
        for s in 1:ai, output in 1:ao, input in 1:bi
            product[:, :, output, input, :, :] .+=
                reshape(b[:, s, input, :], bl, 1, br, 1) .* reshape(a[:, output, s, :], 1, al, 1, ar)
        end
        reshape(product, bl * al, ao, bi, br * ar)
    end
end

"""
    kron_mpo(operators)

Exact product of full-register array MPOs that act on disjoint sets of sites
(for example the outputs of [`embed_mpo`](@ref) for different dimensions).
The bond dimension at each cut is the product of the operators' bonds there;
in the block layout at most one factor is nontrivial at every cut.
"""
kron_mpo(operators::AbstractVector) = reduce(mpo_product_exact, operators)

"""
    fourier_transform_mpo_nd(layout, half_width_per_dim; offsets=zeros(D), kwargs...)

Joint DFT MPO `U = ⊗_d U_d` mapping the real-space register of `layout` to its
Fourier register. Keywords go to [`fourier_transform_mpo`](@ref).
"""
function fourier_transform_mpo_nd(layout, half_width_per_dim::AbstractVector{<:Real};
                                  offsets::AbstractVector{<:Real}=zeros(length(half_width_per_dim)),
                                  kwargs...)
    layout = layout isa RegisterLayout ? layout : RegisterLayout(layout)
    D = length(layout.qubits_per_dim)
    (length(half_width_per_dim) == D && length(offsets) == D) ||
        throw(DimensionMismatch("one half-width and one offset are required per dimension"))
    factors = [
        embed_mpo(
            fourier_transform_mpo(layout.qubits_per_dim[d], half_width_per_dim[d];
                                  offset=offsets[d], kwargs...),
            d, layout,
        )
        for d in 1:D
    ]
    return kron_mpo(factors)
end

"""
    build_separable_hamiltonian(sites, qubits_per_dim, half_width_per_dim;
                                potentials=nothing, powers=fill(2, D),
                                scales=fill(1, D), offsets=zeros(D),
                                scheme=:block, tci_tolerance=1e-12,
                                tci_maxbonddim=200, cutoff=1e-26, tci_kwargs...)

Build `H = sum_d [-1/2 d^2/dx_d^2 + V_d(x_d)]`. Each `V_d` (default
`scales[d]*x^powers[d]`, or `potentials[d]` if given) is turned into a 1D
Fourier-space MPO by [`potential_mpo_dft`](@ref) and embedded exactly; the
kinetic terms are the exact diagonal Fourier MPOs. All `2D` terms are joined
by an exact direct sum. `qubits_per_dim` may also be a [`RegisterLayout`](@ref),
in which case `scheme` is ignored.
"""
function build_separable_hamiltonian(
    sites::AbstractVector{<:Index},
    qubits_per_dim,
    half_width_per_dim::AbstractVector{<:Real};
    potentials=nothing,
    powers::AbstractVector{<:Integer}=fill(2, length(half_width_per_dim)),
    scales::AbstractVector{<:Number}=fill(1, length(half_width_per_dim)),
    offsets::AbstractVector{<:Real}=zeros(length(half_width_per_dim)),
    scheme::Symbol=:block,
    tci_tolerance::Real=1e-12,
    tci_maxbonddim::Integer=200,
    cutoff::Real=1e-26,
    tci_kwargs...,
)
    layout = qubits_per_dim isa RegisterLayout ? qubits_per_dim :
             RegisterLayout(qubits_per_dim; scheme)
    D = length(layout.qubits_per_dim)
    (length(half_width_per_dim) == D && length(powers) == D &&
     length(scales) == D && length(offsets) == D) ||
        throw(DimensionMismatch(
            "qubits_per_dim, half_width_per_dim, powers, scales, and offsets must have equal length",
        ))
    isnothing(potentials) || length(potentials) == D ||
        throw(DimensionMismatch("one potential is required per dimension"))
    length(sites) == nsites(layout) ||
        throw(DimensionMismatch("sites must total sum(qubits_per_dim)"))

    kinetic_terms = Vector{MPO}(undef, D)
    potential_terms = Vector{MPO}(undef, D)
    tci_diagnostics = Vector{NamedTuple}(undef, D)

    hamiltonian = nothing
    for d in 1:D
        Nd, Ld = layout.qubits_per_dim[d], half_width_per_dim[d]
        Vd = isnothing(potentials) ? (x -> scales[d] * x^powers[d]) : potentials[d]
        local_sites = siteinds("Qubit", Nd)
        fit = potential_mpo_dft(
            local_sites, Vd, Nd, Ld;
            offset=offsets[d],
            tci_tolerance=tci_tolerance,
            tci_maxbonddim=tci_maxbonddim,
            cutoff=cutoff,
            tci_kwargs...,
        )
        potential_arrays = embed_mpo(itensor_mpo_to_arrays(fit.potential, local_sites), d, layout)
        kinetic_arrays = embed_mpo(kinetic_mpo(Nd, Ld), d, layout)

        kinetic_terms[d] = array_mpo_to_itensor(sites, kinetic_arrays)
        potential_terms[d] = array_mpo_to_itensor(sites, potential_arrays)
        tci_diagnostics[d] = fit.tci_diagnostics

        for term in (kinetic_terms[d], potential_terms[d])
            hamiltonian = isnothing(hamiltonian) ? term : +(hamiltonian, term; alg="directsum")
        end
    end

    return (;
        sites,
        layout,
        hamiltonian,
        kinetic_terms,
        potential_terms,
        tci_diagnostics,
        qubits_per_dim=layout.qubits_per_dim,
        half_width_per_dim,
        offsets,
    )
end
