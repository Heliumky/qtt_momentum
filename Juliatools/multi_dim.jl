"""Building blocks for separable multi-dimensional Hamiltonians.

Dimension `d` occupies `qubits_per_dim[d]` contiguous sites of the joint
register (dimension 1 first, site 1 within each block is that dimension's
LSB). Every cross-dimension operator used here is an exact tensor product,
so no truncation is introduced beyond what each 1D component already used.
"""

"""Identity MPO of `N` qubit sites (bond dimension 1)."""
function identity_mpo(N::Integer; eltype::Type=ComplexF64)
    N >= 1 || throw(ArgumentError("N must be positive"))
    tensor = zeros(eltype, 1, 2, 2, 1)
    tensor[1, 1, 1, 1] = 1
    tensor[1, 2, 2, 1] = 1
    return [copy(tensor) for _ in 1:N]
end

"""
    embed_mpo(component, dim, qubits_per_dim; eltype=ComplexF64)

Embed a single-dimension array MPO `component` into the joint
`sum(qubits_per_dim)`-site system as `I ⊗ ... ⊗ component ⊗ ... ⊗ I`, with
`component` occupying the contiguous site block belonging to dimension `dim`.
"""
function embed_mpo(component::AbstractVector{<:AbstractArray{<:Number,4}},
                   dim::Integer, qubits_per_dim::AbstractVector{<:Integer};
                   eltype::Type=ComplexF64)
    1 <= dim <= length(qubits_per_dim) || throw(ArgumentError("dim is out of range"))
    length(component) == qubits_per_dim[dim] ||
        throw(DimensionMismatch("component must have qubits_per_dim[dim] sites"))

    blocks = [
        d == dim ? Array{eltype,4}.(component) : identity_mpo(qubits_per_dim[d]; eltype)
        for d in eachindex(qubits_per_dim)
    ]
    return reduce(vcat, blocks)
end

"""
    build_separable_hamiltonian(sites, qubits_per_dim, half_width_per_dim;
                                powers=fill(2, D), scales=fill(1, D),
                                ordering=:lsb_first,
                                tci_tolerance=1e-10, tci_maxbonddim=200)

Build `H = sum_d [-1/2 d^2/dx_d^2 + scale_d*x_d^powers_d]` on a joint QTT
register where dimension `d` occupies `qubits_per_dim[d]` contiguous sites
with periodic half-width `half_width_per_dim[d]`. Each dimension's potential
is fitted independently by TCI from its analytic Fourier coefficients; each
dimension's kinetic term is the exact Fourier-space MPO. All `2*D` terms are
combined by an exact ITensor direct sum, so the Hamiltonian bond dimension is
at most the sum of the `2*D` component bond dimensions.
"""
function build_separable_hamiltonian(
    sites::AbstractVector{<:Index},
    qubits_per_dim::AbstractVector{<:Integer},
    half_width_per_dim::AbstractVector{<:Real};
    powers::AbstractVector{<:Integer}=fill(2, length(qubits_per_dim)),
    scales::AbstractVector{<:Number}=fill(1, length(qubits_per_dim)),
    ordering::Symbol=:lsb_first,
    tci_tolerance::Real=1e-10,
    tci_maxbonddim::Integer=200,
)
    D = length(qubits_per_dim)
    D >= 1 || throw(ArgumentError("qubits_per_dim cannot be empty"))
    (length(half_width_per_dim) == D && length(powers) == D && length(scales) == D) ||
        throw(DimensionMismatch(
            "qubits_per_dim, half_width_per_dim, powers, and scales must have equal length",
        ))
    length(sites) == sum(qubits_per_dim) ||
        throw(DimensionMismatch("sites must total sum(qubits_per_dim)"))

    kinetic_terms = Vector{MPO}(undef, D)
    potential_terms = Vector{MPO}(undef, D)
    tci_diagnostics = Vector{NamedTuple}(undef, D)

    hamiltonian = nothing
    for d in 1:D
        Nd, Ld = qubits_per_dim[d], half_width_per_dim[d]
        kinetic_arrays = embed_mpo(kinetic_mpo(Nd, Ld), d, qubits_per_dim)
        potential_fit = power_mpo_tci(
            powers[d], Nd, Ld;
            scale=scales[d],
            ordering=ordering,
            tolerance=tci_tolerance,
            maxbonddim=tci_maxbonddim,
            return_diagnostics=true,
        )
        potential_arrays = embed_mpo(potential_fit.tensors, d, qubits_per_dim)

        kinetic_terms[d] = array_mpo_to_itensor(sites, kinetic_arrays)
        potential_terms[d] = array_mpo_to_itensor(sites, potential_arrays)
        tci_diagnostics[d] = potential_fit.diagnostics

        for term in (kinetic_terms[d], potential_terms[d])
            hamiltonian = isnothing(hamiltonian) ? term : +(hamiltonian, term; alg="directsum")
        end
    end

    return (;
        sites,
        hamiltonian,
        kinetic_terms,
        potential_terms,
        tci_diagnostics,
        qubits_per_dim,
        half_width_per_dim,
    )
end
