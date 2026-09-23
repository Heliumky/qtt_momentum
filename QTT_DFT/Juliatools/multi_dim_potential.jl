"""Non-separable multi-dimensional potentials through the DFT.

A genuinely multi-dimensional potential such as `-1/|r|` does not factor, so
it is fitted by TCI over the whole joint real-space register (in either
[`RegisterLayout`](@ref) scheme) and rotated to Fourier space with the joint
DFT MPO: `V_k = U_nd * diag(V(x_j)) * U_nd†`.
"""

"""Real-space coordinates `(x_1, ..., x_D)` of a joint real-space-register bit string."""
function _joint_coordinates(sigma::AbstractVector{<:Integer}, layout::RegisterLayout,
                            half_width_per_dim, offsets)
    return ntuple(length(layout.qubits_per_dim)) do d
        grid_coordinate(view(sigma, layout.positions[d]), half_width_per_dim[d]; offset=offsets[d])
    end
end

"""Joint real-space-register bits of the grid point nearest to `point`."""
function _nearest_joint_sigma(point, layout::RegisterLayout, half_width_per_dim, offsets)
    sigma = ones(Int, nsites(layout))
    for d in eachindex(layout.qubits_per_dim)
        sigma[layout.positions[d]] = nearest_grid_sigma(
            point[d], layout.qubits_per_dim[d], half_width_per_dim[d]; offset=offsets[d],
        )
    end
    return sigma
end

"""
    realspace_function_mps_nd(f, layout, half_width_per_dim;
                              offsets=zeros(D), pivot_points=[], kwargs...)

Fit `f(x_1, ..., x_D)` on the joint grid as an MPS in the real-space register
of `layout` (a [`RegisterLayout`](@ref) or `qubits_per_dim`, block layout).
The grid points nearest to each coordinate tuple in `pivot_points` seed the
TCI (in addition to the corner `(-L_1, ..., -L_D)`); remaining keywords go to
[`fit_tensor_train`](@ref).
"""
function realspace_function_mps_nd(f, layout, half_width_per_dim::AbstractVector{<:Real};
                                   offsets::AbstractVector{<:Real}=zeros(length(half_width_per_dim)),
                                   pivot_points::AbstractVector=[],
                                   kwargs...)
    layout = layout isa RegisterLayout ? layout : RegisterLayout(layout)
    D = length(layout.qubits_per_dim)
    (length(half_width_per_dim) == D && length(offsets) == D) ||
        throw(DimensionMismatch("one half-width and one offset are required per dimension"))
    all(>(0), half_width_per_dim) || throw(ArgumentError("half-widths must be positive"))
    all(point -> length(point) == D, pivot_points) ||
        throw(DimensionMismatch("every pivot point needs one coordinate per dimension"))

    pivots = Vector{Int}[ones(Int, nsites(layout))]
    append!(pivots, [
        _nearest_joint_sigma(point, layout, half_width_per_dim, offsets) for point in pivot_points
    ])
    return fit_qtt_mps(
        sigma -> f(_joint_coordinates(sigma, layout, half_width_per_dim, offsets)...),
        nsites(layout);
        initialpivots=unique(pivots),
        kwargs...,
    )
end

"""
    potential_mpo_dft_nd(sites, V, layout, half_width_per_dim;
                         offsets=zeros(D), pivot_points=[],
                         tci_tolerance=1e-10, tci_maxbonddim=200,
                         cutoff=1e-26, maxdim=typemax(Int), tci_kwargs...)

Fourier-basis multiplication MPO of a real-space potential `V(x_1,...,x_D)`
on the joint register: TCI of `V` in real space, then
`V_k = U_nd * diag(V) * U_nd†` with [`fourier_transform_mpo_nd`](@ref).
Choose `offsets[d] = 0.5` to use cell-centred grids when `V` is singular at
the origin. Returns a named tuple with the ITensor MPO `potential`, the
real-space diagonal array MPO, the joint DFT array MPO, and TCI diagnostics.
"""
function potential_mpo_dft_nd(sites::AbstractVector{<:Index}, V, layout,
                              half_width_per_dim::AbstractVector{<:Real};
                              offsets::AbstractVector{<:Real}=zeros(length(half_width_per_dim)),
                              pivot_points::AbstractVector=[],
                              tci_tolerance::Real=1e-10,
                              tci_maxbonddim::Integer=200,
                              cutoff::Real=1e-26,
                              maxdim::Integer=typemax(Int),
                              tci_kwargs...)
    layout = layout isa RegisterLayout ? layout : RegisterLayout(layout)
    length(sites) == nsites(layout) ||
        throw(DimensionMismatch("sites must total sum(qubits_per_dim)"))
    fit = realspace_function_mps_nd(
        V, layout, half_width_per_dim;
        offsets=offsets,
        pivot_points=pivot_points,
        tolerance=tci_tolerance,
        maxbonddim=tci_maxbonddim,
        return_diagnostics=true,
        tci_kwargs...,
    )
    realspace_arrays = mps_to_diagonal_mpo(fit.tensors)
    transform_arrays = fourier_transform_mpo_nd(layout, half_width_per_dim; offsets)
    potential = fourier_space_operator(sites, realspace_arrays, transform_arrays; cutoff, maxdim)
    return (;
        potential,
        layout,
        realspace_arrays,
        transform_arrays,
        realspace_mps_tensors=fit.tensors,
        tci_diagnostics=fit.diagnostics,
    )
end

"""
    build_dft_hamiltonian_nd(sites, V, layout, half_width_per_dim; kwargs...)

`H = -1/2 * sum_d d^2/dx_d^2 + V(x_1, ..., x_D)` on the joint Fourier
register. The potential comes from [`potential_mpo_dft_nd`](@ref) (which
receives all keywords); the kinetic terms are exact and embedded per
dimension. All terms are combined by an exact direct sum.
"""
function build_dft_hamiltonian_nd(sites::AbstractVector{<:Index}, V, layout,
                                  half_width_per_dim::AbstractVector{<:Real}; kwargs...)
    layout = layout isa RegisterLayout ? layout : RegisterLayout(layout)
    fit = potential_mpo_dft_nd(sites, V, layout, half_width_per_dim; kwargs...)
    kinetic_terms = [
        array_mpo_to_itensor(
            sites,
            embed_mpo(kinetic_mpo(layout.qubits_per_dim[d], half_width_per_dim[d]), d, layout),
        )
        for d in eachindex(layout.qubits_per_dim)
    ]
    kinetic = reduce((a, b) -> +(a, b; alg="directsum"), kinetic_terms)
    hamiltonian = +(kinetic, fit.potential; alg="directsum")
    return (;
        sites,
        layout,
        hamiltonian,
        kinetic,
        kinetic_terms,
        potential=fit.potential,
        realspace_potential_arrays=fit.realspace_arrays,
        transform_arrays=fit.transform_arrays,
        tci_diagnostics=fit.tci_diagnostics,
        half_width_per_dim,
    )
end

"""
    realspace_state_mps_nd(f, sites, layout, half_width_per_dim;
                           offsets=zeros(D), pivot_points=[], tolerance=1e-8,
                           cutoff=1e-20)

Fourier-register MPS of a real-space wavefunction guess `f(x_1, ..., x_D)`:
TCI of `f` on the joint grid, then the joint DFT MPO, then normalization. A
physical guess (e.g. `exp(-r)` for a Coulomb problem) is a much better DMRG
starting point than a random MPS; see [`ground_state`](@ref). A 1D problem is
the case `layout = [N]`.
"""
function realspace_state_mps_nd(f, sites::AbstractVector{<:Index}, layout,
                                half_width_per_dim::AbstractVector{<:Real};
                                offsets::AbstractVector{<:Real}=zeros(length(half_width_per_dim)),
                                pivot_points::AbstractVector=[],
                                tolerance::Real=1e-8,
                                cutoff::Real=1e-20)
    layout = layout isa RegisterLayout ? layout : RegisterLayout(layout)
    length(sites) == nsites(layout) ||
        throw(DimensionMismatch("sites must total sum(qubits_per_dim)"))
    tensors = realspace_function_mps_nd(
        f, layout, half_width_per_dim; offsets, pivot_points, tolerance,
    )
    transform = array_mpo_to_itensor(
        sites, fourier_transform_mpo_nd(layout, half_width_per_dim; offsets),
    )
    state = apply(transform, array_mps_to_itensor(sites, tensors); cutoff=cutoff)
    return state / norm(state)
end

"""Antiderivative of `1/sqrt(x^2+y^2)` in both variables for `x, y >= 0`."""
_coulomb_antiderivative(x::Real, y::Real) =
    (iszero(x) || iszero(y)) ? zero(float(x + y)) : x * asinh(y / x) + y * asinh(x / y)

"""
    coulomb2d_cell_average(x, y, hx, hy; charge=1)

Exact average of `-charge/sqrt(x'^2+y'^2)` over the grid cell
`[x-hx/2, x+hx/2] × [y-hy/2, y+hy/2]` centred at `(x, y)`, from the
closed-form rectangle integral

```
∫∫ dx dy / r = G(x2,y2) - G(x1,y2) - G(x2,y1) + G(x1,y1),
G(x, y)      = x*asinh(y/x) + y*asinh(x/y).
```

The cell must not straddle a coordinate axis (true for every cell of a
cell-centred grid, `offset = 0.5`); by symmetry it is mapped to the first
quadrant. Sampling this instead of the point value `-charge/r` removes most of
the quadrature error of the DFT potential near the singularity: the cell
average is the exact projection of `-1/r` onto piecewise-constant functions,
while the point value at a cell centre misses the integrable divergence.
"""
function coulomb2d_cell_average(x::Real, y::Real, hx::Real, hy::Real; charge::Number=1)
    hx > 0 && hy > 0 || throw(ArgumentError("cell sizes must be positive"))
    x1, x2 = x - hx / 2, x + hx / 2
    y1, y2 = y - hy / 2, y + hy / 2
    tolerance = 1e-12 * max(hx, hy)
    if (x1 < -tolerance && x2 > tolerance) || (y1 < -tolerance && y2 > tolerance)
        throw(ArgumentError("the cell must not straddle a coordinate axis"))
    end
    xl, xh = minmax(abs(x1), abs(x2))
    yl, yh = minmax(abs(y1), abs(y2))
    xl < tolerance && (xl = zero(xl))
    yl < tolerance && (yl = zero(yl))
    G = _coulomb_antiderivative
    integral = G(xh, yh) - G(xl, yh) - G(xh, yl) + G(xl, yl)
    return -charge * integral / (hx * hy)
end
