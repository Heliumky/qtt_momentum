#!/usr/bin/env julia

"""
End-to-end two-dimensional (soft-)Coulomb atom in a Fourier QTT basis, with
the potential built through the discrete Fourier transform.

The model is H = -1/2(d²/dx²+d²/dy²) - Z/sqrt(r²+epsilon), with ℏ=m=e=1.
The potential is sampled on the cell-centred real-space grid
`x_j = -L + (j+1/2)*2L/M`, which never contains the origin, so the bare
Coulomb potential (`epsilon = 0`, the default) can be used without softening.
TCI fits V(x,y) over the joint `Nx+Ny`-site real-space register, seeded with a
pivot at the origin, and the Fourier-space MPO is V_k = U diag(V) U† with the
joint quantics DFT MPO U. The kinetic terms are exact Fourier-space MPOs.

Unlike the analytic finite-box coefficients of the original package, the
DFT matrix elements are a grid quadrature, so the grid must resolve the
ground state (Bohr radius 1/2 for Z=1 in 2D); E0 approaches the isolated
2D-hydrogen value -2 as the grid spacing 2L/2^N -> 0 and L -> infinity.
"""

using ITensors, ITensorMPS
using LinearAlgebra
using Printf
using Random

include(joinpath(@__DIR__, "..", "Juliatools", "QTTDFT.jl"))
using .QTTDFT

# Real-space reconstruction densely contracts the whole Nx+Ny-site MPS
# (see itensor_mps_to_vec); that routinely exceeds ITensors' default
# order-14 warning threshold for N_list as small as [7,7] and is expected.
ITensors.disable_warn_order()

Base.@kwdef struct HAtom2DConfig
    N_list::Vector{Int} = [8, 8]
    half_width_list::Vector{Float64} = [16.0, 16.0]
    charge::Float64 = 1.0
    epsilon::Float64 = 0.0
    grid_offset_list::Vector{Float64} = [0.5, 0.5]
    # :cell_average samples the exact cell average of -Z/r (bare Coulomb only);
    # :point samples -Z/sqrt(r^2+epsilon) at the cell centres.
    potential_sampling::Symbol = :cell_average
    # :exponential starts DMRG from exp(-2Z r) built in real space and moved to
    # Fourier space with the DFT MPO; :random uses a random MPS, which gets
    # stuck in a local minimum for N_list = [10, 10] (E = -0.673).
    initial_guess::Symbol = :exponential
    tci_tolerance::Float64 = 1e-10
    tci_maxbonddim::Int = 400
    dft_cutoff::Float64 = 1e-26
    sweeps::Int = 20
    max_bond_dimension::Int = 10000000
    dmrg_cutoff::Float64 = 1e-8
    realspace_points_list::Vector{Int} = [121, 121]
    enable_realspace_reconstruction::Bool = true
end

function build_hatom2d(config::HAtom2DConfig)
    length(config.N_list) == 2 || throw(ArgumentError("N_list must have two entries"))
    length(config.half_width_list) == 2 ||
        throw(ArgumentError("half_width_list must have two entries"))
    length(config.grid_offset_list) == 2 ||
        throw(ArgumentError("grid_offset_list must have two entries"))
    config.epsilon >= 0 || throw(ArgumentError("epsilon must be nonnegative"))
    qubits_per_dim = config.N_list
    half_widths = config.half_width_list
    sites = siteinds("Qubit", sum(qubits_per_dim))

    config.potential_sampling in (:cell_average, :point) ||
        throw(ArgumentError("potential_sampling must be :cell_average or :point"))
    config.potential_sampling === :cell_average && !iszero(config.epsilon) &&
        throw(ArgumentError("cell averaging is implemented for the bare Coulomb potential (epsilon = 0)"))
    config.potential_sampling === :cell_average && config.grid_offset_list != [0.5, 0.5] &&
        throw(ArgumentError("cell averaging needs cell-centred grids (grid_offset_list = [0.5, 0.5])"))

    charge, epsilon = config.charge, config.epsilon
    hx, hy = 2half_widths[1] / 2^qubits_per_dim[1], 2half_widths[2] / 2^qubits_per_dim[2]
    coulomb = if config.potential_sampling === :cell_average
        (x, y) -> coulomb2d_cell_average(x, y, hx, hy; charge=charge)
    else
        (x, y) -> -charge / sqrt(x^2 + y^2 + epsilon)
    end
    model = build_dft_hamiltonian_nd(
        sites,
        coulomb,
        qubits_per_dim,
        half_widths;
        offsets=config.grid_offset_list,
        pivot_points=[(0.0, 0.0)],
        tci_tolerance=config.tci_tolerance,
        tci_maxbonddim=config.tci_maxbonddim,
        cutoff=config.dft_cutoff,
    )

    return (;
        model...,
        qubits_per_dim,
        epsilon=config.epsilon,
        softening_length=sqrt(config.epsilon),
    )
end

mutable struct SweepDimensionObserver{I<:IO} <: AbstractObserver
    diagnostics::DMRGObserver
    io::I
    bond_dimensions_by_sweep::Vector{Vector{Int}}
end

function ITensorMPS.measure!(observer::SweepDimensionObserver;
                             sweep, psi, sweep_is_done=false, kwargs...)
    ITensorMPS.measure!(
        observer.diagnostics;
        sweep=sweep,
        psi=psi,
        sweep_is_done=sweep_is_done,
        kwargs...,
    )

    if sweep_is_done
        dimensions = Int.(mps_bond_dims(psi))
        push!(observer.bond_dimensions_by_sweep, dimensions)
        print(observer.io, sweep, ',', maximum(dimensions; init=1))
        for dimension in dimensions
            print(observer.io, ',', dimension)
        end
        println(observer.io)
        flush(observer.io)
    end
    return nothing
end

function ITensorMPS.checkdone!(observer::SweepDimensionObserver; kwargs...)
    return ITensorMPS.checkdone!(observer.diagnostics; kwargs...)
end

function ground_state_with_dimension_log(
    H::MPO,
    sites::AbstractVector{<:Index},
    dimension_log_path::AbstractString;
    nsweeps::Integer=15,
    maxdim::Integer=64,
    cutoff::Real=1e-12,
    seed::Integer=1,
    eltype::Type=ComplexF64,
    outputlevel::Integer=1,
    initial_state::Union{Nothing,MPS}=nothing,
)
    nsweeps >= 1 || throw(ArgumentError("nsweeps must be positive"))
    maxdim >= 1 || throw(ArgumentError("maxdim must be positive"))
    Random.seed!(seed)
    # Start from a genuinely compressed MPS. `maxdim` is only an upper bound
    # for bond growth during DMRG; using it to size the initial state can make
    # the first environment contraction scale as maxdim^2 and exhaust RAM.
    initial = isnothing(initial_state) ? random_mps(eltype, sites; linkdims=2) : initial_state

    return open(dimension_log_path, "w") do io
        print(io, "sweep,max_bond_dimension")
        for bond in 1:(length(sites) - 1)
            print(io, ",bond_$(bond)_$(bond + 1)")
        end
        println(io)
        flush(io)

        diagnostics_observer = DMRGObserver()
        observer = SweepDimensionObserver(diagnostics_observer, io, Vector{Int}[])
        energy, state = dmrg(
            H,
            initial;
            nsweeps=nsweeps,
            maxdim=maxdim,
            cutoff=cutoff,
            outputlevel=outputlevel,
            observer=observer,
        )
        errors = collect(truncerrors(diagnostics_observer))
        diagnostics = (
            requested_cutoff=Float64(cutoff),
            truncation_error_by_sweep=errors,
            max_truncation_error=maximum(errors; init=0.0),
            bond_dimensions_by_sweep=observer.bond_dimensions_by_sweep,
        )
        return energy, state, diagnostics
    end
end

function fix_phase(values::AbstractMatrix{<:Number})
    # DMRG returns the ground state up to an arbitrary global phase; fix it
    # so the peak amplitude (the s-like ground state's maximum, at/near the
    # origin) is real and positive.
    peak_index = argmax(abs.(values))
    return values .* exp(-im * angle(values[peak_index]))
end

function reconstruct_state(state::MPS, sites, config::HAtom2DConfig)
    reconstructed = mps_to_realspace_nd(
        state,
        sites,
        config.N_list,
        config.half_width_list;
        normalize_coefficients=true,
        points_per_dim=config.realspace_points_list,
    )
    x, y = reconstructed.grids
    numerical = fix_phase(reconstructed.values)
    density = abs2.(numerical)

    return (; x=x, y=y, numerical=numerical, density=density)
end

"""
Diagnostics that only need to be *quantitatively* accurate (not plotted),
computed on the state's own native `2^N_list` grid: near the bare-Coulomb
limit the ground state develops a cusp at the origin, and the simple
Riemann-sum quadrature used here is only exact for a band-limited periodic
function at (a multiple of) its native sample rate. This avoids tying
quadrature accuracy to whatever grid `realspace_points_list` uses for plots.
"""
function reconstruction_diagnostics(state::MPS, sites, config::HAtom2DConfig)
    native_points = 2 .^ config.N_list
    reconstructed = mps_to_realspace_nd(
        state, sites, config.N_list, config.half_width_list;
        normalize_coefficients=true, points_per_dim=native_points,
        offsets=config.grid_offset_list,
    )
    numerical = fix_phase(reconstructed.values)
    dx = 2config.half_width_list[1] / native_points[1]
    dy = 2config.half_width_list[2] / native_points[2]

    # The Hamiltonian is exactly symmetric under x<->y whenever both
    # dimensions share the same qubit count and half-width; the true
    # Coulomb ground state is then rotationally symmetric and in particular
    # invariant under that swap, so this is an honest sanity check on the
    # whole construction, independent of box size.
    symmetric_grid = config.N_list[1] == config.N_list[2] &&
                      config.half_width_list[1] == config.half_width_list[2]
    transpose_asymmetry = symmetric_grid ?
                           maximum(abs, numerical .- transpose(numerical)) : NaN

    return (
        norm_check=sum(abs2, numerical) * dx * dy,
        max_imaginary=maximum(abs, imag.(numerical)),
        transpose_asymmetry=transpose_asymmetry,
        symmetric_grid=symmetric_grid,
    )
end

function write_tensor_shapes(path::AbstractString, state::MPS, sites, model)
    open(path, "w") do io
        println(io, "MPS shape convention: (left_bond, physical_dim, right_bond)")
        for (site, shape) in enumerate(mps_shapes(state, sites))
            println(io, "A_$site = $shape")
        end
        println(io)
        println(io, "MPO shape convention: (left_bond, output_dim, input_dim, right_bond)")
        for d in eachindex(model.qubits_per_dim)
            println(io, "Kinetic MPO (dimension $d):")
            for (site, shape) in enumerate(mpo_shapes(model.kinetic_terms[d], sites))
                println(io, "  T$(d)_$site = $shape")
            end
        end
        println(io, "Potential MPO (non-separable, fit over the joint register):")
        for (site, shape) in enumerate(mpo_shapes(model.potential, sites))
            println(io, "  V_$site = $shape")
        end
        println(io, "Hamiltonian MPO:")
        for (site, shape) in enumerate(mpo_shapes(model.hamiltonian, sites))
            println(io, "  W_$site = $shape")
        end
    end
end

function write_csv(path::AbstractString, reconstruction)
    open(path, "w") do io
        println(io, "x,y,numerical_real,numerical_imag,density")
        for j in eachindex(reconstruction.y), i in eachindex(reconstruction.x)
            @printf(
                io,
                "%.16e,%.16e,%.16e,%.16e,%.16e\n",
                reconstruction.x[i],
                reconstruction.y[j],
                real(reconstruction.numerical[i, j]),
                imag(reconstruction.numerical[i, j]),
                reconstruction.density[i, j],
            )
        end
    end
end

function print_report(io::IO, config, model, energy, state, dmrg_diagnostics,
                      kinetic_energies, potential_energy, reconstruction, diagnostics, validation)
    Nx, Ny = config.N_list
    Lx, Ly = config.half_width_list
    @printf(io, "2D (soft-)Coulomb atom: H = -1/2(d²/dx²+d²/dy²) - Z/sqrt(r²+epsilon)  (DFT potential)\n")
    @printf(io, "N_list                   = %s\n", string(config.N_list))
    @printf(io, "Fourier modes            = %d x %d\n", 2^Nx, 2^Ny)
    @printf(io, "domain                    = [%.1f, %.1f) x [%.1f, %.1f)\n", -Lx, Lx, -Ly, Ly)
    @printf(io, "charge Z                 = %.4f\n", config.charge)
    @printf(io, "softening epsilon        = %.6e\n", model.epsilon)
    @printf(io, "softening length sqrt(epsilon) = %.6e\n", model.softening_length)
    @printf(io, "potential                = grid samples, V_k = U diag(V) U† (quantics DFT MPO)\n")
    @printf(io, "potential sampling       = %s\n", string(config.potential_sampling))
    @printf(io, "DMRG initial guess       = %s\n", string(config.initial_guess))
    @printf(io, "grid offsets (cells)     = %s\n", string(config.grid_offset_list))
    @printf(io, "grid spacing             = %.6e x %.6e\n", 2Lx / 2^Nx, 2Ly / 2^Ny)
    @printf(io, "real-space V MPO max bond dim = %d\n", max_bond_dim(model.realspace_potential_arrays))
    @printf(io, "joint DFT MPO max bond dim    = %d\n", max_bond_dim(model.transform_arrays))
    @printf(io, "U V U† compression cutoff     = %.1e\n", config.dft_cutoff)
    for d in eachindex(model.qubits_per_dim)
        @printf(io, "kinetic MPO max bond dim (dim %d)   = %d\n", d, max_bond_dim(model.kinetic_terms[d]))
    end
    @printf(io, "potential MPO max bond dim (non-separable) = %d\n", max_bond_dim(model.potential))
    @printf(io, "Hamiltonian MPO max bond dim  = %d\n", max_bond_dim(model.hamiltonian))
    @printf(io, "ground-state MPS max bond dim = %d\n", max_bond_dim(state))
    @printf(io, "TCI requested tolerance       = %.6e\n", config.tci_tolerance)
    @printf(io, "TCI estimated relative error  = %.6e\n", model.tci_diagnostics.estimated_relative_error)
    @printf(io, "TCI global error estimate     = %.6e\n", model.tci_diagnostics.global_error_estimate)
    @printf(io, "TCI target evaluations        = %d\n", model.tci_diagnostics.target_evaluations)
    @printf(io, "DMRG requested cutoff         = %.6e\n", config.dmrg_cutoff)
    @printf(io, "DMRG max truncation error     = %.6e\n", dmrg_diagnostics.max_truncation_error)
    @printf(io, "DMRG truncation error/sweep   = %s\n", repr(dmrg_diagnostics.truncation_error_by_sweep))
    if isnothing(reconstruction)
        println(io, "real-space reconstruction = disabled")
        println(io, "real-space plotting       = disabled")
    else
        @printf(io, "real-space plot points   = %d x %d\n", length(reconstruction.x), length(reconstruction.y))
    end
    println(io, "ground-state MPS tensor shapes:")
    for (site, shape) in enumerate(mps_shapes(state, model.sites))
        println(io, "  A_$site = $shape")
    end
    println(io, "Hamiltonian MPO tensor shapes:")
    for (site, shape) in enumerate(mpo_shapes(model.hamiltonian, model.sites))
        println(io, "  W_$site = $shape")
    end
    @printf(io, "\n")
    @printf(io, "DMRG E0                  = %.14f\n", energy)
    @printf(io, "<Tx>                      = %.14f\n", kinetic_energies[1])
    @printf(io, "<Ty>                      = %.14f\n", kinetic_energies[2])
    @printf(io, "<V>                       = %.14f\n", potential_energy)
    @printf(io, "<T> + <V>                = %.14f\n", sum(kinetic_energies) + potential_energy)
    @printf(io, "energy self-consistency error = %.3e\n", abs(energy - (sum(kinetic_energies) + potential_energy)))
    @printf(io, "virial ratio -<V>/<T>    = %.6f  (isolated unregularized limit: exactly 2)\n",
            -potential_energy / sum(kinetic_energies))
    @printf(io, "\n")
    @printf(io, "isolated unregularized 2D hydrogen reference: E_0 = -2\n")
    @printf(io, "the reference is approached only after a joint convergence check:\n")
    @printf(io, "epsilon -> 0, Fourier cutoff -> infinity, and half_width -> infinity\n")
    @printf(io, "\n")
    if isnothing(diagnostics)
        println(io, "real-space diagnostics   = disabled")
    else
        @printf(io, "density normalization (native grid) = %.14f  (exact 1)\n", diagnostics.norm_check)
        @printf(io, "max imaginary component (native grid) = %.3e\n", diagnostics.max_imaginary)
        if diagnostics.symmetric_grid
            @printf(io, "x<->y transpose asymmetry (native grid) = %.3e\n", diagnostics.transpose_asymmetry)
        else
            @printf(io, "x<->y transpose asymmetry = skipped (asymmetric N_list/half_width_list)\n")
        end
    end
    @printf(io, "energy self-consistency tolerance = %.3e\n", validation.energy_tolerance)
    if !isnothing(diagnostics)
        @printf(io, "density/symmetry tolerance         = %.3e\n", validation.wavefunction_tolerance)
    end
    @printf(io, "validation                = %s\n", validation.passed ? "PASS" : "FAIL")
    for failure in validation.failures
        println(io, "  failed: $failure")
    end
end

function validate_result(config, model, dmrg_diagnostics, energy,
                         kinetic_energies, potential_energy, diagnostics)
    algorithm_error = max(
        config.tci_tolerance,
        model.tci_diagnostics.estimated_relative_error,
    ) + max(
        config.dmrg_cutoff,
        dmrg_diagnostics.max_truncation_error,
    )
    energy_tolerance = max(1e-8, 20algorithm_error)
    wavefunction_tolerance = max(1e-6, 10sqrt(algorithm_error))

    # The pure-Coulomb virial identity 2<T>=-<V> (ratio exactly 2) only holds
    # after the unregularized isolated limit. Softening, finite grid spacing,
    # and the finite box all perturb it, so this remains only a loose
    # sign/order-of-magnitude sanity check.
    energy_checks = (
        energy_self_consistency=abs(energy - (sum(kinetic_energies) + potential_energy)) < energy_tolerance,
        bound_state=energy < 0,
        virial_ratio_sane=0.0 < -potential_energy / sum(kinetic_energies) < 10.0,
    )
    checks = if isnothing(diagnostics)
        energy_checks
    else
        (;
            energy_checks...,
            density_normalization=abs(diagnostics.norm_check - 1) < wavefunction_tolerance,
            max_imaginary=diagnostics.max_imaginary < wavefunction_tolerance,
            transpose_symmetry=!diagnostics.symmetric_grid ||
                                diagnostics.transpose_asymmetry < wavefunction_tolerance,
        )
    end
    failures = String[string(name) for (name, passed) in pairs(checks) if !passed]
    return (;
        passed=isempty(failures),
        failures,
        checks,
        algorithm_error,
        energy_tolerance,
        wavefunction_tolerance,
    )
end

function main(config::HAtom2DConfig=HAtom2DConfig())
    model = build_hatom2d(config)
    config.initial_guess in (:exponential, :random) ||
        throw(ArgumentError("initial_guess must be :exponential or :random"))
    initial_state = config.initial_guess === :random ? nothing : realspace_state_mps_nd(
        (x, y) -> exp(-2config.charge * hypot(x, y)),
        model.sites, config.N_list, config.half_width_list;
        offsets=config.grid_offset_list, pivot_points=[(0.0, 0.0)],
    )
    energy, state, dmrg_diagnostics = ground_state_with_dimension_log(
        model.hamiltonian,
        model.sites,
        joinpath(@__DIR__, "hatom2d_mps_dimensions.csv");
        nsweeps=config.sweeps,
        maxdim=config.max_bond_dimension,
        cutoff=config.dmrg_cutoff,
        seed=1,
        outputlevel=0,
        initial_state=initial_state,
    )

    kinetic_energies = [expect_mpo(state, term) for term in model.kinetic_terms]
    potential_energy = expect_mpo(state, model.potential)
    reconstruction = config.enable_realspace_reconstruction ?
                     reconstruct_state(state, model.sites, config) : nothing
    diagnostics = config.enable_realspace_reconstruction ?
                  reconstruction_diagnostics(state, model.sites, config) : nothing
    validation = validate_result(
        config,
        model,
        dmrg_diagnostics,
        energy,
        kinetic_energies,
        potential_energy,
        diagnostics,
    )

    print_report(
        stdout,
        config,
        model,
        energy,
        state,
        dmrg_diagnostics,
        kinetic_energies,
        potential_energy,
        reconstruction,
        diagnostics,
        validation,
    )

    report_path = joinpath(@__DIR__, "hatom2d_results.txt")
    open(report_path, "w") do io
        print_report(
            io,
            config,
            model,
            energy,
            state,
            dmrg_diagnostics,
            kinetic_energies,
            potential_energy,
            reconstruction,
            diagnostics,
            validation,
        )
    end

    write_tensor_shapes(
        joinpath(@__DIR__, "hatom2d_tensor_shapes.txt"),
        state,
        model.sites,
        model,
    )
    if config.enable_realspace_reconstruction
        write_csv(joinpath(@__DIR__, "hatom2d_ground_state.csv"), reconstruction)
        plot_heatmap(
            reconstruction.x,
            reconstruction.y,
            real.(reconstruction.numerical);
            filename=joinpath(@__DIR__, "hatom2d_ground_state.svg"),
            title="2D hydrogen-atom ground state from Fourier-QTT DMRG",
            colorbar_label="Re ψ",
        )
        plot_heatmap(
            reconstruction.x,
            reconstruction.y,
            reconstruction.density;
            filename=joinpath(@__DIR__, "hatom2d_density.svg"),
            title="2D hydrogen-atom ground-state density |ψ|²",
            colorbar_label="|ψ|²",
        )
    end

    validation.passed || error(
        "2D hydrogen atom validation failed ($(join(validation.failures, ", "))); " *
        "full diagnostics were saved to $report_path",
    )

    return (;
        model,
        energy,
        state,
        dmrg_diagnostics,
        kinetic_energies,
        potential_energy,
        reconstruction,
        diagnostics,
        validation,
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) in (0, 1, 2, 3) || error(
        "usage: julia hatom2d_model.jl [epsilon] | " *
        "[points_x points_y [epsilon]]",
    )
    config = if isempty(ARGS)
        HAtom2DConfig()
    elseif length(ARGS) == 1
        HAtom2DConfig(epsilon=parse(Float64, ARGS[1]), potential_sampling=:point)
    elseif length(ARGS) == 2
        HAtom2DConfig(realspace_points_list=parse.(Int, ARGS))
    else
        HAtom2DConfig(
            realspace_points_list=parse.(Int, ARGS[1:2]),
            epsilon=parse(Float64, ARGS[3]),
            potential_sampling=:point,
        )
    end
    main(config)
end
