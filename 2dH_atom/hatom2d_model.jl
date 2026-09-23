#!/usr/bin/env julia

"""
End-to-end two-dimensional soft-Coulomb atom in a Fourier QTT basis.

The model is H = -1/2(d²/dx²+d²/dy²) - Z/sqrt(r²+epsilon), with ℏ=m=e=1.
For every nonzero momentum transfer `(m,n)`, the potential uses the requested
large-box analytic coefficient

    C(m,n) = -Z * exp(-pi*sqrt(epsilon)*sqrt(m^2/Lx^2+n^2/Ly^2)) /
             (2*sqrt(m^2*Ly^2+n^2*Lx^2)).

TCI fits this target directly over the whole joint `Nx+Ny`-site register; no
real-space grid, FFT, or numerical quadrature is used. The zero mode is the
exact finite-rectangle average of the same soft-Coulomb potential. `Lx`, `Ly`,
and `epsilon` are direct user parameters; the code does not rescale epsilon.
"""

using ITensors, ITensorMPS
using LinearAlgebra
using Printf
using Random

include(joinpath(@__DIR__, "..", "Juliatools", "QTTFourier.jl"))
using .QTTFourier

# Real-space reconstruction densely contracts the whole Nx+Ny-site MPS
# (see itensor_mps_to_vec); that routinely exceeds ITensors' default
# order-14 warning threshold for N_list as small as [7,7] and is expected.
ITensors.disable_warn_order()

Base.@kwdef struct HAtom2DConfig
    N_list::Vector{Int} = [10, 10]
    half_width_list::Vector{Float64} = [1000.0, 1000.0]
    charge::Float64 = 1.0
    epsilon::Float64 = 1e-8
    tci_tolerance::Float64 = 1e-10
    tci_maxbonddim::Int = 400
    sweeps::Int = 20
    max_bond_dimension::Int = 10000000
    dmrg_cutoff::Float64 = 1e-8
    realspace_points_list::Vector{Int} = [121, 121]
    enable_realspace_reconstruction::Bool = false
end

"""
Exact finite-rectangle zero Fourier mode of
`-charge/sqrt(x^2+y^2+epsilon)` on `[-Lx,Lx] x [-Ly,Ly]`.
"""
function softened_coulomb_zero_mode(Lx::Real, Ly::Real, epsilon::Real;
                                    charge::Number=1)
    Lx > 0 && Ly > 0 || throw(ArgumentError("Lx and Ly must be positive"))
    epsilon >= 0 || throw(ArgumentError("epsilon must be nonnegative"))
    a = sqrt(epsilon)
    quadrant_integral = if iszero(epsilon)
        Lx * asinh(Ly / Lx) + Ly * asinh(Lx / Ly)
    else
        radius = sqrt(Lx^2 + Ly^2 + epsilon)
        Lx * asinh(Ly / sqrt(Lx^2 + epsilon)) +
        Ly * asinh(Lx / sqrt(Ly^2 + epsilon)) -
        a * atan(Lx * Ly, a * radius)
    end
    return complex(-charge * quadrant_integral / (Lx * Ly))
end

"""
Requested piecewise analytic Fourier coefficient of the softened 2D Coulomb
potential: the large-box closed form for nonzero modes and the exact
finite-rectangle average for `(0,0)`.
"""
function softened_coulomb2d_fourier_coeff(kx::Integer, ky::Integer,
                                           Lx::Real, Ly::Real, epsilon::Real;
                                           charge::Number=1)
    Lx > 0 && Ly > 0 || throw(ArgumentError("Lx and Ly must be positive"))
    epsilon >= 0 || throw(ArgumentError("epsilon must be nonnegative"))
    if kx == 0 && ky == 0
        return softened_coulomb_zero_mode(Lx, Ly, epsilon; charge=charge)
    end
    scaled_mode = hypot(kx / Lx, ky / Ly)
    denominator = 2 * hypot(kx * Ly, ky * Lx)
    damping = exp(-pi * sqrt(epsilon) * scaled_mode)
    return complex(-charge * damping / denominator)
end

function build_hatom2d(config::HAtom2DConfig)
    length(config.N_list) == 2 || throw(ArgumentError("N_list must have two entries"))
    length(config.half_width_list) == 2 ||
        throw(ArgumentError("half_width_list must have two entries"))
    config.epsilon >= 0 || throw(ArgumentError("epsilon must be nonnegative"))
    qubits_per_dim = config.N_list
    half_widths = config.half_width_list
    sites = siteinds("Qubit", sum(qubits_per_dim))

    kinetic_terms = [
        array_mpo_to_itensor(
            sites,
            embed_mpo(kinetic_mpo(qubits_per_dim[d], half_widths[d]), d, qubits_per_dim),
        )
        for d in 1:2
    ]

    Lx, Ly = half_widths
    coulomb_coeff(kx, ky) = softened_coulomb2d_fourier_coeff(
        kx, ky, Lx, Ly, config.epsilon; charge=config.charge,
    )
    potential_fit = coefficient_mpo_tci_nd(
        coulomb_coeff,
        qubits_per_dim;
        tolerance=config.tci_tolerance,
        maxbonddim=config.tci_maxbonddim,
        return_diagnostics=true,
    )
    potential = array_mpo_to_itensor(sites, potential_fit.tensors)

    hamiltonian = +(kinetic_terms[1], kinetic_terms[2]; alg="directsum")
    hamiltonian = +(hamiltonian, potential; alg="directsum")

    return (;
        sites,
        hamiltonian,
        kinetic_terms,
        potential,
        tci_diagnostics=potential_fit.diagnostics,
        qubits_per_dim,
        half_width_per_dim=half_widths,
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
)
    nsweeps >= 1 || throw(ArgumentError("nsweeps must be positive"))
    maxdim >= 1 || throw(ArgumentError("maxdim must be positive"))
    Random.seed!(seed)
    # Start from a genuinely compressed MPS. `maxdim` is only an upper bound
    # for bond growth during DMRG; using it to size the initial state can make
    # the first environment contraction scale as maxdim^2 and exhaust RAM.
    initial = random_mps(eltype, sites; linkdims=2)

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
    @printf(io, "2D soft-Coulomb atom: H = -1/2(d²/dx²+d²/dy²) - Z/sqrt(r²+epsilon)\n")
    @printf(io, "N_list                   = %s\n", string(config.N_list))
    @printf(io, "Fourier modes            = %d x %d\n", 2^Nx, 2^Ny)
    @printf(io, "domain                    = [%.1f, %.1f) x [%.1f, %.1f)\n", -Lx, Lx, -Ly, Ly)
    @printf(io, "charge Z                 = %.4f\n", config.charge)
    @printf(io, "softening epsilon        = %.6e\n", model.epsilon)
    @printf(io, "softening length sqrt(epsilon) = %.6e\n", model.softening_length)
    @printf(io, "potential coefficients   = requested closed form (nonzero mode), exact finite-box C00\n")
    for d in eachindex(model.qubits_per_dim)
        @printf(io, "kinetic MPO max bond dim (dim %d)   = %d\n", d, max_bond_dim(model.kinetic_terms[d]))
    end
    @printf(io, "potential MPO max bond dim (non-separable) = %d\n", max_bond_dim(model.potential))
    @printf(io, "Hamiltonian MPO max bond dim  = %d\n", max_bond_dim(model.hamiltonian))
    @printf(io, "ground-state MPS max bond dim = %d\n", max_bond_dim(state))
    @printf(io, "TCI requested tolerance       = %.6e\n", config.tci_tolerance)
    @printf(io, "TCI estimated relative error  = %.6e\n", model.tci_diagnostics.estimated_relative_error)
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
    # after the unregularized isolated limit. Softening, finite Fourier cutoff,
    # the large-box coefficient approximation, and the finite box all perturb
    # it, so this remains only a loose sign/order-of-magnitude sanity check.
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
    energy, state, dmrg_diagnostics = ground_state_with_dimension_log(
        model.hamiltonian,
        model.sites,
        joinpath(@__DIR__, "hatom2d_mps_dimensions.csv");
        nsweeps=config.sweeps,
        maxdim=config.max_bond_dimension,
        cutoff=config.dmrg_cutoff,
        seed=1,
        outputlevel=0,
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
        HAtom2DConfig(epsilon=parse(Float64, ARGS[1]))
    elseif length(ARGS) == 2
        HAtom2DConfig(realspace_points_list=parse.(Int, ARGS))
    else
        HAtom2DConfig(
            realspace_points_list=parse.(Int, ARGS[1:2]),
            epsilon=parse(Float64, ARGS[3]),
        )
    end
    main(config)
end
