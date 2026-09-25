#!/usr/bin/env julia

"""
End-to-end one-dimensional harmonic-oscillator validation in a Fourier QTT basis,
with the potential built through the discrete Fourier transform.

The model is H = -1/2 d²/dx² + x²/2 with ℏ = m = ω = 1. V(x_j) is fitted by TCI
on the real-space grid (a quadratic has quantics rank 3), rotated to the
Fourier basis with the quantics DFT MPO as V_k = U diag(V) U†, the derivative
is an exact Fourier-space MPO, and the ground state is obtained with ITensor
DMRG.
"""

using ITensors, ITensorMPS
using LinearAlgebra
using Printf
using Random

include(joinpath(@__DIR__, "..", "Juliatools", "QTTDFT.jl"))
using .QTTDFT

Base.@kwdef struct SHOConfig
    qubits::Int = 10
    half_width::Float64 = 8.0
    grid_offset::Float64 = 0.0
    tci_tolerance::Float64 = 1e-12
    dft_cutoff::Float64 = 1e-26
    sweeps::Int = 20
    max_bond_dimension::Int = 10000000
    dmrg_cutoff::Float64 = 1e-8
    realspace_points::Int = 1601
end

function build_sho(config::SHOConfig)
    sites = siteinds("Qubit", config.qubits)

    # V(x_j) = x_j²/2 is fitted in real space, then V_k = U diag(V) U†.
    model = build_power_hamiltonian(
        sites,
        2,
        config.qubits,
        config.half_width;
        scale=0.5,
        offset=config.grid_offset,
        pivot_points=[0.0],
        tci_tolerance=config.tci_tolerance,
        cutoff=config.dft_cutoff,
    )
    return (; sites, model...)
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

function align_and_compare(state::MPS, sites, half_width::Real, realspace_points::Integer)
    reconstructed = mps_to_realspace(
        state,
        sites,
        half_width;
        normalize_coefficients=true,
        npoints=realspace_points,
    )
    x = reconstructed.x
    numerical = reconstructed.values
    dx = 2half_width / length(x)

    exact = ComplexF64.(pi^(-1 / 4) .* exp.(-x .^ 2 ./ 2))
    exact ./= sqrt(sum(abs2, exact) * dx)

    overlap = sum(conj.(exact) .* numerical) * dx
    numerical .*= exp(-im * angle(overlap))
    error = numerical - exact

    return (
        x=x,
        numerical=numerical,
        exact=exact,
        error=error,
        l2_error=sqrt(sum(abs2, error) * dx),
        max_error=maximum(abs, error),
        max_imaginary=maximum(abs, imag.(numerical)),
        overlap=abs(sum(conj.(exact) .* numerical) * dx),
        reconstruction_method=reconstructed.method,
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
        println(io, "Potential MPO:")
        for (site, shape) in enumerate(mpo_shapes(model.potential, sites))
            println(io, "V_$site = $shape")
        end
        println(io, "Kinetic MPO:")
        for (site, shape) in enumerate(mpo_shapes(model.kinetic, sites))
            println(io, "T_$site = $shape")
        end
        println(io, "Hamiltonian MPO:")
        for (site, shape) in enumerate(mpo_shapes(model.hamiltonian, sites))
            println(io, "W_$site = $shape")
        end
    end
end

function write_csv(path::AbstractString, comparison)
    open(path, "w") do io
        println(io, "x,numerical_real,numerical_imag,exact,error_abs")
        for index in eachindex(comparison.x)
            @printf(
                io,
                "%.16e,%.16e,%.16e,%.16e,%.16e\n",
                comparison.x[index],
                real(comparison.numerical[index]),
                imag(comparison.numerical[index]),
                real(comparison.exact[index]),
                abs(comparison.error[index]),
            )
        end
    end
end

function print_report(io::IO, config, model, energy, state, dmrg_diagnostics,
                      kinetic_energy, potential_energy, comparison, validation)
    @printf(io, "1D harmonic oscillator: H = -1/2 d²/dx² + x²/2  (DFT potential)\n")
    @printf(io, "qubits                  = %d\n", config.qubits)
    @printf(io, "Fourier modes            = %d\n", 2^config.qubits)
    @printf(io, "domain                   = [%.1f, %.1f)\n", -config.half_width, config.half_width)
    @printf(io, "grid offset (cells)      = %.3f\n", config.grid_offset)
    @printf(io, "real-space V MPO max bond dim = %d\n", max_bond_dim(model.realspace_potential_arrays))
    @printf(io, "DFT MPO max bond dim          = %d\n", max_bond_dim(model.transform_arrays))
    @printf(io, "U V U† compression cutoff     = %.1e\n", config.dft_cutoff)
    @printf(io, "potential MPO max bond dim    = %d\n", max_bond_dim(model.potential))
    @printf(io, "kinetic MPO max bond dim      = %d\n", max_bond_dim(model.kinetic))
    @printf(io, "Hamiltonian MPO max bond dim  = %d\n", max_bond_dim(model.hamiltonian))
    @printf(io, "ground-state MPS max bond dim = %d\n", max_bond_dim(state))
    @printf(io, "TCI requested tolerance       = %.6e\n", config.tci_tolerance)
    @printf(io, "TCI estimated relative error  = %.6e\n",
            model.tci_diagnostics.estimated_relative_error)
    @printf(io, "TCI global error estimate     = %.6e\n",
            model.tci_diagnostics.global_error_estimate)
    @printf(io, "TCI target evaluations        = %d\n",
            model.tci_diagnostics.target_evaluations)
    @printf(io, "DMRG requested cutoff         = %.6e\n", config.dmrg_cutoff)
    @printf(io, "DMRG max truncation error     = %.6e\n",
            dmrg_diagnostics.max_truncation_error)
    @printf(io, "DMRG truncation error/sweep   = %s\n",
            repr(dmrg_diagnostics.truncation_error_by_sweep))
    @printf(io, "real-space plot points   = %d\n", length(comparison.x))
    @printf(io, "real-space grid spacing  = %.8e\n",
            2config.half_width / length(comparison.x))
    @printf(io, "reconstruction method    = %s\n", string(comparison.reconstruction_method))
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
    @printf(io, "exact E0                 = %.14f\n", 0.5)
    @printf(io, "energy error             = %+.3e\n", energy - 0.5)
    @printf(io, "<T>                       = %.14f  (exact 0.25)\n", kinetic_energy)
    @printf(io, "<V>                       = %.14f  (exact 0.25)\n", potential_energy)
    @printf(io, "<T> + <V>                = %.14f\n", kinetic_energy + potential_energy)
    @printf(io, "\n")
    @printf(io, "wavefunction overlap     = %.14f\n", comparison.overlap)
    @printf(io, "wavefunction L2 error    = %.3e\n", comparison.l2_error)
    @printf(io, "wavefunction max error   = %.3e\n", comparison.max_error)
    @printf(io, "max imaginary component = %.3e\n", comparison.max_imaginary)
    @printf(io, "energy tolerance          = %.3e\n", validation.energy_tolerance)
    @printf(io, "component-energy tolerance = %.3e\n", validation.component_tolerance)
    @printf(io, "wavefunction L2 tolerance = %.3e\n", validation.wavefunction_tolerance)
    @printf(io, "validation                = %s\n", validation.passed ? "PASS" : "FAIL")
    for failure in validation.failures
        println(io, "  failed: $failure")
    end
end

function validate_result(config, model, dmrg_diagnostics, energy,
                         kinetic_energy, potential_energy, comparison)
    # TCI's estimate and DMRG's discarded weight are algorithm diagnostics.
    # Individual <T> and <V> errors are first order in the state error, which
    # scales as sqrt(discarded weight); they must not use the much tighter
    # total-energy tolerance.
    algorithm_error = max(
        config.tci_tolerance,
        model.tci_diagnostics.estimated_relative_error,
    ) + max(
        config.dmrg_cutoff,
        dmrg_diagnostics.max_truncation_error,
    )
    energy_tolerance = max(1e-8, 20algorithm_error)
    component_tolerance = max(1e-6, 2sqrt(algorithm_error))
    wavefunction_tolerance = max(1e-6, 10sqrt(algorithm_error))
    overlap_tolerance = max(1e-10, wavefunction_tolerance^2)

    checks = (
        energy=abs(energy - 0.5) < energy_tolerance,
        kinetic_energy=abs(kinetic_energy - 0.25) < component_tolerance,
        potential_energy=abs(potential_energy - 0.25) < component_tolerance,
        wavefunction=comparison.l2_error < wavefunction_tolerance,
        overlap=comparison.overlap > 1 - overlap_tolerance,
    )
    failures = String[string(name) for (name, passed) in pairs(checks) if !passed]
    return (;
        passed=isempty(failures),
        failures,
        checks,
        algorithm_error,
        energy_tolerance,
        component_tolerance,
        wavefunction_tolerance,
        overlap_tolerance,
    )
end

function main(config::SHOConfig=SHOConfig())
    model = build_sho(config)
    energy, state, dmrg_diagnostics = ground_state_with_dimension_log(
        model.hamiltonian,
        model.sites,
        joinpath(@__DIR__, "sho_mps_dimensions.csv");
        nsweeps=config.sweeps,
        maxdim=config.max_bond_dimension,
        cutoff=config.dmrg_cutoff,
        seed=1,
        outputlevel=0,
    )

    kinetic_energy = expect_mpo(state, model.kinetic)
    potential_energy = expect_mpo(state, model.potential)
    comparison = align_and_compare(
        state,
        model.sites,
        config.half_width,
        config.realspace_points,
    )
    validation = validate_result(
        config,
        model,
        dmrg_diagnostics,
        energy,
        kinetic_energy,
        potential_energy,
        comparison,
    )

    print_report(
        stdout,
        config,
        model,
        energy,
        state,
        dmrg_diagnostics,
        kinetic_energy,
        potential_energy,
        comparison,
        validation,
    )

    report_path = joinpath(@__DIR__, "sho_results.txt")
    open(report_path, "w") do io
        print_report(
            io,
            config,
            model,
            energy,
            state,
            dmrg_diagnostics,
            kinetic_energy,
            potential_energy,
            comparison,
            validation,
        )
    end

    write_csv(joinpath(@__DIR__, "sho_ground_state.csv"), comparison)
    write_tensor_shapes(
        joinpath(@__DIR__, "sho_tensor_shapes.txt"),
        state,
        model.sites,
        model,
    )
    plot_realspace(
        comparison.x,
        comparison.numerical;
        filename=joinpath(@__DIR__, "sho_ground_state.svg"),
        quantity=:real,
        title="1D SHO ground state from Fourier-QTT DMRG (DFT potential)",
    )
    plot_realspace(
        comparison.x,
        comparison.error;
        filename=joinpath(@__DIR__, "sho_error.svg"),
        quantity=:abs,
        title="Absolute wavefunction error",
    )

    validation.passed || error(
        "SHO validation failed ($(join(validation.failures, ", "))); " *
        "full diagnostics were saved to $report_path",
    )

    return (;
        model,
        energy,
        state,
        dmrg_diagnostics,
        kinetic_energy,
        potential_energy,
        comparison,
        validation,
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) <= 1 || error("usage: julia sho_model.jl [realspace_points]")
    config = isempty(ARGS) ? SHOConfig() :
             SHOConfig(realspace_points=parse(Int, only(ARGS)))
    main(config)
end
