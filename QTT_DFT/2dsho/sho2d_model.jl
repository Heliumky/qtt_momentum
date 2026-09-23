#!/usr/bin/env julia

"""
End-to-end two-dimensional harmonic-oscillator validation in a Fourier QTT basis.

The model is H = -1/2(d²/dx²+d²/dy²) + (x²+y²)/2 with ℏ=m=ω=1. Each dimension
occupies its own contiguous block of qubits (`N_list = [Nx, Ny]`, in the
per-dimension style of `3dH_p0_l/onepar.py`'s `N_list`). Each dimension's
potential is fitted by TCI on its real-space grid and rotated to Fourier space
with the quantics DFT MPO (V_k = U diag(V) U†), each dimension's kinetic term
is an exact Fourier-space MPO, the two
dimensions are combined by an exact ITensor direct sum, and the ground state
is obtained with ITensor DMRG.
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

Base.@kwdef struct SHO2DConfig
    N_list::Vector{Int} = [8, 8]
    half_width_list::Vector{Float64} = [8.0, 8.0]
    tci_tolerance::Float64 = 1e-12
    dft_cutoff::Float64 = 1e-26
    sweeps::Int = 20
    max_bond_dimension::Int = 10000000
    dmrg_cutoff::Float64 = 1e-8
    realspace_points_list::Vector{Int} = [121, 121]
end

function build_sho2d(config::SHO2DConfig)
    length(config.N_list) == 2 || throw(ArgumentError("N_list must have two entries"))
    sites = siteinds("Qubit", sum(config.N_list))

    # V(x,y) = (x²+y²)/2 is the sum of two independently TCI-fitted 1D potentials.
    return build_separable_hamiltonian(
        sites,
        config.N_list,
        config.half_width_list;
        powers=[2, 2],
        scales=[0.5, 0.5],
        pivot_points=[0.0],
        tci_tolerance=config.tci_tolerance,
        cutoff=config.dft_cutoff,
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

function align_and_compare(state::MPS, sites, config::SHO2DConfig)
    reconstructed = mps_to_realspace_nd(
        state,
        sites,
        config.N_list,
        config.half_width_list;
        normalize_coefficients=true,
        points_per_dim=config.realspace_points_list,
    )
    x, y = reconstructed.grids
    numerical = reconstructed.values
    dx = 2config.half_width_list[1] / length(x)
    dy = 2config.half_width_list[2] / length(y)

    X = reshape(x, :, 1)
    Y = reshape(y, 1, :)
    exact = ComplexF64.(pi^(-1 / 2) .* exp.(-(X .^ 2 .+ Y .^ 2) ./ 2))
    exact ./= sqrt(sum(abs2, exact) * dx * dy)

    overlap = sum(conj.(exact) .* numerical) * dx * dy
    numerical = numerical .* exp(-im * angle(overlap))
    error = numerical .- exact

    return (
        x=x,
        y=y,
        numerical=numerical,
        exact=exact,
        error=error,
        l2_error=sqrt(sum(abs2, error) * dx * dy),
        max_error=maximum(abs, error),
        max_imaginary=maximum(abs, imag.(numerical)),
        overlap=abs(sum(conj.(exact) .* numerical) * dx * dy),
        reconstruction_method=:direct,
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
            println(io, "Potential MPO (dimension $d):")
            for (site, shape) in enumerate(mpo_shapes(model.potential_terms[d], sites))
                println(io, "  V$(d)_$site = $shape")
            end
            println(io, "Kinetic MPO (dimension $d):")
            for (site, shape) in enumerate(mpo_shapes(model.kinetic_terms[d], sites))
                println(io, "  T$(d)_$site = $shape")
            end
        end
        println(io, "Hamiltonian MPO:")
        for (site, shape) in enumerate(mpo_shapes(model.hamiltonian, sites))
            println(io, "  W_$site = $shape")
        end
    end
end

function write_csv(path::AbstractString, comparison)
    open(path, "w") do io
        println(io, "x,y,numerical_real,numerical_imag,exact,error_abs")
        for j in eachindex(comparison.y), i in eachindex(comparison.x)
            @printf(
                io,
                "%.16e,%.16e,%.16e,%.16e,%.16e,%.16e\n",
                comparison.x[i],
                comparison.y[j],
                real(comparison.numerical[i, j]),
                imag(comparison.numerical[i, j]),
                real(comparison.exact[i, j]),
                abs(comparison.error[i, j]),
            )
        end
    end
end

function print_report(io::IO, config, model, energy, state, dmrg_diagnostics,
                      kinetic_energies, potential_energies, comparison, validation)
    Nx, Ny = config.N_list
    Lx, Ly = config.half_width_list
    @printf(io, "2D harmonic oscillator: H = -1/2(d²/dx²+d²/dy²) + (x²+y²)/2  (DFT potential)\n")
    @printf(io, "N_list                   = %s\n", string(config.N_list))
    @printf(io, "Fourier modes            = %d x %d\n", 2^Nx, 2^Ny)
    @printf(io, "domain                   = [%.1f, %.1f) x [%.1f, %.1f)\n", -Lx, Lx, -Ly, Ly)
    for d in eachindex(model.qubits_per_dim)
        @printf(io, "potential MPO max bond dim (dim %d) = %d\n", d, max_bond_dim(model.potential_terms[d]))
        @printf(io, "kinetic MPO max bond dim (dim %d)   = %d\n", d, max_bond_dim(model.kinetic_terms[d]))
    end
    @printf(io, "Hamiltonian MPO max bond dim  = %d\n", max_bond_dim(model.hamiltonian))
    @printf(io, "ground-state MPS max bond dim = %d\n", max_bond_dim(state))
    @printf(io, "TCI requested tolerance       = %.6e\n", config.tci_tolerance)
    for d in eachindex(model.tci_diagnostics)
        @printf(io, "TCI estimated relative error (dim %d) = %.6e\n",
                d, model.tci_diagnostics[d].estimated_relative_error)
        @printf(io, "TCI global error estimate (dim %d)    = %.6e\n",
                d, model.tci_diagnostics[d].global_error_estimate)
        @printf(io, "TCI target evaluations (dim %d)       = %d\n",
                d, model.tci_diagnostics[d].target_evaluations)
    end
    @printf(io, "DMRG requested cutoff         = %.6e\n", config.dmrg_cutoff)
    @printf(io, "DMRG max truncation error     = %.6e\n",
            dmrg_diagnostics.max_truncation_error)
    @printf(io, "DMRG truncation error/sweep   = %s\n",
            repr(dmrg_diagnostics.truncation_error_by_sweep))
    @printf(io, "real-space plot points   = %d x %d\n", length(comparison.x), length(comparison.y))
    @printf(io, "real-space grid spacing  = %.8e x %.8e\n",
            2Lx / length(comparison.x), 2Ly / length(comparison.y))
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
    @printf(io, "exact E0                 = %.14f\n", 1.0)
    @printf(io, "energy error             = %+.3e\n", energy - 1.0)
    @printf(io, "<Tx>                      = %.14f  (exact 0.25)\n", kinetic_energies[1])
    @printf(io, "<Ty>                      = %.14f  (exact 0.25)\n", kinetic_energies[2])
    @printf(io, "<Vx>                      = %.14f  (exact 0.25)\n", potential_energies[1])
    @printf(io, "<Vy>                      = %.14f  (exact 0.25)\n", potential_energies[2])
    @printf(io, "<T> + <V>                = %.14f\n",
            sum(kinetic_energies) + sum(potential_energies))
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
                         kinetic_energies, potential_energies, comparison)
    # TCI's estimate and DMRG's discarded weight are algorithm diagnostics.
    # Individual <T> and <V> errors are first order in the state error, which
    # scales as sqrt(discarded weight); they must not use the much tighter
    # total-energy tolerance.
    algorithm_error = max(
        config.tci_tolerance,
        maximum(d -> d.estimated_relative_error, model.tci_diagnostics),
    ) + max(
        config.dmrg_cutoff,
        dmrg_diagnostics.max_truncation_error,
    )
    energy_tolerance = max(1e-8, 20algorithm_error)
    component_tolerance = max(1e-6, 2sqrt(algorithm_error))
    wavefunction_tolerance = max(1e-6, 10sqrt(algorithm_error))
    overlap_tolerance = max(1e-10, wavefunction_tolerance^2)

    checks = (
        energy=abs(energy - 1.0) < energy_tolerance,
        kinetic_energy_x=abs(kinetic_energies[1] - 0.25) < component_tolerance,
        kinetic_energy_y=abs(kinetic_energies[2] - 0.25) < component_tolerance,
        potential_energy_x=abs(potential_energies[1] - 0.25) < component_tolerance,
        potential_energy_y=abs(potential_energies[2] - 0.25) < component_tolerance,
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

function main(config::SHO2DConfig=SHO2DConfig())
    model = build_sho2d(config)
    energy, state, dmrg_diagnostics = ground_state_with_dimension_log(
        model.hamiltonian,
        model.sites,
        joinpath(@__DIR__, "sho2d_mps_dimensions.csv");
        nsweeps=config.sweeps,
        maxdim=config.max_bond_dimension,
        cutoff=config.dmrg_cutoff,
        seed=1,
        outputlevel=0,
    )

    kinetic_energies = [expect_mpo(state, term) for term in model.kinetic_terms]
    potential_energies = [expect_mpo(state, term) for term in model.potential_terms]
    comparison = align_and_compare(state, model.sites, config)
    validation = validate_result(
        config,
        model,
        dmrg_diagnostics,
        energy,
        kinetic_energies,
        potential_energies,
        comparison,
    )

    print_report(
        stdout,
        config,
        model,
        energy,
        state,
        dmrg_diagnostics,
        kinetic_energies,
        potential_energies,
        comparison,
        validation,
    )

    report_path = joinpath(@__DIR__, "sho2d_results.txt")
    open(report_path, "w") do io
        print_report(
            io,
            config,
            model,
            energy,
            state,
            dmrg_diagnostics,
            kinetic_energies,
            potential_energies,
            comparison,
            validation,
        )
    end

    write_csv(joinpath(@__DIR__, "sho2d_ground_state.csv"), comparison)
    write_tensor_shapes(
        joinpath(@__DIR__, "sho2d_tensor_shapes.txt"),
        state,
        model.sites,
        model,
    )
    plot_heatmap(
        comparison.x,
        comparison.y,
        real.(comparison.numerical);
        filename=joinpath(@__DIR__, "sho2d_ground_state.svg"),
        title="2D SHO ground state from Fourier-QTT DMRG",
        colorbar_label="Re ψ",
    )
    plot_heatmap(
        comparison.x,
        comparison.y,
        abs.(comparison.error);
        filename=joinpath(@__DIR__, "sho2d_error.svg"),
        title="Absolute wavefunction error",
        colorbar_label="|error|",
    )

    validation.passed || error(
        "2D SHO validation failed ($(join(validation.failures, ", "))); " *
        "full diagnostics were saved to $report_path",
    )

    return (;
        model,
        energy,
        state,
        dmrg_diagnostics,
        kinetic_energies,
        potential_energies,
        comparison,
        validation,
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) in (0, 2) || error("usage: julia sho2d_model.jl [points_x points_y]")
    config = isempty(ARGS) ? SHO2DConfig() :
             SHO2DConfig(realspace_points_list=[parse(Int, ARGS[1]), parse(Int, ARGS[2])])
    main(config)
end
