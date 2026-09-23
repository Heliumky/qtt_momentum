using ITensors, ITensorMPS
using LinearAlgebra

@testset "DMRG engine and tensor shapes" begin
    N, L = 7, 8.0
    sites = siteinds("Qubit", N)
    model = build_power_hamiltonian(
        sites,
        2,
        N,
        L;
        scale=0.5,
        tci_tolerance=1e-12,
    )

    @test length(model.hamiltonian) == N
    @test length(model.potential) == N
    @test all(shape -> shape[2:3] == (2, 2), mpo_shapes(model.hamiltonian, sites))
    @test max_bond_dim(model.kinetic) == 3
    @test model.tci_diagnostics.requested_tolerance == 1e-12
    @test model.tci_diagnostics.estimated_relative_error <= 1e-12

    dense_hamiltonian = mpo_to_dense(
        mpo_add(model.kinetic_arrays, model.potential_arrays),
    )
    exact_energy = eigmin(Hermitian((dense_hamiltonian + dense_hamiltonian') / 2))
    @test exact_energy ≈ 0.5 atol=2e-6

    energy, state, diagnostics = ground_state(
        model.hamiltonian,
        sites;
        nsweeps=12,
        maxdim=24,
        cutoff=1e-12,
        outputlevel=0,
        return_diagnostics=true,
    )
    @test energy ≈ exact_energy atol=2e-6
    @test length(state) == N

    shapes = mps_shapes(state, sites)
    @test first(shapes)[1] == 1
    @test last(shapes)[3] == 1
    @test all(shape -> shape[2] == 2, shapes)
    @test mps_bond_dims(state) == [shapes[site][3] for site in 1:(N - 1)]
    @test max_bond_dim(state) == maximum(mps_bond_dims(state))
    @test diagnostics.requested_cutoff == 1e-12
    @test length(diagnostics.truncation_error_by_sweep) == 12
    @test diagnostics.max_truncation_error == maximum(diagnostics.truncation_error_by_sweep)
end
