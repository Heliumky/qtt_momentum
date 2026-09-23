using ITensors, ITensorMPS
using LinearAlgebra

"""Permutation `p` with `p[axis_linear] = register_linear` for dense checks."""
_axis_order(layout) =
    vec(QTTDFT.register_to_axes(collect(1:2^sum(layout.qubits_per_dim)), layout))

@testset "Separable multi-dimensional Hamiltonians" begin
    Nx, Ny, Lx, Ly = 4, 5, 6.0, 7.0
    qubits_per_dim = [Nx, Ny]
    half_widths = [Lx, Ly]

    @testset "register layouts" begin
        block = RegisterLayout(qubits_per_dim)
        @test block.scheme === :block
        @test dimension_sites(block, 1) == 1:Nx
        @test dimension_sites(block, 2) == (Nx + 1):(Nx + Ny)
        interleaved = RegisterLayout(qubits_per_dim; scheme=:interleaved)
        @test dimension_sites(interleaved, 1) == [1, 3, 5, 7]
        @test dimension_sites(interleaved, 2) == [2, 4, 6, 8, 9]
        @test_throws ArgumentError RegisterLayout(qubits_per_dim; scheme=:fused)
    end

    @testset "identity_mpo, embed_mpo, kron_mpo ($scheme)" for scheme in (:block, :interleaved)
        layout = RegisterLayout(qubits_per_dim; scheme)
        order = _axis_order(layout)
        id3 = identity_mpo(3)
        @test all(t -> size(t) == (1, 2, 2, 1), id3)
        @test mpo_to_dense(id3) ≈ Matrix{Float64}(I, 8, 8)

        component_x = kinetic_mpo(Nx, Lx)
        component_y = kinetic_mpo(Ny, Ly)
        embedded_x = mpo_to_dense(embed_mpo(component_x, 1, layout))[order, order]
        embedded_y = mpo_to_dense(embed_mpo(component_y, 2, layout))[order, order]
        # Axis order: x is the faster (first) axis.
        @test embedded_x ≈ kron(Matrix{Float64}(I, 2^Ny, 2^Ny), mpo_to_dense(component_x))
        @test embedded_y ≈ kron(mpo_to_dense(component_y), Matrix{Float64}(I, 2^Nx, 2^Nx))

        product = kron_mpo([embed_mpo(component_x, 1, layout), embed_mpo(component_y, 2, layout)])
        @test mpo_to_dense(product)[order, order] ≈ embedded_x * embedded_y

        @test_throws ArgumentError embed_mpo(component_x, 3, layout)
        @test_throws DimensionMismatch embed_mpo(identity_mpo(2), 1, layout)
    end

    @testset "joint DFT MPO ($scheme)" for scheme in (:block, :interleaved)
        layout = RegisterLayout(qubits_per_dim; scheme)
        offsets = [0.5, 0.0]
        U = mpo_to_dense(fourier_transform_mpo_nd(layout, half_widths; offsets))
        fourier_order = _axis_order(layout)
        realspace_order = vec(QTTDFT.register_to_axes(
            collect(1:2^(Nx + Ny)), layout; register=:realspace,
        ))
        reference = kron(
            dense_fourier_matrix(Ny, Ly; offset=offsets[2]),
            dense_fourier_matrix(Nx, Lx; offset=offsets[1]),
        )
        @test U[fourier_order, realspace_order] ≈ reference atol=1e-11
    end

    @testset "build_separable_hamiltonian ground state ($scheme)" for scheme in (:block, :interleaved)
        sites = siteinds("Qubit", Nx + Ny)
        model = build_separable_hamiltonian(
            sites, qubits_per_dim, half_widths;
            powers=[2, 2], scales=[0.5, 0.5], tci_tolerance=1e-12, scheme,
        )
        @test length(model.hamiltonian) == Nx + Ny
        @test length(model.kinetic_terms) == 2
        @test length(model.potential_terms) == 2
        @test max_bond_dim(model.kinetic_terms[1]) == 3
        @test max_bond_dim(model.kinetic_terms[2]) == 3
        @test all(d -> d.estimated_relative_error <= 1e-12, model.tci_diagnostics)

        dense_hamiltonian = mpo_to_dense(itensor_mpo_to_arrays(model.hamiltonian, sites))
        exact_energy = eigmin(Hermitian((dense_hamiltonian + dense_hamiltonian') / 2))
        # Nx=4 on [-6,6) resolves the SHO only to ~1e-3; Ny=5 is converged.
        @test exact_energy ≈ 1.0 atol=5e-3

        energy, state = ground_state(
            model.hamiltonian, sites;
            nsweeps=14, maxdim=32, cutoff=1e-12, outputlevel=0,
        )
        @test energy ≈ exact_energy atol=1e-6

        reconstruction = mps_to_realspace_nd(state, sites, model.layout, half_widths;
                                             normalize_coefficients=true)
        @test reconstruction.method == :dft_mpo
        @test size(reconstruction.values) == (2^Nx, 2^Ny)
        cell = (2Lx / 2^Nx) * (2Ly / 2^Ny)
        @test sum(abs2, reconstruction.values) * cell ≈ 1 atol=1e-8

        direct = mps_to_realspace_nd(state, sites, model.layout, half_widths;
                                     normalize_coefficients=true,
                                     points_per_dim=[2^Nx + 1, 2^Ny + 3])
        @test direct.method == :direct
        @test size(direct.values) == (2^Nx + 1, 2^Ny + 3)
    end
end
