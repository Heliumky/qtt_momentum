using ITensors, ITensorMPS
using LinearAlgebra

@testset "Separable multi-dimensional Hamiltonians" begin
    Nx, Ny, Lx, Ly = 5, 5, 8.0, 8.0
    qubits_per_dim = [Nx, Ny]
    half_widths = [Lx, Ly]

    @testset "identity_mpo and embed_mpo" begin
        id3 = identity_mpo(3)
        @test length(id3) == 3
        @test all(t -> size(t) == (1, 2, 2, 1), id3)
        @test mpo_to_dense(id3) ≈ Matrix{Float64}(I, 8, 8)

        component_x = kinetic_mpo(Nx, Lx)
        embedded_x = embed_mpo(component_x, 1, qubits_per_dim)
        @test length(embedded_x) == Nx + Ny
        @test mpo_to_dense(embedded_x) ≈
              kron(Matrix{Float64}(I, 2^Ny, 2^Ny), mpo_to_dense(component_x))

        component_y = kinetic_mpo(Ny, Ly)
        embedded_y = embed_mpo(component_y, 2, qubits_per_dim)
        @test mpo_to_dense(embedded_y) ≈
              kron(mpo_to_dense(component_y), Matrix{Float64}(I, 2^Nx, 2^Nx))

        @test_throws ArgumentError embed_mpo(component_x, 3, qubits_per_dim)
        @test_throws DimensionMismatch embed_mpo(identity_mpo(2), 1, qubits_per_dim)
    end

    @testset "build_separable_hamiltonian ground state" begin
        sites = siteinds("Qubit", Nx + Ny)
        model = build_separable_hamiltonian(
            sites, qubits_per_dim, half_widths;
            powers=[2, 2], scales=[0.5, 0.5], tci_tolerance=1e-12,
        )

        @test length(model.hamiltonian) == Nx + Ny
        @test length(model.kinetic_terms) == 2
        @test length(model.potential_terms) == 2
        @test max_bond_dim(model.kinetic_terms[1]) == 3
        @test max_bond_dim(model.kinetic_terms[2]) == 3
        @test all(d -> d.estimated_relative_error <= 1e-12, model.tci_diagnostics)

        # A 2D separable Hamiltonian is the direct sum of two 1D dense
        # Hamiltonians (x indexes the faster/first block of sites).
        Hx = mpo_to_dense(kinetic_mpo(Nx, Lx)) +
             mpo_to_dense(power_mpo_tci(2, Nx, Lx; scale=0.5, tolerance=1e-12))
        Hy = mpo_to_dense(kinetic_mpo(Ny, Ly)) +
             mpo_to_dense(power_mpo_tci(2, Ny, Ly; scale=0.5, tolerance=1e-12))
        dense_hamiltonian = kron(Matrix{ComplexF64}(I, 2^Ny, 2^Ny), Hx) +
                             kron(Hy, Matrix{ComplexF64}(I, 2^Nx, 2^Nx))
        exact_energy = eigmin(Hermitian((dense_hamiltonian + dense_hamiltonian') / 2))
        @test exact_energy ≈ 1.0 atol=1e-8

        energy, state = ground_state(
            model.hamiltonian, sites;
            nsweeps=14, maxdim=32, cutoff=1e-12, outputlevel=0,
        )
        @test energy ≈ exact_energy atol=1e-6

        @test expect_mpo(state, model.kinetic_terms[1]) ≈ 0.25 atol=1e-3
        @test expect_mpo(state, model.kinetic_terms[2]) ≈ 0.25 atol=1e-3
        @test expect_mpo(state, model.potential_terms[1]) ≈ 0.25 atol=1e-3
        @test expect_mpo(state, model.potential_terms[2]) ≈ 0.25 atol=1e-3

        @test_throws DimensionMismatch build_separable_hamiltonian(
            sites, [Nx, Ny, 2], half_widths,
        )
    end

    @testset "coeffs_to_realspace_nd matches an outer product of 1D reconstructions" begin
        Nx2, Ny2, Lx2, Ly2 = 4, 5, 3.0, 4.0
        coeff_x = zeros(ComplexF64, 2^Nx2)
        coeff_x[3] = 1.0 # n = 2
        coeff_y = zeros(ComplexF64, 2^Ny2)
        coeff_y[2] = 0.5 # n = 1
        coeff_2d = coeff_x * transpose(coeff_y)

        xg = collect(range(-2.7, 2.1; length=11))
        yg = collect(range(-3.5, 3.9; length=13))
        values = coeffs_to_realspace_nd(coeff_2d, [xg, yg], [Lx2, Ly2])
        expected = fourier_basis(2, xg, Lx2) * transpose(0.5 .* fourier_basis(1, yg, Ly2))
        @test values ≈ expected atol=1e-12
    end

    @testset "mps_to_realspace_nd matches direct evaluation of its own coefficients" begin
        Nx3, Ny3, Lx3, Ly3 = 4, 3, 3.0, 2.0
        coeff_x = zeros(ComplexF64, 2^Nx3)
        coeff_x[3] = 1.0
        coeff_y = zeros(ComplexF64, 2^Ny3)
        coeff_y[2] = 0.5
        coeff_2d = coeff_x * transpose(coeff_y)

        sites = siteinds("Qubit", Nx3 + Ny3)
        mps_tensors = dense_to_mps(vec_to_site_tensor(vec(coeff_2d), Nx3 + Ny3))
        state = array_mps_to_itensor(sites, mps_tensors)

        reconstructed = mps_to_realspace_nd(state, sites, [Nx3, Ny3], [Lx3, Ly3])
        @test reconstructed.coefficients ≈ coeff_2d atol=1e-12
        @test length(reconstructed.grids[1]) == 2^Nx3
        @test length(reconstructed.grids[2]) == 2^Ny3

        expected = coeffs_to_realspace_nd(coeff_2d, reconstructed.grids, [Lx3, Ly3])
        @test reconstructed.values ≈ expected atol=1e-10

        custom = mps_to_realspace_nd(
            state, sites, [Nx3, Ny3], [Lx3, Ly3]; points_per_dim=[9, 7],
        )
        @test length(custom.grids[1]) == 9
        @test length(custom.grids[2]) == 7

        svg = plot_heatmap(
            reconstructed.grids[1], reconstructed.grids[2], real.(reconstructed.values);
            title="test heatmap",
        )
        @test startswith(svg, "<svg")
        @test occursin("test heatmap", svg)
        @test occursin("rect", svg)
    end
end
