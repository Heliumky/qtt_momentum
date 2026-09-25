using ITensors, ITensorMPS
using LinearAlgebra

@testset "Non-separable multi-dimensional potentials" begin
    qubits_per_dim = [3, 4]
    half_widths = [2.0, 3.0]
    offsets = [0.5, 0.0]
    potential(x, y) = exp(-(x^2 + 2y^2)) + 0.1x * y

    grids = [realspace_grid(2^qubits_per_dim[d], half_widths[d]; offset=offsets[d]) for d in 1:2]
    samples = [potential(x, y) for x in grids[1], y in grids[2]]
    U = kron(
        dense_fourier_matrix(qubits_per_dim[2], half_widths[2]; offset=offsets[2]),
        dense_fourier_matrix(qubits_per_dim[1], half_widths[1]; offset=offsets[1]),
    )
    reference = U * Diagonal(vec(samples)) * U'

    @testset "$scheme layout" for scheme in (:block, :interleaved)
        layout = RegisterLayout(qubits_per_dim; scheme)
        sites = siteinds("Qubit", sum(qubits_per_dim))

        fit = realspace_function_mps_nd(potential, layout, half_widths;
                                        offsets, tolerance=1e-13, return_diagnostics=true)
        values = QTTDFT.register_to_axes(vec(mps_to_dense(fit.tensors)), layout;
                                         register=:realspace)
        @test values ≈ samples atol=1e-11

        model = potential_mpo_dft_nd(sites, potential, layout, half_widths;
                                     offsets, tci_tolerance=1e-13)
        order = vec(QTTDFT.register_to_axes(collect(1:2^sum(qubits_per_dim)), layout))
        Vk = mpo_to_dense(itensor_mpo_to_arrays(model.potential, sites))
        @test Vk[order, order] ≈ reference atol=1e-11
    end

    @testset "cell-averaged Coulomb" begin
        h = 0.1
        # Cell touching the origin: integral over [0,h]^2 of 1/r is 2h*asinh(1).
        @test coulomb2d_cell_average(h / 2, h / 2, h, h) ≈ -2asinh(1) / h rtol=1e-14
        @test coulomb2d_cell_average(-h / 2, h / 2, h, h) ≈ -2asinh(1) / h rtol=1e-14
        # Away from the origin: compare with a fine midpoint sum.
        midpoint(x, y, hx, hy; n=400) = -sum(
            1 / hypot(x - hx / 2 + (i - 0.5) * hx / n, y - hy / 2 + (j - 0.5) * hy / n)
            for i in 1:n, j in 1:n
        ) / n^2
        for (x, y, hx, hy) in ((-0.05, 0.15, 0.1, 0.1), (0.35, -0.25, 0.1, 0.2), (3.0, -2.0, 0.1, 0.1))
            @test coulomb2d_cell_average(x, y, hx, hy) ≈ midpoint(x, y, hx, hy) rtol=1e-6
        end
        @test coulomb2d_cell_average(3.0, 2.0, 0.1, 0.1; charge=2) ≈
              2coulomb2d_cell_average(3.0, 2.0, 0.1, 0.1)
        @test_throws ArgumentError coulomb2d_cell_average(0.0, 0.3, 0.1, 0.1)
    end

    @testset "2D Coulomb bound state" begin
        # Cell-centred grids never sample the r=0 singularity; the pivot at the
        # origin seeds TCI at the potential's extremum.
        qubits = [5, 5]
        Ls = [6.0, 6.0]
        sites = siteinds("Qubit", sum(qubits))
        model = build_dft_hamiltonian_nd(
            sites, (x, y) -> -1 / hypot(x, y), qubits, Ls;
            offsets=[0.5, 0.5], pivot_points=[(0.0, 0.0)], tci_tolerance=1e-10,
        )
        @test length(model.kinetic_terms) == 2
        @test model.tci_diagnostics.global_error_estimate < 1e-8

        dense = mpo_to_dense(itensor_mpo_to_arrays(model.hamiltonian, sites))
        @test norm(dense - dense') < 1e-9
        exact = eigmin(Hermitian((dense + dense') / 2))
        @test exact < 0

        energy, state = ground_state(model.hamiltonian, sites;
                                     nsweeps=12, maxdim=64, cutoff=1e-12, outputlevel=0)
        @test energy ≈ exact atol=1e-6

        guess = realspace_state_mps_nd((x, y) -> exp(-2hypot(x, y)), sites, qubits, Ls;
                                       offsets=[0.5, 0.5], pivot_points=[(0.0, 0.0)])
        @test norm(guess) ≈ 1 atol=1e-12
        # The guess is already close: its energy is bound and above the ground state.
        guess_energy = real(inner(guess', model.hamiltonian, guess))
        @test exact - 1e-9 <= guess_energy < 0
        guessed_energy, _ = ground_state(model.hamiltonian, sites; initial_state=guess,
                                         nsweeps=12, maxdim=64, cutoff=1e-12, outputlevel=0)
        @test guessed_energy ≈ exact atol=1e-6

        density = abs2.(mps_to_realspace_nd(state, sites, qubits, Ls;
                                            normalize_coefficients=true,
                                            offsets=[0.5, 0.5]).values)
        @test density ≈ transpose(density) atol=1e-6
        @test density ≈ reverse(density) atol=1e-6
    end
end
