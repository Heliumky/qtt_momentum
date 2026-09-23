using ITensors, ITensorMPS
using LinearAlgebra

@testset "Non-separable multi-dimensional potentials" begin
    @testset "grid_potential_fourier_coefficients matches a known finite trig polynomial" begin
        # A finite trig polynomial has no aliasing once N covers its modes,
        # so this checks the exact sign/normalization convention (unlike a
        # non-periodic function such as x^2, whose Fourier series only
        # converges algebraically on this grid).
        N, L = 7, 8.0
        f(x) = 1.0 + 2 * cos(pi * 3x / L) + 5 * sin(pi * 7x / L)
        coefficients = grid_potential_fourier_coefficients(f, [N], [L])
        @test length(coefficients) == 2^N
        # coefficient(delta) = a_{-delta} for f(x) = sum_k a_k*exp(i*pi*k*x/L)
        expected = Dict(0 => complex(1.0), 3 => complex(1.0), -3 => complex(1.0),
                        7 => 2.5im, -7 => -2.5im)
        for (delta, value) in expected
            idx = mod(delta, 2^N) + 1
            @test coefficients[idx] ≈ value atol=1e-10
        end
        for delta in (1, 2, 4, 5, 6, 10)
            idx = mod(delta, 2^N) + 1
            @test coefficients[idx] ≈ 0 atol=1e-10
        end
    end

    @testset "potential_mpo_tci_nd reproduces a separable potential exactly" begin
        # A separable V(x,y)=0.5(x^2+y^2) lets the general non-separable ND
        # fit be cross-checked against the already-validated separable path
        # (build_separable_hamiltonian), including the known exact E0=1.
        Nx, Ny, Lx, Ly = 5, 5, 8.0, 8.0
        qubits_per_dim = [Nx, Ny]
        half_widths = [Lx, Ly]
        sites = siteinds("Qubit", Nx + Ny)

        nd_fit = potential_mpo_tci_nd(
            (x, y) -> 0.5 * (x^2 + y^2), qubits_per_dim, half_widths;
            tolerance=1e-10, maxbonddim=200, return_diagnostics=true,
        )
        potential_mpo = array_mpo_to_itensor(sites, nd_fit.tensors)
        kinetic_mpo_x = array_mpo_to_itensor(sites, embed_mpo(kinetic_mpo(Nx, Lx), 1, qubits_per_dim))
        kinetic_mpo_y = array_mpo_to_itensor(sites, embed_mpo(kinetic_mpo(Ny, Ly), 2, qubits_per_dim))
        H = +(kinetic_mpo_x, kinetic_mpo_y; alg="directsum")
        H = +(H, potential_mpo; alg="directsum")

        energy, state = ground_state(H, sites; nsweeps=16, maxdim=32, cutoff=1e-12, outputlevel=0)
        @test energy ≈ 1.0 atol=1e-4

        separable_model = build_separable_hamiltonian(
            sites, qubits_per_dim, half_widths;
            powers=[2, 2], scales=[0.5, 0.5], tci_tolerance=1e-12,
        )
        separable_energy, _ = ground_state(
            separable_model.hamiltonian, sites; nsweeps=16, maxdim=32, cutoff=1e-12, outputlevel=0,
        )
        @test energy ≈ separable_energy atol=1e-6
    end

    @testset "potential_mpo_tci_nd for a softened 2D Coulomb potential vs. dense diagonalization" begin
        Nx, Ny, Lx, Ly, softening = 5, 5, 8.0, 8.0, 0.5
        qubits_per_dim = [Nx, Ny]
        half_widths = [Lx, Ly]
        V(x, y) = -1.0 / sqrt(x^2 + y^2 + softening^2)

        pot_fit = potential_mpo_tci_nd(
            V, qubits_per_dim, half_widths;
            tolerance=1e-10, maxbonddim=300, return_diagnostics=true,
        )
        @test pot_fit.diagnostics.estimated_relative_error <= 1e-10
        @test max_bond_dim(pot_fit.tensors) < 2^min(Nx, Ny)

        sites = siteinds("Qubit", Nx + Ny)
        potential_mpo = array_mpo_to_itensor(sites, pot_fit.tensors)
        kinetic_mpo_x = array_mpo_to_itensor(sites, embed_mpo(kinetic_mpo(Nx, Lx), 1, qubits_per_dim))
        kinetic_mpo_y = array_mpo_to_itensor(sites, embed_mpo(kinetic_mpo(Ny, Ly), 2, qubits_per_dim))
        H = +(kinetic_mpo_x, kinetic_mpo_y; alg="directsum")
        H = +(H, potential_mpo; alg="directsum")

        dense_potential = mpo_to_dense(pot_fit.tensors)
        dense_kinetic = kron(Matrix{ComplexF64}(I, 2^Ny, 2^Ny), mpo_to_dense(kinetic_mpo(Nx, Lx))) +
                        kron(mpo_to_dense(kinetic_mpo(Ny, Ly)), Matrix{ComplexF64}(I, 2^Nx, 2^Nx))
        dense_hamiltonian = dense_kinetic + dense_potential
        exact_energy = eigmin(Hermitian((dense_hamiltonian + dense_hamiltonian') / 2))
        @test exact_energy < 0

        energy, state = ground_state(H, sites; nsweeps=18, maxdim=64, cutoff=1e-12, outputlevel=0)
        @test energy ≈ exact_energy atol=1e-6
    end

    @testset "coefficient_mpo_tci_nd for the analytic (bare) 2D Coulomb potential vs. dense diagonalization" begin
        # coulomb2d_fourier_coeff evaluates the exact finite-rectangle cosine
        # integral with its singularity removed analytically; no grid sampling
        # or softening is involved. coefficient_mpo_tci_nd fits it directly.
        Nx, Ny, Lx, Ly = 5, 5, 8.0, 8.0
        qubits_per_dim = [Nx, Ny]
        half_widths = [Lx, Ly]
        coulomb_coeff(kx, ky) = coulomb2d_fourier_coeff(kx, ky, Lx, Ly)

        pot_fit = coefficient_mpo_tci_nd(
            coulomb_coeff, qubits_per_dim;
            tolerance=1e-10, maxbonddim=300, return_diagnostics=true,
        )
        @test pot_fit.diagnostics.estimated_relative_error <= 1e-10

        sites = siteinds("Qubit", Nx + Ny)
        potential_mpo = array_mpo_to_itensor(sites, pot_fit.tensors)
        kinetic_mpo_x = array_mpo_to_itensor(sites, embed_mpo(kinetic_mpo(Nx, Lx), 1, qubits_per_dim))
        kinetic_mpo_y = array_mpo_to_itensor(sites, embed_mpo(kinetic_mpo(Ny, Ly), 2, qubits_per_dim))
        H = +(kinetic_mpo_x, kinetic_mpo_y; alg="directsum")
        H = +(H, potential_mpo; alg="directsum")

        dense_potential = mpo_to_dense(pot_fit.tensors)
        @test dense_potential ≈ dense_potential' atol=1e-8
        @test coulomb_coeff(1, 0) != periodic_coulomb2d_fourier_coeff(1, 0, Lx, Ly)
        dense_kinetic = kron(Matrix{ComplexF64}(I, 2^Ny, 2^Ny), mpo_to_dense(kinetic_mpo(Nx, Lx))) +
                        kron(mpo_to_dense(kinetic_mpo(Ny, Ly)), Matrix{ComplexF64}(I, 2^Nx, 2^Nx))
        dense_hamiltonian = dense_kinetic + dense_potential
        exact_energy = eigmin(Hermitian((dense_hamiltonian + dense_hamiltonian') / 2))
        @test exact_energy < 0

        energy, state = ground_state(H, sites; nsweeps=18, maxdim=64, cutoff=1e-12, outputlevel=0)
        @test energy ≈ exact_energy atol=1e-6

        # Both finite-box calculations should remain bound and above the
        # isolated continuum value. Monotonicity is not asserted at fixed N:
        # increasing L simultaneously coarsens the resolved real-space scale.
        function ground_energy(N, L)
            dims = [N, N]
            coeff(kx, ky) = coulomb2d_fourier_coeff(kx, ky, L, L)
            pot = coefficient_mpo_tci_nd(coeff, dims; tolerance=1e-10, maxbonddim=300)
            trial_sites = siteinds("Qubit", 2N)
            trial_H = +(
                array_mpo_to_itensor(trial_sites, embed_mpo(kinetic_mpo(N, L), 1, dims)),
                array_mpo_to_itensor(trial_sites, embed_mpo(kinetic_mpo(N, L), 2, dims));
                alg="directsum",
            )
            trial_H = +(trial_H, array_mpo_to_itensor(trial_sites, pot); alg="directsum")
            trial_energy, _ = ground_state(trial_H, trial_sites; nsweeps=18, maxdim=64, cutoff=1e-12, outputlevel=0)
            return trial_energy
        end
        smaller_box_energy = ground_energy(7, 8.0)
        larger_box_energy = ground_energy(7, 16.0)
        @test -2 < smaller_box_energy < 0
        @test -2 < larger_box_energy < 0
    end
end
