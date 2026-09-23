using LinearAlgebra

function x2_coefficient(k::Integer, L::Real; scale::Number=1)
    iszero(k) && return scale * L^2 / 3
    endpoint_phase = isodd(k) ? -1.0 : 1.0
    return scale * 2L^2 * endpoint_phase / (pi^2 * k^2)
end

@testset "analytic elementary functions and direct TCI" begin
    N, L = 5, 8.0

    @test freqs(3) == [0, 1, 2, 3, -4, -3, -2, -1]
    @test sigma_to_n([1, 2, 1, 2]) == -6
    @test_throws ArgumentError sigma_to_n([0, 1])

    for k in (0, 1, -2, 7)
        @test power_fourier_coeff(2, k, L) ≈ x2_coefficient(k, L) atol=1e-13
    end
    for power in 0:7, k in (1, 2, 7)
        positive = power_fourier_coeff(power, k, L; scale=0.3)
        negative = power_fourier_coeff(power, -k, L; scale=0.3)
        @test negative ≈ conj(positive) atol=1e-13
    end
    @test power_fourier_coeff(3, 0, L) == 0

    direct_qtt = fit_qtt_mps(
        sigma -> sigma_to_n(sigma)^2,
        N;
        tolerance=1e-12,
    )
    @test vec(mps_to_dense(direct_qtt)) ≈ freqs(N) .^ 2 atol=1e-9
    @test all(shape -> length(shape) == 3 && shape[2] == 2, mps_shapes(direct_qtt))

    coefficients = power_mps(2, N, L; scale=0.5, tolerance=1e-12)
    exact_coefficients = x2_coefficient.(freqs(N), L; scale=0.5)
    @test vec(mps_to_dense(coefficients)) ≈ exact_coefficients atol=1e-9

    potential_fit = power_mpo_tci(
        2,
        N,
        L;
        scale=0.5,
        ordering=:lsb_first,
        tolerance=1e-12,
        return_diagnostics=true,
    )
    potential = potential_fit.tensors
    modes = freqs(N)
    exact_operator = [
        power_fourier_coeff(2, modes[column] - modes[row], L; scale=0.5)
        for row in 1:2^N, column in 1:2^N
    ]
    @test mpo_to_dense(potential) ≈ exact_operator atol=1e-8
    @test all(
        shape -> length(shape) == 4 && shape[2:3] == (2, 2),
        mpo_shapes(potential),
    )
    @test potential_fit.diagnostics.requested_tolerance == 1e-12
    @test potential_fit.diagnostics.estimated_relative_error <= 1e-12
    @test potential_fit.diagnostics.target_evaluations > 0
    @test_throws ArgumentError power_mpo_tci(2, N, L; ordering=:unknown)
end

@testset "coulomb2d_fourier_coeff" begin
    Lx, Ly = 8.0, 6.0

    expected_dc = -(Lx * asinh(Ly / Lx) + Ly * asinh(Lx / Ly)) / (Lx * Ly)
    @test coulomb2d_fourier_coeff(0, 0, Lx, Ly) ≈ expected_dc atol=1e-14
    @test periodic_coulomb2d_fourier_coeff(0, 0, Lx, Ly) == 0
    @test_throws ArgumentError coulomb2d_fourier_coeff(
        0, 0, Lx, Ly; quadrature_tolerance=0.0,
    )
    @test_throws ArgumentError coulomb2d_fourier_coeff(1, 1, -1.0, Ly)

    # The finite rectangle and centered Coulomb potential are even in each
    # mode independently.
    for (kx, ky) in ((3, 2), (5, 0), (0, 4), (7, 7))
        value = coulomb2d_fourier_coeff(kx, ky, Lx, Ly)
        @test isreal(value)
        @test real(value) < 0 # attractive potential -charge/r has negative coefficients
        @test coulomb2d_fourier_coeff(-kx, ky, Lx, Ly) ≈ value atol=1e-14
        @test coulomb2d_fourier_coeff(kx, -ky, Lx, Ly) ≈ value atol=1e-14
        @test coulomb2d_fourier_coeff(-kx, -ky, Lx, Ly) ≈ value atol=1e-14
    end

    # Independent high-accuracy reference values for the exact finite-box
    # cosine integral (not the infinite-plane 2*pi/|q| approximation).
    @test coulomb2d_fourier_coeff(1, 0, Lx, Ly) ≈ -0.08785406427730724 atol=1e-12
    @test coulomb2d_fourier_coeff(0, 1, Lx, Ly) ≈ -0.07060125874379533 atol=1e-12
    @test coulomb2d_fourier_coeff(1, 1, Lx, Ly) ≈ -0.05216091116820801 atol=1e-12
    @test coulomb2d_fourier_coeff(3, 5, Lx, Ly) ≈ -0.011403238187788487 atol=1e-12

    # The old infinite-plane/periodized coefficient remains separately named.
    kx, ky, charge = 3, 5, 2.5
    q = sqrt((pi * kx / Lx)^2 + (pi * ky / Ly)^2)
    expected = -charge * pi / (2 * Lx * Ly * q)
    @test periodic_coulomb2d_fourier_coeff(kx, ky, Lx, Ly; charge=charge) ≈
          expected atol=1e-14
    @test abs(coulomb2d_fourier_coeff(kx, ky, Lx, Ly; charge=charge) - expected) > 1e-6

    # Square-box zero mode and the separately named periodic nonzero modes.
    L = 8.0
    @test coulomb2d_fourier_coeff(0, 0, L, L) ≈
          -2log(1 + sqrt(2)) / L atol=1e-14
    for (kx, ky) in ((1, 0), (2, 3), (10, 10))
        @test periodic_coulomb2d_fourier_coeff(kx, ky, L, L) ≈
              -1 / (2L * sqrt(kx^2 + ky^2)) atol=1e-14
    end
end
