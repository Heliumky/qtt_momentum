using ITensors, ITensorMPS
using LinearAlgebra

"""Dense matrix of an array MPO whose input legs are the real-space register,
with columns permuted to natural grid order."""
function _dense_grid_columns(tensors)
    dense = mpo_to_dense(tensors)
    ordered = similar(dense)
    ordered[:, grid_register_order(length(tensors))] = dense
    return ordered
end

@testset "DFT MPO" begin
    L = 2.5
    for N in (2, 5, 8), offset in (0.0, 0.25, 0.5)
        transform = fourier_transform_mpo(N, L; offset)
        @test all(tensor -> size(tensor, 2) == size(tensor, 3) == 2, transform)
        U = _dense_grid_columns(transform)
        @test U ≈ dense_fourier_matrix(N, L; offset) atol=1e-12
        @test U * U' ≈ I atol=1e-12
    end
    @test max_bond_dim(fourier_transform_mpo(10, L)) <= 16
    @test_throws ArgumentError fourier_transform_mpo(1, L)

    # U maps grid samples of a Fourier mode onto the corresponding unit vector:
    # phi_n(x_j) = sqrt(M/2L) * conj(U[n, j]).
    N = 6
    U = dense_fourier_matrix(N, L; offset=0.5)
    x = realspace_grid(2^N, L; offset=0.5)
    mode = 3
    samples = fourier_basis(mode, x, L)
    coefficients = U * samples .* sqrt(2L / 2^N)
    expected = zeros(ComplexF64, 2^N)
    expected[findfirst(==(mode), freqs(N))] = 1
    @test coefficients ≈ expected atol=1e-12
end

@testset "Fourier-space potentials" begin
    N, L = 6, 4.0
    sites = siteinds("Qubit", N)
    potential(x) = 0.5x^2 + 0.2x^3 - exp(-x^2)

    for offset in (0.0, 0.5)
        fit = potential_mpo_dft(sites, potential, N, L; offset, tci_tolerance=1e-13)
        Vk = mpo_to_dense(itensor_mpo_to_arrays(fit.potential, sites))
        U = dense_fourier_matrix(N, L; offset)
        reference = U * Diagonal(potential.(realspace_grid(2^N, L; offset))) * U'
        @test Vk ≈ reference atol=1e-11
        @test norm(Vk - Vk') < 1e-11
        @test fit.tci_diagnostics.estimated_relative_error < 1e-12
    end

    # A band-limited potential has exact Fourier coefficients:
    # cos(pi*x/L) = (phi_1 + phi_{-1}) * sqrt(2L)/2, so V_k[n, n±1] = 1/2.
    fit = potential_mpo_dft(sites, x -> cos(pi * x / L), N, L)
    Vk = mpo_to_dense(itensor_mpo_to_arrays(fit.potential, sites))
    modes = freqs(N)
    expected = [abs(n_in - n_out) == 1 ? 0.5 : 0.0 for n_out in modes, n_in in modes]
    # The only periodic wrap-around couples the two band edges -2^(N-1), 2^(N-1)-1.
    wrap = [abs(n_in - n_out) == 2^N - 1 ? 0.5 : 0.0 for n_out in modes, n_in in modes]
    @test Vk ≈ expected + wrap atol=1e-12

    power = power_potential_mpo_dft(sites, 2, N, L; scale=0.5)
    direct = potential_mpo_dft(sites, x -> 0.5x^2, N, L)
    @test mpo_to_dense(itensor_mpo_to_arrays(power.potential, sites)) ≈
          mpo_to_dense(itensor_mpo_to_arrays(direct.potential, sites)) atol=1e-12
end
