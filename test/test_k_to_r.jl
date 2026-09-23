using ITensors, ITensorMPS

@testset "Fourier coefficients to real space" begin
    N, L = 5, 4.0
    coefficients = zeros(ComplexF64, 2^N)
    coefficients[3] = 1 # n = 2
    x = realspace_grid(length(coefficients), L)
    direct = evaluate_fourier_series(coefficients, x, L)
    fast = coeffs_to_realspace(coefficients, L)

    @test fast ≈ direct atol=1e-13
    @test direct ≈ fourier_basis(2, x, L) atol=1e-13

    sites = siteinds("Qubit", N)
    coefficient_mps = dense_to_mps(vec_to_site_tensor(coefficients, N))
    state = array_mps_to_itensor(sites, coefficient_mps)

    selected_x = collect(range(-3.7, 2.9; length=37))
    selected = mps_to_realspace(state, sites, L; x=selected_x)
    @test selected.x == selected_x
    @test selected.values ≈ fourier_basis(2, selected_x, L) atol=1e-12
    @test selected.method == :direct

    resized = mps_to_realspace(state, sites, L; npoints=53)
    @test length(resized.x) == 53
    @test resized.values ≈ fourier_basis(2, resized.x, L) atol=1e-12
    @test resized.method == :direct

    native = mps_to_realspace(state, sites, L)
    @test length(native.x) == 2^N
    @test native.method == :ifft
    @test_throws ArgumentError mps_to_realspace(
        state,
        sites,
        L;
        x=selected_x,
        npoints=53,
    )

    svg = plot_realspace(x, fast; quantity=:abs, title="single Fourier mode")
    @test startswith(svg, "<svg")
    @test occursin("single Fourier mode", svg)
    @test occursin("polyline", svg)
end
