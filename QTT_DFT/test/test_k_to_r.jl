using ITensors, ITensorMPS

@testset "Fourier coefficients to real space" begin
    N, L = 5, 4.0
    coefficients = zeros(ComplexF64, 2^N)
    coefficients[3] = 1 # n = 2
    x = realspace_grid(length(coefficients), L)
    direct = evaluate_fourier_series(coefficients, x, L)
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

    for offset in (0.0, 0.5)
        native = mps_to_realspace(state, sites, L; offset)
        @test length(native.x) == 2^N
        @test native.x ≈ realspace_grid(2^N, L; offset)
        @test native.method == :dft_mpo
        @test native.values ≈ fourier_basis(2, native.x, L) atol=1e-12

        realspace = fourier_to_realspace_mps(state, sites, L; offset)
        @test realspace_mps_to_vec(realspace, sites) ≈ native.values atol=1e-12
    end

    # Normalization acts on the state, not on every tensor.
    scaled = mps_to_realspace(3.0 * state, sites, L; normalize_coefficients=true)
    @test sum(abs2, scaled.coefficients) ≈ 1 atol=1e-12
    @test scaled.values ≈ fourier_basis(2, scaled.x, L) atol=1e-12

    @test_throws ArgumentError mps_to_realspace(
        state,
        sites,
        L;
        x=selected_x,
        npoints=53,
    )

    svg = plot_realspace(x, direct; quantity=:abs, title="single Fourier mode")
    @test startswith(svg, "<svg")
    @test occursin("single Fourier mode", svg)
    @test occursin("polyline", svg)
end
