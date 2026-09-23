using LinearAlgebra

@testset "index conventions and real-space grids" begin
    @test freqs(3) == [0, 1, 2, 3, -4, -3, -2, -1]
    @test sigma_to_n([2, 1, 2]) == 1 - 4
    @test sigma_to_grid_index([2, 1, 1]) == 4
    @test sigma_to_grid_index([1, 1, 2]) == 1
    for index in 0:15
        @test sigma_to_grid_index(grid_index_to_sigma(index, 4)) == index
    end

    x = realspace_grid(8, 2.0)
    @test x ≈ -2.0 .+ (0:7) .* 0.5
    @test 0.0 in x
    centred = realspace_grid(8, 2.0; offset=0.5)
    @test centred ≈ x .+ 0.25
    @test !(0.0 in centred)
    @test grid_coordinate([2, 1, 1], 2.0) ≈ x[5]
    @test grid_coordinate([2, 1, 1], 2.0; offset=0.5) ≈ centred[5]
    @test nearest_grid_sigma(0.3, 3, 2.0) == grid_index_to_sigma(5, 3)
    @test_throws ArgumentError realspace_grid(8, 2.0; offset=1.0)
end

@testset "TCI of real-space functions" begin
    N, L = 9, 3.0
    x = realspace_grid(2^N, L)
    order = grid_register_order(N)

    # A degree-m polynomial has quantics rank m+1 and is reproduced exactly.
    for m in 0:4
        fit = power_potential_mps(m, N, L; scale=0.7, tolerance=1e-13, return_diagnostics=true)
        @test maximum(mps_bond_dims(fit.tensors); init=1) <= m + 1
        values = vec(mps_to_dense(fit.tensors))
        grid_values = similar(values)
        grid_values[order] = values
        @test grid_values ≈ 0.7 .* x .^ m rtol=1e-11 atol=1e-11
        @test fit.diagnostics.global_error_estimate < 1e-11
        @test fit.diagnostics.target_evaluations < 2^N
    end

    # A narrow peak far from the default pivot is invisible to TCI unless a
    # pivot point near it is supplied.
    peak(x) = exp(-((x - 1.3) / 0.02)^2)
    @test_throws ArgumentError realspace_function_mps(peak, N, L; tolerance=1e-8)
    fit = realspace_function_mps(peak, N, L; pivot_points=[1.3], tolerance=1e-8,
                                 return_diagnostics=true)
    values = vec(mps_to_dense(fit.tensors))
    grid_values = similar(values)
    grid_values[order] = values
    @test grid_values ≈ peak.(x) atol=1e-7
    @test length(fit.diagnostics.initial_pivots) >= 1

    # Two separated peaks: both seeded, both resolved.
    twin(x) = exp(-((x + 1.5) / 0.05)^2) + 0.5exp(-((x - 1.5) / 0.05)^2)
    fit = realspace_function_mps(twin, N, L; pivot_points=[-1.5, 1.5], tolerance=1e-10)
    values = vec(mps_to_dense(fit))
    grid_values = similar(values)
    grid_values[order] = values
    @test grid_values ≈ twin.(x) atol=1e-9

    # Cached and threaded evaluators build the same interpolant.
    smooth(x) = sin(2x) / (1 + x^2)
    cached = realspace_function_mps(smooth, N, L; tolerance=1e-12, return_diagnostics=true)
    threaded = realspace_function_mps(smooth, N, L; tolerance=1e-12, threaded=true,
                                      return_diagnostics=true)
    @test vec(mps_to_dense(cached.tensors)) ≈ vec(mps_to_dense(threaded.tensors)) atol=1e-10
    @test cached.diagnostics.target_evaluations <= threaded.diagnostics.target_evaluations

    diagonal = mps_to_diagonal_mpo(cached.tensors)
    @test all(tensor -> size(tensor, 2) == size(tensor, 3) == 2, diagonal)
    dense = mpo_to_dense(diagonal)
    @test dense ≈ Diagonal(vec(mps_to_dense(cached.tensors)))
end
