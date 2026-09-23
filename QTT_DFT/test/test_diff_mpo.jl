using LinearAlgebra

@testset "differential MPO" begin
    N, L = 5, 3.5
    modes = freqs(N)

    for order in 0:5
        tensors = differential_mpo(order, N, L)
        operator = mpo_to_dense(tensors)
        expected = Diagonal(ComplexF64[(im * pi * mode / L)^order for mode in modes])
        @test operator ≈ expected rtol=1e-12 atol=2e-9
        @test maximum(size(tensor, 4) for tensor in tensors[1:end-1]) <= order + 1
    end

    kinetic = mpo_to_dense(kinetic_mpo(N, L))
    @test kinetic ≈ Diagonal(0.5 .* (pi .* modes ./ L) .^ 2) atol=1e-11
end
