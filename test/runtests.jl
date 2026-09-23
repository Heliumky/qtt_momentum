using Test

# Tests exercise the implementation entry point in Juliatools directly.
include(joinpath(@__DIR__, "..", "Juliatools", "QTTFourier.jl"))
using .QTTFourier

@testset "QTTFourier" begin
    include("test_elementary.jl")
    include("test_diff_mpo.jl")
    include("test_k_to_r.jl")
    include("test_dmrg_engine.jl")
    include("test_multi_dim.jl")
    include("test_multi_dim_potential.jl")
end
